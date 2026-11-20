"""Evaluacion de robustez del sistema EBGM frente a los supuestos de ataque que
pide el guion: que un evaluador se salte alguna fase del sistema, consiga leer
la base de datos o consiga escribir en ella. El objetivo es que el grupo conozca
sus propias debilidades y proponga mitigaciones.

Cada prueba registra: el supuesto (que parte del sistema controla el evaluador),
la metrica observada y la lectura defensiva. Salida: resultados/ataques.json.

Uso: python scripts/ataques.py
"""
import json
import os
import shutil
import sys
import tempfile
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
from ebgm.sistema import SistemaEBGM, codificar_template  # noqa: E402
from ebgm.normalizacion import cargar_imagen  # noqa: E402
from ebgm.detector import elegir_cara  # noqa: E402
from inscribir_dataset import id_desde_fichero  # noqa: E402
from inversion import OperadorJets, invertir, a_uint8  # noqa: E402

BD = os.path.join(RAIZ, "datos", "bd.csv")
DS = os.path.join(RAIZ, "datos", "dataset")
FIG = os.path.join(RAIZ, "resultados", "figuras")
os.makedirs(FIG, exist_ok=True)
FOTOS = {id_desde_fichero(f): os.path.join(DS, f) for f in sorted(os.listdir(DS))}
RES = {}


def sistema_temporal(clave=None):
    d = tempfile.mkdtemp()
    ruta = os.path.join(d, "bd.csv")
    shutil.copy(BD, ruta)
    return SistemaEBGM(ruta, clave_hmac=clave), ruta


def registrar(nombre, supuesto, lectura, datos):
    RES[nombre] = {"supuesto": supuesto, "lectura_defensiva": lectura, "datos": datos}
    print(f"\n### {nombre}\n{supuesto}\n-> {json.dumps(datos, ensure_ascii=False)[:600]}")


def recorte_cara(pid, S):
    img = cargar_imagen(FOTOS[pid])
    d = elegir_cara(S.detector.detectar(img))
    x, y, w, h = map(int, d["caja"])
    mx, my = int(0.12 * w), int(0.12 * h)
    return img[max(0, y - my):y + h + my, max(0, x - mx):x + w + mx]


# -- P1. Entradas que no son caras por la ruta sin detector -----------------
def p1_entradas_no_faciales(S):
    rng = np.random.default_rng(1)
    entradas = {
        "gris_uniforme": np.full((200, 200, 3), 128, np.uint8),
        "ruido_uniforme": rng.integers(0, 256, (200, 200, 3), dtype=np.uint8),
        "degradado": np.tile(np.linspace(0, 255, 200, np.uint8)[None, :, None], (200, 1, 3)),
        "recorte_de_fondo": cargar_imagen(FOTOS["jackie_chan"])[:220, :220],
    }
    cv2.imwrite(os.path.join(FIG, "no_caras.png"),
                np.hstack([cv2.resize(v, (150, 150)) for v in entradas.values()]))
    fuera = []
    caras_sf = [S.ultima_similitud_fbg for pid in list(FOTOS)[:20]
                if (S.extraer_template_sd(recorte_cara(pid, S)) is not None)]
    for k, img in entradas.items():
        try:
            idn, sc = S.identificar_sd(img)
            fuera.append((k, idn, round(sc, 3), round(S.ultima_similitud_fbg, 3)))
        except ValueError as e:
            fuera.append((k, f"rechazada: {e}", None, round(S.ultima_similitud_fbg, 3)))
    datos = dict(no_caras=[dict(entrada=k, identificado=i, score=s, S_FBG=f) for k, i, s, f in fuera],
                 S_FBG_caras_reales=dict(min=round(min(caras_sf), 3),
                                         mediana=round(float(np.median(caras_sf)), 3)))
    registrar("P1_entradas_no_faciales_ruta_sd",
              "El evaluador usa identificar_sd/verificar_sd (sin detector) con imagenes que no son caras.",
              "La ruta sin detector no comprueba que haya una cara: hay que validar la calidad del ajuste "
              "(S_FBG) y rechazar por debajo de un umbral; con detector, estas entradas se rechazan solas.",
              datos)
    return datos


# -- P2. Reenvio de un template leido de la BD ------------------------------
def p2_reenvio_template(S):
    acc = sum(S.verificar_t(i, t)[0] for i, t in S.bd.items())
    idn = sum(S.identificar_t(t)[0] == i for i, t in S.bd.items())
    registrar("P2_reenvio_de_template",
              "El evaluador consigue LEER la BD y reenvia por verificar_t/identificar_t el template de la victima.",
              "La comparacion de templates no distingue 'misma persona' de 'mismo fichero'. Mitigacion: canal "
              "autenticado hacia el comparador y deteccion de duplicados exactos; la verificacion no debe aceptar "
              "un template identico bit a bit al almacenado.",
              dict(verificaciones=f"{acc}/{len(S.bd)}", identificaciones=f"{idn}/{len(S.bd)}", score=1.0))


# -- P3. Template sintetico sin foto de la victima --------------------------
def p3_template_generico(S):
    T = np.stack(list(S.bd.values()))
    cand = {"media_bd": T.mean(0), "mediana_bd": np.median(T, 0),
            "jets_medios_fbg": np.abs(S.fbg.jets).mean(0).ravel()}
    out = {}
    for k, t in cand.items():
        p = S.puntuaciones(t)
        out[k] = dict(mejor=round(max(p.values()),3), verifica_a=sum(v >= S.umbral for v in p.values()))
    registrar("P3_template_generico",
              "El evaluador fabrica un template 'promedio' para intentar colarse como cualquier inscrito sin su foto.",
              "El promedio de magnitudes queda lejos del umbral: no existe un 'template maestro' trivial. "
              "La defensa real sigue siendo el umbral y la unicidad de los jets por persona.",
              out)


# -- P4. Busqueda por consultas (solo verificar_t devuelve la puntuacion) ----
def p4_busqueda_por_consultas(S, n_victimas=8, presupuesto=4000):
    rng = np.random.default_rng(7)
    victimas = [v for v in S.bd][:n_victimas]
    partida = np.abs(S.fbg.jets).mean(0).ravel()     # el atacante NO parte de ninguna foto real
    consultas = []
    for v in victimas:
        t = partida.copy().reshape(33, 40)
        mejor = S.verificar_t(v, t.ravel())[1]
        q = 1
        while mejor < S.umbral and q < presupuesto:
            c = t.copy()
            c[rng.integers(33)] *= np.exp(rng.normal(0, 0.3, 40))
            sc = S.verificar_t(v, c.ravel())[1]
            q += 1
            if sc > mejor:
                t, mejor = c, sc
        consultas.append(q if mejor >= S.umbral else None)
    logrados = [q for q in consultas if q]
    registrar("P4_busqueda_por_consultas",
              "El evaluador puede llamar a verificar_t sin limite y usa la PUNTUACION devuelta para 'escalar la colina'.",
              "Devolver la puntuacion permite reconstruir un template que verifica sin foto. Mitigaciones: no "
              "exponer el score (solo aceptar/rechazar), limitar intentos y registrar rafagas de consultas.",
              dict(exitos=f"{len(logrados)}/{n_victimas}",
                   consultas_mediana=int(np.median(logrados)) if logrados else None))


# -- P5. Reconstruccion de imagen a partir del template (inversion) ---------
def p5_inversion_template(S, pid="keanu_reeves"):
    # cara media a partir de OTRAS personas (el atacante no necesita la BD de la victima)
    otras = [S.ultima_canonica for p in list(FOTOS)[:20] if p != pid
             and (S.extraer_template(FOTOS[p]) is not None)]
    from ebgm.normalizacion import fotometria
    mu = np.mean([fotometria(c) for c in otras], 0)
    S.extraer_template(FOTOS[pid]); orig = S.ultima_canonica
    pos = S.ultimo_grafo
    t = S.bd[pid].reshape(33, 40)                       # magnitudes leidas de la BD
    # recuperacion de fase por iteracion de proyecciones y despues inversion lineal
    op = OperadorJets(pos)
    rec = mu.copy()
    for _ in range(10):
        ph = np.angle(op.directo(rec))
        rec = invertir(pos, t * np.exp(1j * ph), mu, l1=2.0, l2=0.1, iters=60)
    img = a_uint8(rec)
    cv2.imwrite(os.path.join(FIG, "inversion_keanu.png"),
                np.hstack([cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), (180, 180)),
                           cv2.resize(a_uint8(mu), (180, 180)), cv2.resize(img, (180, 180))]))
    v_sd = S.verificar_sd(pid, img)
    registrar("P5_inversion_de_template",
              "El evaluador LEE la BD (solo magnitudes) y reconstruye una imagen de cara que presenta por la ruta con/sin detector.",
              "Aun guardando solo magnitudes, el template es invertible porque la respuesta de Gabor es lineal: "
              "se recupera una cara reconocible. Mitigaciones: cifrar la BD, plantilla protegida/'cancelable' "
              "(proyeccion no invertible) y deteccion de vivacidad para que una imagen plana no baste.",
              dict(verificar_sd=[bool(v_sd[0]), round(v_sd[1], 3)],
                   baseline_cara_media=round(S.verificar_sd(pid, a_uint8(mu))[1], 3)))


# -- P6. Escritura en la BD -------------------------------------------------
def p6_escritura_bd():
    S, ruta = sistema_temporal()
    # 6a. sustituir el template de una victima por el del atacante
    import csv
    filas = list(csv.DictReader(open(ruta, encoding="utf-8")))
    t_atacante = S.bd["diego_ramon_jimenez_salazar_el_cigala"]
    for f in filas:
        if f["id"] == "keanu_reeves":
            f["template"] = codificar_template(t_atacante)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "template", "hmac"], quoting=csv.QUOTE_ALL)
        w.writeheader(); w.writerows(filas)
    S2 = SistemaEBGM(ruta)                              # sin clave: no detecta la manipulacion
    suplanta = S2.verificar("keanu_reeves", FOTOS["diego_ramon_jimenez_salazar_el_cigala"])
    # 6b. el mismo CSV con verificacion HMAC activada
    S3 = SistemaEBGM(ruta, clave_hmac="clave-secreta-del-sistema")
    registrar("P6_escritura_en_bd",
              "El evaluador consigue ESCRIBIR en el CSV y cambia el template de la victima por el suyo.",
              "Sin integridad, quien escribe el CSV decide quien es quien. Con HMAC por fila y clave fuera de la "
              "BD, las filas manipuladas se rechazan al cargar; ademas conviene control de acceso y registro de cambios.",
              dict(sin_hmac_suplanta=[bool(suplanta[0]), round(suplanta[1], 3)],
                   con_hmac_filas_rechazadas=len(S3.filas_rechazadas),
                   con_hmac_keanu_en_bd="keanu_reeves" in S3.bd))


# -- P7. Entradas malformadas (robustez de la API) --------------------------
def p7_entradas_malformadas(S):
    casos = {}
    pruebas = {
        "id_con_salto_de_linea": lambda: S.inscribir_t("a\nb", S.bd["adele"]),
        "id_vacio": lambda: S.inscribir_t("", S.bd["adele"]),
        "template_corto": lambda: S.verificar_t("adele", np.zeros(10, np.float32)),
        "template_nan": lambda: S.verificar_t("adele", np.full(1320, np.nan, np.float32)),
        "template_negativo": lambda: S.verificar_t("adele", -np.ones(1320, np.float32)),
        "id_inexistente": lambda: S.verificar_t("no_existe", S.bd["adele"]),
        "reinscribir_existente": lambda: S.inscribir_t("adele", S.bd["adele"]),
    }
    for k, fn in pruebas.items():
        try:
            casos[k] = f"aceptado -> {fn()}"
        except ValueError as e:
            casos[k] = f"rechazado: {e}"
    registrar("P7_entradas_malformadas",
              "El evaluador envia identificadores y templates no validos para provocar errores o inyecciones en el CSV.",
              "La API valida id (lista blanca de caracteres) y template (longitud, finitud, rango) antes de tocar "
              "la BD, de modo que no se puede inyectar una fila corrupta ni un id con saltos de linea.",
              casos)


def main():
    S = SistemaEBGM(BD)
    p1_entradas_no_faciales(S)
    p2_reenvio_template(S)
    p3_template_generico(S)
    p4_busqueda_por_consultas(S)
    p5_inversion_template(S)
    p6_escritura_bd()
    p7_entradas_malformadas(S)
    json.dump(RES, open(os.path.join(RAIZ, "resultados", "ataques.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n{len(RES)} pruebas -> resultados/ataques.json")


if __name__ == "__main__":
    main()
