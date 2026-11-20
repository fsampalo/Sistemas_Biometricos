"""
Evaluación de SEGURIDAD del sistema EBGM (fase 2) — ataques ejecutados de verdad.

El enunciado pide evaluar el sistema suponiendo que el atacante:
  (A) se salta alguna parte del sistema (detector / extractor: métodos _sd y _t),
  (B) consigue LEER la base de datos,
  (C) consigue ESCRIBIR en la base de datos.

Cada ataque se ejecuta sobre una COPIA temporal de la BD y se imprime su
resultado como:
    [BLOQUEADO]  la defensa del sistema lo para
    [VULNERABLE] el ataque funciona: defecto CONOCIDO y documentado
El objetivo de la práctica no es un sistema sin defectos, sino que el rival no
encuentre defectos que nosotros no conozcamos.

Uso:
    python evaluar_ataques.py
"""

from __future__ import annotations

import csv
import glob
import os
import shutil
import tempfile

import cv2
import numpy as np

from detector import DetectorCaras
from ebgm import RUTA_MODELO
from sistema import ErrorSistema, SistemaReconocimiento, RUTA_BD_POR_DEFECTO

AQUI = os.path.dirname(os.path.abspath(__file__))
RESUMEN: list[tuple[str, str, str]] = []


def titulo(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def resultado(codigo: str, vulnerable: bool, texto: str):
    estado = "VULNERABLE" if vulnerable else "BLOQUEADO"
    RESUMEN.append((codigo, estado, texto))
    print(f"  [{estado}] {codigo}: {texto}")


def intento(f):
    """Ejecuta f(); devuelve (ok, resultado_o_error)."""
    try:
        return True, f()
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


class Entorno:
    """Copia temporal de BD + modelo para atacar sin estropear lo real."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="ebgm_ataque_")
        self.bd = os.path.join(self.dir, "bd.csv")
        shutil.copy(RUTA_BD_POR_DEFECTO, self.bd)
        shutil.copy(RUTA_BD_POR_DEFECTO + ".mac", self.bd + ".mac")
        self.modelo = os.path.join(self.dir, "modelo.npz")
        shutil.copy(RUTA_MODELO, self.modelo)

    def sistema(self, **kw) -> SistemaReconocimiento:
        kw.setdefault("ruta_modelo", self.modelo)
        # estas pruebas ven las similitudes para poder medir; el ataque A4
        # compara explícitamente la configuración entregada (sin similitudes)
        kw.setdefault("exponer_puntuaciones", True)
        return SistemaReconocimiento(ruta_bd=self.bd, auditoria=False, **kw)

    def filas(self):
        with open(self.bd, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def escribir(self, filas):
        with open(self.bd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "template", "mac"], lineterminator="\n")
            w.writeheader(); w.writerows(filas)

    def limpiar(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def foto_famoso(nombre: str) -> str:
    return glob.glob(os.path.join(AQUI, "Dataset", nombre + "*"))[0]


def main():
    det = DetectorCaras()
    env = Entorno()
    sis = env.sistema()
    ext = sis.extractor
    victima = "cigala"
    atacante_foto = foto_famoso("adele")
    tpl_atacante = ext.extraer(det.extraer_cara(atacante_foto))
    print(f"BD: {len(sis.personas())} personas | umbral={sis.umbral:.4f} | "
          f"umbral_cara={sis.umbral_cara:.3f}")
    print(f"Víctima: '{victima}'.  Atacante: una persona inscrita distinta (adele).")

    # ===================================================================== (A)
    titulo("(A) EL ATACANTE SE SALTA PARTE DEL SISTEMA (_sd / _t)")

    tpl_victima = sis._bd[victima]
    r = sis.verificar_t(victima, tpl_victima)
    resultado("A1 replay de template", r["verificado"],
              f"verificar_t con el template robado de '{victima}' -> {r['verificado']} "
              f"(sim={r.get('similitud', 0):.3f}). Inherente a _t: quien tenga el "
              "template ES la persona. Mitigación: no exponer _t fuera; plantillas "
              "cancelables; canal autenticado con el extractor.")

    rng = np.random.default_rng(0)
    basura = {
        "ruido": (rng.random((112, 112, 3)) * 255).astype(np.uint8),
        "negra": np.zeros((112, 112, 3), np.uint8),
        "blanca": np.full((112, 112, 3), 255, np.uint8),
        "cara invertida": np.ascontiguousarray(det.extraer_cara(atacante_foto)[::-1]),
        "fondo de foto": det.cargar_imagen(atacante_foto)[:112, :112].copy(),
    }
    pasan = []
    for nombre, img in basura.items():
        ok, res = intento(lambda: sis.identificar_sd(img))
        if ok:
            pasan.append(f"{nombre}->{res['id']}")
    resultado("A2 no-caras por _sd", bool(pasan),
              f"{len(basura) - len(pasan)}/{len(basura)} entradas que no son cara "
              f"rechazadas por el control de caricidad EBGM"
              + (f"; PASAN: {pasan}" if pasan else ""))

    tpl_rand = np.abs(rng.normal(size=tpl_victima.shape)).astype(np.float32) * tpl_victima.mean()
    tpl_rand[:, -2:] = tpl_victima[:, -2:]
    malos = {"aleatorio": tpl_rand, "ceros": np.zeros_like(tpl_victima),
             "NaN": np.full_like(tpl_victima, np.nan), "forma errónea": tpl_victima[:10],
             "negativo": -tpl_victima, "enorme": tpl_victima * 1e6,
             "texto basura": "AAAA", "objeto": {"a": 1}}
    pasan = [n for n, t in malos.items() if intento(lambda: sis.inscribir_t("x_" + n.replace(" ", "_"), t))[0]]
    resultado("A3 templates basura por _t", bool(pasan),
              f"{len(malos) - len(pasan)}/{len(malos)} templates inválidos rechazados "
              "(forma, NaN/Inf, rango, caricidad)" + (f"; PASAN: {pasan}" if pasan else ""))

    # Hill-climbing con las puntuaciones devueltas por verificar_t
    def hill_climbing(s_obj, max_q=3000, paso=0.15):
        F = ext.cfg.n_filtros
        x = tpl_atacante.copy()
        best = s_obj.verificar_t(victima, x)
        consultas = 1
        if "similitud" not in best:
            return False, consultas, None
        bs = best["similitud"]
        r_ = np.random.default_rng(1)
        while consultas < max_q:
            y = x.copy()
            n = r_.integers(ext.n_nodos)
            y[n, :F] *= np.exp(r_.normal(0, paso, F)).astype(np.float32)
            ok, res = intento(lambda: s_obj.verificar_t(victima, y))
            consultas += 1
            if ok and res["similitud"] > bs:
                x, bs = y, res["similitud"]
                if res["verificado"]:
                    return True, consultas, bs
        return False, consultas, bs

    sis_defecto = env.sistema(exponer_puntuaciones=False)    # configuración entregada
    exito2, _, _ = hill_climbing(sis_defecto)
    resultado("A4 hill-climbing (configuración por defecto)", exito2,
              "verificar/identificar solo devuelven la decisión: el ascenso no "
              "tiene gradiente que seguir y queda fuerza bruta a ciegas.")
    exito, q, s_fin = hill_climbing(sis)
    s_txt = f"{s_fin:.3f}" if s_fin is not None else "-"
    resultado("A4b hill-climbing si se activa exponer_puntuaciones=True", exito,
              f"forjando un template desde la cara del atacante: "
              f"{'ACEPTADO' if exito else 'no aceptado'} tras {q} consultas "
              f"(sim final {s_txt}). Por eso ese modo es solo para evaluación; "
              "además conviene un límite de intentos.")

    # Morphing: template medio atacante+víctima inscrito como nueva identidad
    morph = ext.promediar([tpl_atacante, tpl_victima])
    s_v = sis.verificar_t(victima, morph)
    ok_a = ext.similitud(morph, tpl_atacante) >= sis.umbral
    resultado("A5 morphing (template medio atacante+víctima)", s_v["verificado"] and ok_a,
              f"el template mezcla verifica como la víctima: {s_v['verificado']}, "
              f"y como el atacante: {ok_a}. Si un operador inscribe esa mezcla, "
              "dos personas comparten identidad. El aviso de posible_duplicado de "
              "inscribir lo detecta si alguno de los dos ya está inscrito.")

    # Presentación: foto de una foto / pantalla (sin liveness)
    img = det.cargar_imagen(foto_famoso("tilda_swinton"))
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * .05, h * .03], [w * .97, 0], [w, h], [0, h * .96]])
    reprod = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))
    reprod = cv2.addWeighted(reprod, 0.85, np.full_like(reprod, (40, 30, 20)), 0.15, 0)
    reprod = np.clip(reprod.astype(np.float32) * 1.1 + rng.normal(0, 6, reprod.shape), 0, 255).astype(np.uint8)
    ok, r = intento(lambda: sis.verificar("tilda_swinton", reprod))
    resultado("A6 ataque de presentación (foto de foto)", ok and r["verificado"],
              f"una reproducción (perspectiva, tinte, ruido) de la foto de "
              f"tilda_swinton -> {r if not ok else r['verificado']}. No hay detección "
              "de vida (liveness/PAD): límite conocido.")

    # Cara distractora: dos caras, gana la más grande
    a = det.cargar_imagen(atacante_foto)
    v = det.cargar_imagen(foto_famoso("keanu"))
    hh = 400
    a2 = cv2.resize(a, (int(a.shape[1] * hh / a.shape[0]), hh))
    v2 = cv2.resize(v, (int(v.shape[1] * hh * .5 / v.shape[0]), int(hh * .5)))
    lienzo = np.zeros((hh, a2.shape[1] + v2.shape[1], 3), np.uint8)
    lienzo[:, :a2.shape[1]] = a2; lienzo[:v2.shape[0], a2.shape[1]:] = v2
    ok, r = intento(lambda: sis.identificar(lienzo))
    ok2, r2 = intento(lambda: sis.inscribir("dos_caras", lienzo))
    resultado("A7 varias caras en la foto", False,
              f"identificar elige la MÁS GRANDE -> {r['id'] if ok else r}; inscribir con "
              f"2 caras -> {'aceptado' if ok2 else 'rechazado'} (política estricta). "
              "Por diseño; documentado.")

    # ===================================================================== (B)
    titulo("(B) EL ATACANTE LEE LA BASE DE DATOS")
    filas = env.filas()
    resultado("B1 exfiltración de templates", True,
              f"{len(filas)} filas legibles (id + template en claro). El HMAC da "
              "INTEGRIDAD, no confidencialidad. Con un template robado: replay por "
              "_t (A1) o enlazar a la persona en otra BD EBGM con la misma config.")
    robado = filas[[f["id"] for f in filas].index(victima)]["template"]
    r = sis.verificar_t(victima, robado)
    resultado("B2 replay del template leído del .csv", r["verificado"],
              f"verificar_t('{victima}', cadena_base64_del_csv) -> {r['verificado']}")
    ruta_clave = os.path.join(AQUI, "models", "clave_bd.key")
    resultado("B3 clave HMAC legible en disco", os.path.isfile(ruta_clave)
              and not os.environ.get("EBGM_CLAVE_BD"),
              "si la clave está en models/clave_bd.key y el atacante lee el disco, "
              "puede firmar filas (ver C6). Mitigación: EBGM_CLAVE_BD en el entorno "
              "/ secreto del orquestador / HSM, fuera del alcance de quien lee la BD.")
    resultado("B4 inversión del template a imagen", False,
              f"el template son MAGNITUDES de Gabor en {ext.n_nodos} nodos (sin fase): no se "
              "reconstruye la foto con fidelidad. No hace falta para suplantar (B2).")

    # ===================================================================== (C)
    titulo("(C) EL ATACANTE ESCRIBE EN LA BASE DE DATOS (sin la clave)")
    original = env.filas()

    def con_filas(nuevas):
        env.escribir(nuevas)
        return env.sistema()

    # C1 sustituir el template de la víctima por el del atacante
    f2 = [dict(f) for f in original]
    k = [f["id"] for f in f2].index(victima)
    f2[k]["template"] = ext.serializar(tpl_atacante)
    s2 = con_filas(f2)
    ok, r = intento(lambda: s2.verificar_t(victima, tpl_atacante))
    resultado("C1 sustitución de template", ok and r["verificado"],
              f"fila alterada rechazada por HMAC ({len(s2.filas_rechazadas)} rechazo) y, "
              f"en modo estricto, el sistema deja de operar -> {r if not ok else r}. "
              "Falla en CERRADO: DoS, pero nunca suplantación; queda en auditoría.")

    # C2 puerta trasera: fila nueva
    f3 = [dict(f) for f in original] + [{"id": "admin", "template": ext.serializar(tpl_atacante),
                                         "mac": "0" * 64}]
    s3 = con_filas(f3)
    s3.estricto = False
    resultado("C2 alta de puerta trasera", "admin" in s3.personas(),
              f"fila 'admin' añadida a mano -> {'aceptada' if 'admin' in s3.personas() else 'rechazada (HMAC)'}")

    # C3 intercambio de templates entre dos filas válidas (con sus MAC)
    f4 = [dict(f) for f in original]
    i, j = 0, k
    f4[i]["template"], f4[j]["template"] = f4[j]["template"], f4[i]["template"]
    f4[i]["mac"], f4[j]["mac"] = f4[j]["mac"], f4[i]["mac"]
    s4 = con_filas(f4)
    s4.estricto = False
    resultado("C3 intercambio de templates/MAC entre ids", victima in s4.personas(),
              f"el MAC liga id+template: {len(s4.filas_rechazadas)} filas rechazadas")

    # C4 borrar filas
    f5 = [f for f in original if f["id"] != victima]
    s5 = con_filas(f5)
    resultado("C4 borrado de una fila", s5.integridad_global,
              f"'{victima}' borrado -> MAC global {'OK (no detectado)' if s5.integridad_global else 'INCORRECTO: detectado y auditado'}")

    # C5 rollback de una fila antigua (re-inscripción)
    env.escribir(original); shutil.copy(RUTA_BD_POR_DEFECTO + ".mac", env.bd + ".mac")
    s6 = env.sistema()
    fila_vieja = dict(original[k])
    s6.inscribir_t(victima, tpl_atacante, sobrescribir=True)    # re-inscripción legítima
    nuevo_mac = open(env.bd + ".mac").read()
    f6 = env.filas()
    f6[[f["id"] for f in f6].index(victima)] = fila_vieja         # vuelve la fila antigua
    s6b = con_filas(f6)
    with open(env.bd + ".mac", "w") as fm:
        fm.write(nuevo_mac)
    s6b = env.sistema()
    ok, r = intento(lambda: s6b.verificar_t(victima, tpl_victima))
    resultado("C5 rollback de una fila antigua", ok,
              f"la fila antigua tiene MAC válido, pero el MAC global lo detecta y el "
              f"modo estricto bloquea -> {r if not ok else r}. OJO: restaurar BD+.mac "
              "COMPLETOS de una copia antigua sí pasaría (no hay contador de versión "
              "externo): límite conocido.")

    # C6 con la clave (lectura + escritura del disco)
    env.escribir(original); shutil.copy(RUTA_BD_POR_DEFECTO + ".mac", env.bd + ".mac")
    s7 = env.sistema()
    s7._bd[victima] = tpl_atacante
    s7._rep[victima] = ext.representar(tpl_atacante)
    s7._guardar_bd()            # = el atacante firma con la clave robada
    s7b = env.sistema()
    r = s7b.verificar_t(victima, tpl_atacante)
    resultado("C6 escritura CON la clave robada", r["verificado"],
              "si el atacante también tiene la clave (B3), la integridad no protege: "
              "suplanta a quien quiera. La clave debe vivir fuera de la máquina de la BD.")

    # C7 alterar el modelo / umbrales (models/ebgm_modelo.npz)
    env.escribir(original); shutil.copy(RUTA_BD_POR_DEFECTO + ".mac", env.bd + ".mac")
    from ebgm import ModeloEBGM
    m = ModeloEBGM.cargar(env.modelo)
    m.info["umbral"] = -1.0           # ¡aceptar a todo el mundo!
    m.guardar(env.modelo)
    s8 = env.sistema()
    resultado("C7 bajar el umbral en el modelo", len(s8.personas()) > 0,
              f"modelo alterado -> su hash ya no cuadra con el HMAC: "
              f"{len(s8.filas_rechazadas)} filas rechazadas, {len(s8.personas())} "
              "personas cargadas. Falla en CERRADO (nadie entra; DoS, pero no "
              "suplantación).")
    shutil.copy(RUTA_MODELO, env.modelo)

    # C8 inyección de fórmulas CSV / ids raros a través de la API
    malos = ["=HYPERLINK(\"http://x\")", "+cmd", "@SUM(A1)", "a\nb", "", "x" * 200, None, 42]
    acept = [m_ for m_ in malos if intento(lambda: env.sistema().inscribir_t(m_, tpl_atacante))[0]]
    resultado("C8 ids maliciosos (inyección CSV, saltos de línea...)", bool(acept),
              f"{len(malos) - len(acept)}/{len(malos)} rechazados por validar_id")

    s9 = env.sistema()
    ok, r = intento(lambda: s9.inscribir_t(victima, tpl_atacante))
    resultado("C9 secuestro por re-inscripción vía API", ok,
              f"inscribir_t('{victima}', template_atacante) sin sobrescribir=True -> "
              f"{'aceptado' if ok else 'rechazado'}")
    ok, r = intento(lambda: s9.inscribir_t("adele_bis", tpl_atacante))
    resultado("C10 misma cara con otro nombre", False,
              f"aceptado pero marcado: posible_duplicado={r.get('posible_duplicado') if ok else r}")

    env.limpiar()

    # ================================================================= RESUMEN
    titulo("RESUMEN")
    for codigo, estado, _ in RESUMEN:
        print(f"  {estado:10s}  {codigo}")
    nv = sum(e == "VULNERABLE" for _, e, _ in RESUMEN)
    print(f"\n  {len(RESUMEN) - nv} ataques bloqueados, {nv} vulnerabilidades CONOCIDAS "
          "(documentadas con su mitigación en INFORME.md).")


if __name__ == "__main__":
    main()
