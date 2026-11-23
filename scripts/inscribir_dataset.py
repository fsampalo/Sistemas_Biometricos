"""
Crea el sistema listo para usar:

  1. Entrena el MODELO EBGM (Face Bunch Graph + estadísticos de normalización)
     con la cohorte de los 49 famosos comunes -> models/ebgm_modelo.npz
  2. Inscribe a las 50 personas conocidas en la BD (.csv):
       - los 49 famosos del docente (Dataset/, 1 foto cada uno)
       - nuestro famoso: EL CIGALA, también con UNA sola foto: la más
         foto de fotos_cigala/ (si hubiera varias, la más representativa: la
         que más se parece de media a las demás). Sus demás fotos están en
         fotos_cigala_test/ y solo se usan para evaluar.
  3. Calibra los umbrales -> models/calibracion.json
       - umbral de decisión: el que da un FAR objetivo (por defecto 1 %) en los
         pares impostores disponibles (famoso vs famoso, cigala vs famosos).
       - umbral de "caricidad": separa caras reales NO vistas por el modelo
         (cada famoso medido sin su propia foto en el bunch, y la foto del
         cigala) de entradas que no son caras (ruido, imágenes
         planas, texturas, caras invertidas...).

NUNCA usa fotos_cigala_test/ (reservado para la evaluación final).

Uso:
    python scripts/inscribir_dataset.py
    python scripts/inscribir_dataset.py --far-objetivo 0.005
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import re

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raíz del repo

from reconocimiento.detector import DetectorCaras, ErrorDeteccion
from reconocimiento.ebgm import ExtractorEBGM, RUTA_MODELO
from reconocimiento.rutas import DIR_CIGALA, DIR_DATASET, RAIZ
from reconocimiento.sistema import (RUTA_BD_POR_DEFECTO, RUTA_CALIBRACION, SistemaReconocimiento)


def id_desde_fichero(ruta: str) -> str:
    """adele_2016.jpg -> 'adele'. Quita la extensión y el año final."""
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    return re.sub(r"_\d{4}$", "", nombre)


def caras_de(det: DetectorCaras, rutas: list[str]) -> tuple[list, list, list]:
    caras, nombres, fallos = [], [], []
    for ruta in rutas:
        try:
            caras.append(det.extraer_cara(ruta)); nombres.append(ruta)
        except ErrorDeteccion as e:
            fallos.append((os.path.basename(ruta), str(e)))
    return caras, nombres, fallos


def no_caras(caras_fam: list[np.ndarray], fotos: list[np.ndarray]) -> list[tuple[str, np.ndarray]]:
    """Entradas de 112x112 que NO son caras (categoría, imagen), para calibrar."""
    rng = np.random.default_rng(123)
    out = [("plana", np.full((112, 112, 3), v, np.uint8)) for v in (0, 128, 255)]
    out += [("ruido", (rng.random((112, 112, 3)) * 255).astype(np.uint8)) for _ in range(5)]
    out += [("ruido suavizado", cv2.GaussianBlur(
        (rng.random((112, 112, 3)) * 255).astype(np.uint8), (0, 0), s)) for s in (2, 4, 8)]
    g = np.linspace(0, 255, 112, dtype=np.uint8)
    out += [("gradiente", np.dstack([np.tile(g, (112, 1))] * 3)),
            ("gradiente", np.dstack([np.tile(g[:, None], (1, 112))] * 3))]
    xs = np.arange(112)
    for f in (4, 8, 16):
        r = (127 + 120 * np.sin(2 * np.pi * xs / f)).astype(np.uint8)
        out.append(("periódica", np.dstack([np.tile(r, (112, 1))] * 3)))
    out += [("cara invertida", np.ascontiguousarray(c[::-1])) for c in caras_fam[:10]]
    out += [("cara girada 90°", np.ascontiguousarray(np.rot90(c))) for c in caras_fam[10:20]]
    for img in fotos:   # recortes de fondo (esquinas de las fotos)
        h, w = img.shape[:2]
        if min(h, w) >= 120:
            for y, x in ((0, 0), (0, w - 112), (h - 112, 0), (h - 112, w - 112)):
                out.append(("fondo de foto", img[y:y + 112, x:x + 112].copy()))
    return out


def main():
    p = argparse.ArgumentParser(description="Entrena el modelo EBGM e inscribe famosos + cigala")
    p.add_argument("--dataset", default=DIR_DATASET)
    p.add_argument("--cigala-dir", default=DIR_CIGALA)
    p.add_argument("--foto-cigala", default=None,
                   help="nombre del fichero de fotos_cigala/ con el que inscribir al cigala "
                        "(por defecto, la foto más representativa)")
    p.add_argument("--bd", default=RUTA_BD_POR_DEFECTO)
    p.add_argument("--far-objetivo", type=float, default=0.01,
                   help="FAR objetivo para fijar el umbral de decisión")
    a = p.parse_args()

    det = DetectorCaras(politica_multiples="mayor")
    ext = ExtractorEBGM(modelo=None)

    # ------------------------------------------------ 1) modelo EBGM (cohorte)
    rutas_fam = sorted(glob.glob(os.path.join(a.dataset, "*")))
    caras_fam, rutas_ok, fallos = caras_de(det, rutas_fam)
    if len(caras_fam) < 5:
        raise SystemExit(f"Solo {len(caras_fam)} caras en {a.dataset}; ¿falta el Dataset?")
    modelo = ext.entrenar_modelo(caras_fam, info={"cohorte": "Dataset docente"})
    modelo.guardar(RUTA_MODELO)
    ext.fijar_modelo(modelo)
    print(f"[modelo] Face Bunch Graph de {len(caras_fam)} caras (+espejos), "
          f"{ext.n_nodos} nodos x {ext.cfg.n_filtros} filtros -> "
          f"{os.path.relpath(RUTA_MODELO, RAIZ)}")

    # mismo camino que inscribir(): template de inscripción (con micro-variaciones)
    tpl_fam = {id_desde_fichero(r): ext.extraer_inscripcion(c) for r, c in zip(rutas_ok, caras_fam)}
    # para calibrar se usan templates de SONDA (como llegan en verificar/identificar)
    sonda_fam = [ext.extraer(c) for c in caras_fam]

    # ------------------------------------------------ 2) cigala (UNA foto)
    caras_cig, rutas_cig, _ = caras_de(det, sorted(glob.glob(os.path.join(a.cigala_dir, "*"))))
    tpl_cig_fotos = [ext.extraer(c) for c in caras_cig]
    if not tpl_cig_fotos:
        raise SystemExit("No se pudo extraer ninguna foto del cigala")
    nombres_cig = [os.path.basename(r) for r in rutas_cig]
    if a.foto_cigala:
        if a.foto_cigala not in nombres_cig:
            raise SystemExit(f"{a.foto_cigala} no está en {a.cigala_dir} (o no tiene cara)")
        k_cig = nombres_cig.index(a.foto_cigala)
    elif len(tpl_cig_fotos) == 1:
        k_cig = 0
    else:
        # la foto más representativa: mayor similitud media con las demás
        reps = [ext.representar(t) for t in tpl_cig_fotos]
        medias = [np.mean([ext.similitud_rep(reps[j], reps[i])
                           for j in range(len(reps)) if j != i]) for i in range(len(reps))]
        k_cig = int(np.argmax(medias))
    foto_cigala = nombres_cig[k_cig]
    tpl_cigala = ext.extraer_inscripcion(caras_cig[k_cig])   # igual que los famosos

    # ------------------------------------------------ 3) calibración
    # galería = templates de inscripción; sonda = templates normales
    gal_f = [ext.representar(t) for t in tpl_fam.values()]
    son_f = [ext.representar(t) for t in sonda_fam]
    son_c = [ext.representar(t) for t in tpl_cig_fotos]
    gal_c = [ext.representar(ext.extraer_inscripcion(c)) for c in caras_cig]
    S = ext.similitud_rep
    nf, nc = len(gal_f), len(son_c)
    imp = [S(son_f[i], gal_f[j]) for i in range(nf) for j in range(nf) if i != j]
    imp += [S(sc, gf) for sc in son_c for gf in gal_f]
    imp = np.sort(np.array(imp))
    umbral = float(np.quantile(imp, 1 - a.far_objetivo))
    gen = [S(son_c[i], gal_c[j]) for i in range(nc) for j in range(nc) if i != j]
    frr = float(np.mean(np.array(gen) < umbral)) if gen else None   # con 1 foto no hay pares

    # Caricidad: SOLO caras no vistas por el modelo. Los famosos están en el
    # bunch y puntuarían alto de forma optimista, así que cada uno se mide
    # SIN su propia foto (ni su espejo) en el bunch (leave-one-out); y las
    # fotos del cigala, que el modelo no ha visto.
    F = ext.cfg.n_filtros
    n_fam = len(caras_fam)

    def caricidad_sin(tpl, excluir):
        j = ext._l2(ext._comprimir(tpl[:, :F]))
        bunch = np.delete(ext.modelo.bunch, excluir, axis=0)
        return float(np.einsum("nf,bnf->bn", j, bunch).max(axis=0).mean())

    car_caras = np.array([caricidad_sin(t, [i, i + n_fam]) for i, t in enumerate(sonda_fam)]
                         + [ext.puntuacion_cara(t) for t in tpl_cig_fotos])
    imgs = [det.cargar_imagen(r) for r in rutas_ok]
    nc = no_caras(caras_fam, imgs)
    car_no = np.array([ext.puntuacion_cara(ext.extraer(x)) for _, x in nc])
    # Prioridad: NO rechazar caras reales -> margen de 0.02 bajo la peor cara
    # real no vista. Lo que quede por encima se documenta como límite conocido.
    umbral_cara = float(car_caras.min() - 0.02)
    por_cat: dict[str, list] = {}
    for (cat, _), v in zip(nc, car_no):
        por_cat.setdefault(cat, []).append(v >= umbral_cara)
    cal = {"umbral": round(umbral, 5), "umbral_cara": round(umbral_cara, 5),
           "far_objetivo": a.far_objetivo,
           "frr_estimado_cigala_1foto": None if frr is None else round(frr, 4),
           "caricidad_caras_min": round(float(car_caras.min()), 4),
           "caricidad_no_caras_max": round(float(car_no.max()), 4),
           "no_caras_que_pasan_por_categoria": {k: f"{sum(v)}/{len(v)}" for k, v in por_cat.items()},
           "n_impostores": int(len(imp)), "n_genuinos": int(len(gen)),
           "foto_inscripcion_cigala": foto_cigala}
    # los umbrales viajan DENTRO del modelo (protegido por el HMAC de la BD)
    modelo.info.update(umbral=cal["umbral"], umbral_cara=cal["umbral_cara"],
                       far_objetivo=a.far_objetivo)
    modelo.guardar(RUTA_MODELO)
    os.makedirs(os.path.dirname(RUTA_CALIBRACION), exist_ok=True)
    with open(RUTA_CALIBRACION, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    print(f"[calibración] umbral={umbral:.4f} (FAR objetivo {a.far_objetivo:.1%}"
          + (f"; FRR 1 foto cigala ≈ {frr:.0%})" if frr is not None else ")"))
    print(f"[calibración] caricidad: caras reales no vistas min={car_caras.min():.3f} "
          f"-> umbral_cara={umbral_cara:.3f}")
    print("[calibración] entradas NO-cara que el control deja pasar (límite conocido de _sd/_t):")
    for k, v in por_cat.items():
        print(f"     {k:18s} {sum(v)}/{len(v)}")

    # ------------------------------------------------ 4) inscripción en la BD
    for ruta in (a.bd, a.bd + ".mac"):
        if os.path.isfile(ruta):
            os.remove(ruta)
    sis = SistemaReconocimiento(ruta_bd=a.bd, auditoria=False)
    for id_, tpl in tpl_fam.items():
        sis.inscribir_t(id_, tpl)
    sis.inscribir_t("cigala", tpl_cigala)
    sis.auditoria = True
    sis._auditar("bd_creada", n=len(sis.personas()))
    print(f"[cigala] inscrito con UNA foto: {os.path.relpath(a.cigala_dir, RAIZ)}/{foto_cigala}"
          + (f" (la más representativa de {len(tpl_cig_fotos)})" if len(tpl_cig_fotos) > 1 else ""))
    print(f"\nInscritas {len(sis.personas())} personas en {os.path.relpath(a.bd, RAIZ)}")
    if fallos:
        print(f"No se pudieron procesar {len(fallos)}:")
        for n, e in fallos:
            print(f"   - {n}: {e}")


if __name__ == "__main__":
    main()
