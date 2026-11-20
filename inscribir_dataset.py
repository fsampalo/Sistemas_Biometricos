"""
Inscribe en el sistema a las personas conocidas:
  - los 50 famosos comunes del docente (carpeta Dataset/)
  - nuestro famoso elegido: EL CIGALA (una foto frontal de fotos_cigala/)

Además calcula la "cara media" de fondo (models/ebgm_fondo.npy) a partir de los
famosos comunes, que el matcher usa para normalizar (ver ebgm.py).

Uso:
    python inscribir_dataset.py
    python inscribir_dataset.py --dataset Dataset --cigala fotos_cigala/1.jpg
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

from detector import DetectorCaras, ErrorDeteccion
from ebgm import ExtractorEBGM
from sistema import SistemaReconocimiento, RUTA_BD_POR_DEFECTO

AQUI = os.path.dirname(os.path.abspath(__file__))


def id_desde_fichero(ruta: str) -> str:
    """adele_2016.jpg -> 'adele'. Quita la extensión y el año final."""
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    return re.sub(r"_\d{4}$", "", nombre)


def main():
    p = argparse.ArgumentParser(description="Inscribe famosos + cigala")
    p.add_argument("--dataset", default=os.path.join(AQUI, "Dataset"))
    p.add_argument("--cigala-dir", default=os.path.join(AQUI, "fotos_cigala"),
                   help="carpeta con las fotos de referencia del cigala (se promedian)")
    p.add_argument("--bd", default=RUTA_BD_POR_DEFECTO)
    a = p.parse_args()

    # Empezar de cero
    if os.path.isfile(a.bd):
        os.remove(a.bd)

    sis = SistemaReconocimiento(ruta_bd=a.bd)
    det = DetectorCaras(politica_multiples="mayor")
    ext = sis.extractor

    # 1) Extraer templates de los famosos comunes
    rutas = sorted(glob.glob(os.path.join(a.dataset, "*")))
    templates: dict[str, np.ndarray] = {}
    fallos = []
    for ruta in rutas:
        try:
            cara = det.extraer_cara(ruta)
            templates[id_desde_fichero(ruta)] = ext.extraer(cara)
        except ErrorDeteccion as e:
            fallos.append((os.path.basename(ruta), str(e)))

    # 2) Calcular y guardar la cara media de fondo (solo con los famosos comunes)
    ext.ajustar_fondo(list(templates.values()))
    ext.guardar_fondo()
    print(f"[fondo] cara media guardada ({len(templates)} famosos) -> models/ebgm_fondo.npy")

    # 3) Añadir nuestro famoso: el cigala, como CENTROIDE de varias fotos.
    #    Promediar varios ejemplos da una plantilla más robusta a pose/edad que
    #    una sola foto, SIN dejar de ser un único template por persona (igual de
    #    "justo" que los famosos: no infla las puntuaciones como haría un bunch).
    cig_tpls = []
    for ruta in sorted(glob.glob(os.path.join(a.cigala_dir, "*"))):
        try:
            cig_tpls.append(ext.extraer(det.extraer_cara(ruta)))
        except ErrorDeteccion:
            pass
    if cig_tpls:
        # Centroide en el espacio NORMALIZADO (resta de fondo + L2 por nodo): así
        # cada foto pesa igual y no dominan las de más energía de Gabor. Se vuelve
        # a sumar la media de fondo para que, al normalizar el matcher de nuevo,
        # recupere exactamente este centroide. Sigue siendo un único template.
        centroide = np.mean([ext._normalizar(t) for t in cig_tpls], axis=0)
        templates["cigala"] = (ext.media + centroide).astype(np.float32)
        print(f"[cigala] inscrito como centroide de {len(cig_tpls)} fotos de "
              f"{os.path.relpath(a.cigala_dir, AQUI)}")
    else:
        print("[cigala] ERROR: no se pudo extraer ninguna foto del cigala")

    # 4) Guardar todos en la BD (vía inscribir_t: ya tenemos los templates)
    for id_, tpl in templates.items():
        sis.inscribir_t(id_, tpl)

    print(f"\nInscritas {len(sis.personas())} personas en {os.path.relpath(a.bd, AQUI)}")
    if fallos:
        print(f"No se pudieron procesar {len(fallos)}:")
        for n, e in fallos:
            print(f"   - {n}: {e}")


if __name__ == "__main__":
    main()
