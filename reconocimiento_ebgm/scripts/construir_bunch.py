"""Construye el Face Bunch Graph (FBG) a partir de las 49 fotos del docente
(y sus espejos). Los nodos de los grafos modelo se etiquetan automaticamente
con el modelo LBF de 68 puntos de OpenCV (en el articulo original se etiquetaban
a mano). Solo hace falta ejecutarlo una vez; el resultado (modelos/fbg.npz) se
distribuye con el repositorio y el LBF no se usa en ejecucion.

Uso: python scripts/construir_bunch.py --lbf ruta/lbfmodel.yaml
(descarga: https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml)
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ebgm.detector import detector_por_defecto, elegir_cara  # noqa: E402
from ebgm.normalizacion import cargar_imagen, alinear, fotometria, TAM  # noqa: E402
from ebgm.gabor import BancoGabor, muestrear_jets  # noqa: E402
from ebgm.grafo import nodos_desde_68, aristas_delaunay, BunchGraph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lbf", required=True)
    ap.add_argument("--dataset", default="datos/dataset")
    ap.add_argument("--excluir", default="diego_ramon", help="prefijo de fotos a excluir del FBG")
    ap.add_argument("--salida", default="modelos/fbg.npz")
    a = ap.parse_args()

    fm = cv2.face.createFacemarkLBF()
    fm.loadModel(a.lbf)
    det = detector_por_defecto()
    banco = BancoGabor(TAM)
    escala = 2  # LBF funciona mejor sobre la cara a 288 px
    rect = np.array([[20 * escala, 15 * escala, 104 * escala, 120 * escala]], dtype=np.int32)

    grafos, jets = [], []
    fotos = sorted(f for f in glob.glob(os.path.join(a.dataset, "*"))
                   if not os.path.basename(f).startswith(a.excluir))
    for f in fotos:
        img = cargar_imagen(f)
        dets = det.detectar(img)
        if not dets:
            print("sin cara:", f)
            continue
        can = alinear(img, elegir_cara(dets))
        for espejo in (False, True):
            c = cv2.flip(can, 1) if espejo else can
            big = cv2.resize(c, (TAM * escala, TAM * escala))
            ok, lm = fm.fit(big, rect)
            if not ok:
                continue
            pos = nodos_desde_68(np.array(lm).reshape(-1, 2) / escala)
            pos = np.clip(pos, 1, TAM - 2)
            R = banco.transformar(fotometria(c))
            grafos.append(pos)
            jets.append(muestrear_jets(R, pos))
    grafos = np.array(grafos)
    pos_media = grafos.mean(0)
    fbg = BunchGraph(pos_media, aristas_delaunay(pos_media), np.array(jets))
    fbg.guardar(a.salida)
    print(f"FBG: {len(jets)} grafos modelo, {pos_media.shape[0]} nodos, "
          f"{len(fbg.aristas)} aristas -> {a.salida}")


if __name__ == "__main__":
    main()
