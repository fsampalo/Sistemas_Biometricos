"""Evaluacion del sistema propio: verificacion (FAR/FRR/EER) e identificacion (rank-1).

Protocolo: galeria = foto original inscrita con inscribir(). Sondas = la misma
foto sometida a cada transformacion de scripts/aumentos.py. Se compara cada
sonda con su template (genuino) y con los otros 49 (impostores).
Limitacion: al haber una unica foto por persona, los genuinos proceden de la
misma captura y los resultados son optimistas (ver memoria).

Uso: python scripts/evaluar.py [--bd datos/bd.csv] [--salida resultados]
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
from aumentos import AUMENTOS  # noqa: E402
from inscribir_dataset import id_desde_fichero  # noqa: E402

_S = None


def _init(bd):
    global _S
    import warnings
    warnings.filterwarnings("ignore")
    from ebgm.sistema import SistemaEBGM
    _S = SistemaEBGM(bd)


def _tarea(args):
    ruta, nombre_aum, modo = args
    from ebgm.normalizacion import cargar_imagen
    from ebgm.detector import elegir_cara
    img = cargar_imagen(ruta)
    pid = id_desde_fichero(ruta)
    try:
        sonda = AUMENTOS[nombre_aum](img, _S.detector)
        if modo == "detector":
            tpl = _S.extraer_template(sonda)
        else:  # sin detector: el "atacante/usuario" entrega un recorte de la cara
            d = elegir_cara(_S.detector.detectar(img))
            x, y, w, h = map(int, d["caja"])
            mx, my = int(0.12 * w), int(0.12 * h)
            rec = sonda[max(0, y - my):y + h + my, max(0, x - mx):x + w + mx]
            tpl = _S.extraer_template_sd(rec)
    except ValueError as e:
        return dict(id=pid, aum=nombre_aum, modo=modo, fallo=str(e))
    p = _S.puntuaciones(tpl)
    return dict(id=pid, aum=nombre_aum, modo=modo, genuino=p[pid],
                impostores={k: v for k, v in p.items() if k != pid},
                rank1=max(p, key=p.get))


def metricas(gen, imp, umbrales):
    gen, imp = np.asarray(gen), np.asarray(imp)
    far = np.array([(imp >= u).mean() for u in umbrales])
    frr = np.array([(gen < u).mean() for u in umbrales])
    i = int(np.argmin(np.abs(far - frr)))
    return far, frr, float((far[i] + frr[i]) / 2), float(umbrales[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bd", default="datos/bd.csv")
    ap.add_argument("--dataset", default="datos/dataset")
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--umbral", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    fotos = sorted(glob.glob(os.path.join(a.dataset, "*")))
    tareas = [(f, n, "detector") for f in fotos for n in AUMENTOS]
    tareas += [(f, n, "sin_detector") for f in fotos
               for n in ["original", "brillo", "desenfoque", "ruido", "rotacion_12", "gafas_sol"]]
    with Pool(os.cpu_count(), initializer=_init, initargs=(a.bd,)) as p:
        res = p.map(_tarea, tareas, chunksize=4)
    with open(os.path.join(a.salida, "evaluacion_bruta.json"), "w") as f:
        json.dump(res, f)
    print(f"{len(res)} sondas procesadas; fallos de deteccion: {sum('fallo' in r for r in res)}")


if __name__ == "__main__":
    main()
