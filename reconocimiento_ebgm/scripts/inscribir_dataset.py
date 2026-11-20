"""Inscribe las 50 fotos del dataset (49 del docente + El Cigala) en la BD CSV.
El id es el nombre del fichero sin el ano: 'keanu_reeves_2021.png' -> 'keanu_reeves'.

Uso: python scripts/inscribir_dataset.py [--bd datos/bd.csv] [--dataset datos/dataset]
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ebgm.sistema import SistemaEBGM  # noqa: E402


def id_desde_fichero(ruta):
    base = os.path.splitext(os.path.basename(ruta))[0]
    return re.sub(r"_\d{4}$", "", base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bd", default="datos/bd.csv")
    ap.add_argument("--dataset", default="datos/dataset")
    a = ap.parse_args()
    if os.path.exists(a.bd):
        os.remove(a.bd)
    s = SistemaEBGM(a.bd)
    for f in sorted(glob.glob(os.path.join(a.dataset, "*"))):
        s.inscribir(id_desde_fichero(f), f)
        print(f"inscrito {id_desde_fichero(f):45s} (S_FBG={s.ultima_similitud_fbg:.3f})")
    print(f"Total inscritos: {len(s.bd)}")


if __name__ == "__main__":
    main()
