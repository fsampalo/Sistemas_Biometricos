"""Interfaz de linea de comandos del sistema de reconocimiento facial EBGM.

Ejemplos:
    python cli.py inscribir   --id "ada_lovelace" --foto ada.jpg
    python cli.py verificar   --id "ada_lovelace" --foto prueba.jpg
    python cli.py identificar --foto prueba.jpg
    python cli.py listar
Anade --sd para omitir el detector (la foto debe ser un recorte de la cara).
"""
import argparse
import sys

from ebgm.sistema import SistemaEBGM


def main():
    ap = argparse.ArgumentParser(description="Reconocimiento facial EBGM")
    ap.add_argument("accion", choices=["inscribir", "verificar", "identificar", "listar"])
    ap.add_argument("--id")
    ap.add_argument("--foto")
    ap.add_argument("--bd", default="datos/bd.csv")
    ap.add_argument("--sd", action="store_true", help="omitir el detector de caras")
    a = ap.parse_args()
    S = SistemaEBGM(a.bd)
    suf = "_sd" if a.sd else ""
    if a.accion == "listar":
        for i in sorted(S.bd):
            print(i)
        print(f"\n{len(S.bd)} personas inscritas", file=sys.stderr)
        return
    if a.accion == "inscribir":
        getattr(S, "inscribir" + suf)(a.id, a.foto)
        print(f"Inscrito: {a.id}")
    elif a.accion == "verificar":
        ok, sc = getattr(S, "verificar" + suf)(a.id, a.foto)
        print(f"{'ACEPTADO' if ok else 'RECHAZADO'}  (similitud {sc:.4f}, umbral {S.umbral})")
    elif a.accion == "identificar":
        idn, sc = getattr(S, "identificar" + suf)(a.foto)
        print(f"{idn or 'DESCONOCIDO'}  (similitud {sc:.4f}, umbral {S.umbral})")


if __name__ == "__main__":
    main()
