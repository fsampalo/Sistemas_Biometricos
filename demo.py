"""
Demo de arranque: muestra los 9 métodos del sistema EBGM funcionando.

Si no existe la BD, la crea inscribiendo los famosos + cigala (necesita Dataset/).
Luego ejecuta un ejemplo de cada método.

Uso:
    python demo.py
"""

from __future__ import annotations

import os

import cv2

from detector import DetectorCaras
from ebgm import ExtractorEBGM
from sistema import SistemaReconocimiento, RUTA_BD_POR_DEFECTO

AQUI = os.path.dirname(os.path.abspath(__file__))


def asegurar_bd():
    if not os.path.isfile(RUTA_BD_POR_DEFECTO):
        print("No hay BD; inscribiendo famosos + cigala...\n")
        import inscribir_dataset
        inscribir_dataset.main()
        print()


def main():
    asegurar_bd()
    sis = SistemaReconocimiento()
    print(f"\nBD: {len(sis.personas())} personas inscritas | umbral={sis.umbral:+.4f}\n")

    foto_prueba = os.path.join(AQUI, "fotos_cigala", "4.jpg")   # cigala NO inscrita
    print(f"Foto de prueba: {os.path.relpath(foto_prueba, AQUI)} (el cigala)\n")

    print("── pipeline completo ──")
    print("  identificar(foto) ->", sis.identificar(foto_prueba)["id"])
    print("  verificar('cigala', foto) ->", sis.verificar("cigala", foto_prueba)["verificado"])

    print("\n── sin detector (_sd): sobre la cara ya recortada ──")
    cara = DetectorCaras().extraer_cara(foto_prueba)
    ruta_cara = os.path.join(AQUI, "salida", "cara_demo.png")
    os.makedirs(os.path.dirname(ruta_cara), exist_ok=True)
    cv2.imwrite(ruta_cara, cara)
    print("  identificar_sd(cara) ->", sis.identificar_sd(ruta_cara)["id"])
    print("  verificar_sd('cigala', cara) ->", sis.verificar_sd("cigala", ruta_cara)["verificado"])

    print("\n── sólo template (_t) ──")
    tpl = ExtractorEBGM().extraer(cara)
    print("  identificar_t(template) ->", sis.identificar_t(tpl)["id"])
    print("  verificar_t('cigala', template) ->", sis.verificar_t("cigala", tpl)["verificado"])

    print("\n── inscribir una identidad nueva y recuperarla ──")
    sis.inscribir_t("persona_demo", tpl)
    print("  identificar_t(mismo template) ->", sis.identificar_t(tpl)["id"])
    # limpiar la identidad de demo
    sis._bd.pop("persona_demo", None); sis._guardar_bd()

    print("\nListo. Para métricas: python evaluar.py | Para seguridad: python evaluar_ataques.py")


if __name__ == "__main__":
    main()
