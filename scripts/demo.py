"""
Demo de arranque: los 9 métodos del sistema EBGM funcionando.

Si no existe la BD/modelo, los crea (inscribir_dataset.py; necesita Dataset/).
Las inscripciones de prueba se hacen sobre una COPIA temporal de la BD para no
ensuciar la real.

Uso:
    python scripts/demo.py
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raíz del repo

from reconocimiento.ebgm import RUTA_MODELO
from reconocimiento.rutas import DIR_CIGALA_TEST, DIR_DATASET, RAIZ
from reconocimiento.sistema import RUTA_BD_POR_DEFECTO, ErrorSistema, SistemaReconocimiento


def asegurar_bd():
    if not (os.path.isfile(RUTA_BD_POR_DEFECTO) and os.path.isfile(RUTA_MODELO)):
        print("No hay BD/modelo; entrenando EBGM e inscribiendo famosos + cigala...\n")
        import inscribir_dataset
        inscribir_dataset.main()
        print()


def corto(r: dict) -> str:
    r = dict(r)
    if "ranking" in r:
        r["ranking"] = [(k, round(s, 3)) for k, s in r["ranking"][:3]]
    for k in ("similitud", "umbral"):
        if isinstance(r.get(k), float):
            r[k] = round(r[k], 4)
    return str(r)


def main():
    asegurar_bd()
    tmp = tempfile.mkdtemp(prefix="ebgm_demo_")
    bd = os.path.join(tmp, "bd.csv")
    shutil.copy(RUTA_BD_POR_DEFECTO, bd)
    shutil.copy(RUTA_BD_POR_DEFECTO + ".mac", bd + ".mac")
    # la demo muestra similitudes (modo evaluación); al final, el modo por defecto
    sis = SistemaReconocimiento(ruta_bd=bd, auditoria=False, exponer_puntuaciones=True)
    print(f"BD: {len(sis.personas())} personas | umbral={sis.umbral:+.4f}\n")

    # foto del cigala que el sistema NUNCA ha visto (no usada al inscribir)
    foto = os.path.join(DIR_CIGALA_TEST, "baltycka2024_01.jpg")
    otra = os.path.join(DIR_DATASET, "keanu_reeves_2021.png")
    print(f"Foto de prueba: {os.path.relpath(foto, RAIZ)} (cigala, no inscrita)\n")

    print("── pipeline completo (detector + extractor + matcher) ──")
    print("  identificar(foto)            ->", corto(sis.identificar(foto)))
    print("  verificar('cigala', foto)    ->", corto(sis.verificar("cigala", foto)))
    print("  verificar('adele', foto)     ->", corto(sis.verificar("adele", foto)))

    print("\n── sin detector (_sd): cara ya recortada y alineada ──")
    cara = sis.extraer_cara(foto)
    print("  identificar_sd(cara)         ->", corto(sis.identificar_sd(cara)))
    print("  verificar_sd('cigala', cara) ->", corto(sis.verificar_sd("cigala", cara)))

    print("\n── solo template (_t) ──")
    tpl = sis.extraer_template(foto)
    print("  identificar_t(template)         ->", corto(sis.identificar_t(tpl)))
    print("  verificar_t('cigala', template) ->", corto(sis.verificar_t("cigala", tpl)))

    print("\n── modo por defecto: NO se exponen similitudes (anti hill-climbing) ──")
    sis_def = SistemaReconocimiento(ruta_bd=bd, auditoria=False)
    print("  identificar(foto)         ->", sis_def.identificar(foto))
    print("  verificar('cigala', foto) ->", sis_def.verificar("cigala", foto))

    print("\n── inscribir (las 3 variantes) sobre la copia temporal ──")
    print("  inscribir('demo_foto', foto_keanu)  ->", sis.inscribir("demo_foto", otra))
    print("  inscribir_sd('demo_cara', cara)     ->", sis.inscribir_sd("demo_cara", cara))
    print("  inscribir_t('demo_tpl', template)   ->", sis.inscribir_t("demo_tpl", tpl))
    try:
        sis.inscribir("cigala", foto)
    except ErrorSistema as e:
        print("  inscribir('cigala', ...) otra vez  -> rechazado:", e)

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nListo. Evaluación: python scripts/evaluar.py | Seguridad: python scripts/evaluar_ataques.py")


if __name__ == "__main__":
    main()
