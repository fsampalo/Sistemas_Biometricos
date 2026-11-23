"""Rutas del proyecto, en un único sitio.

    reconocimiento/   código del sistema (este paquete)
    scripts/          programas que se ejecutan (inscribir, demo, evaluar...)
    datos/            Dataset del docente, fotos del Cigala y la BD .csv
    models/           YuNet, modelo EBGM, calibración y clave de la BD
    salida/           figuras, caché del ajuste y auditoría
"""

import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIR_DATOS = os.path.join(RAIZ, "datos")
DIR_MODELOS = os.path.join(RAIZ, "models")
DIR_SALIDA = os.path.join(RAIZ, "salida")

DIR_DATASET = os.path.join(DIR_DATOS, "Dataset")
DIR_CIGALA = os.path.join(DIR_DATOS, "fotos_cigala")
DIR_CIGALA_TEST = os.path.join(DIR_DATOS, "fotos_cigala_test")
RUTA_BD = os.path.join(DIR_DATOS, "bd_sistema.csv")
