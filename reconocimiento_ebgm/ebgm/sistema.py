"""Sistema de reconocimiento facial basado en Elastic Bunch Graph Matching.

API exigida por el guion:
    inscribir(id, foto)      verificar(id, foto)      identificar(foto)
    inscribir_sd(id, foto)   verificar_sd(id, foto)   identificar_sd(foto)
    inscribir_t(id, tpl)     verificar_t(id, tpl)     identificar_t(tpl)

`foto` puede ser una ruta o un array de imagen (BGR o gris).
`tpl` es un template: np.ndarray float32 de longitud LONG_TEMPLATE o su
codificacion base64 tal y como aparece en el CSV.

Valores devueltos:
    inscribir*   -> True (lanza ValueError si el id ya existe o la entrada no es valida)
    verificar*   -> (aceptado: bool, puntuacion: float)
    identificar* -> (id | None, puntuacion: float)   None si ningun candidato supera el umbral
"""
import base64
import csv
import hashlib
import hmac
import os
import re
import warnings

import numpy as np

from .detector import detector_por_defecto, elegir_cara
from .gabor import BancoGabor, N_KERNELS
from .grafo import BunchGraph, N_NODOS, NOMBRES_NODOS
from .matching import ajustar_grafo
from .normalizacion import (TAM, _ojos_desde_caja, a_original, alinear, cargar_imagen,
                            fotometria, sin_detector)

DIR_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Template = magnitudes de los jets de los 33 nodos (33 x 40 = 1320 floats).
# La version 1 guardaba tambien posiciones y fases; se eliminaron porque no
# intervienen en la comparacion y facilitaban invertir el template (ver memoria).
LONG_TEMPLATE = N_NODOS * N_KERNELS
UMBRAL_POR_DEFECTO = 0.62           # coseno con resta de media; FAR=0 en evaluacion (ver memoria)
_IDX_PA, _IDX_PB = NOMBRES_NODOS.index("pupila_a"), NOMBRES_NODOS.index("pupila_b")
_RE_ID = re.compile(r"^[A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ_ .()'-]{1,80}$")
CAMPOS = ["id", "template", "hmac"]


class SistemaEBGM:
    def __init__(self, ruta_bd="datos/bd.csv", umbral=UMBRAL_POR_DEFECTO, detector=None,
                 ruta_fbg=None, clave_hmac=None):
        self.ruta_bd = ruta_bd
        self.umbral = umbral
        self.detector = detector or detector_por_defecto()
        self.fbg = BunchGraph.cargar(ruta_fbg or os.path.join(DIR_RAIZ, "modelos", "fbg.npz"))
        self.banco = BancoGabor(TAM)
        # jet medio por nodo (publico, viene del FBG): se resta antes del coseno
        # para eliminar la componente comun a todas las caras, que permitia un
        # 'template maestro' (ver memoria, P3). mag media -> (N_NODOS, 40).
        self.mu_mag = np.abs(self.fbg.jets).mean(0).astype(np.float64)
        clave = clave_hmac if clave_hmac is not None else os.environ.get("SB_CLAVE_HMAC")
        self.clave = clave.encode() if clave else None
        self.bd = {}                 # id -> template (np.float32)
        self.filas_rechazadas = []
        self._cargar_bd()

    # ------------------------------------------------------------------ BD
    def _firma(self, id_, tpl_b64):
        return hmac.new(self.clave, f"{id_}\x1f{tpl_b64}".encode(), hashlib.sha256).hexdigest()

    def _cargar_bd(self):
        self.bd, self.filas_rechazadas = {}, []
        if not os.path.exists(self.ruta_bd):
            return
        with open(self.ruta_bd, newline="", encoding="utf-8") as f:
            for i, fila in enumerate(csv.DictReader(f)):
                try:
                    id_, b64 = fila["id"], fila["template"]
                    if self.clave is not None and not hmac.compare_digest(
                            self._firma(id_, b64), fila.get("hmac") or ""):
                        raise ValueError("firma HMAC incorrecta")
                    tpl = self._validar_template(b64)
                    if id_ in self.bd:
                        raise ValueError("id duplicado")
                    self.bd[id_] = tpl
                except Exception as e:  # una fila corrupta no tumba el sistema
                    self.filas_rechazadas.append((i + 2, str(e)))
                    warnings.warn(f"BD: fila {i + 2} rechazada ({e})")

    def _guardar_fila(self, id_, tpl):
        nuevo = not os.path.exists(self.ruta_bd)
        os.makedirs(os.path.dirname(os.path.abspath(self.ruta_bd)), exist_ok=True)
        b64 = codificar_template(tpl)
        firma = self._firma(id_, b64) if self.clave is not None else ""
        with open(self.ruta_bd, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS, quoting=csv.QUOTE_ALL)
            if nuevo:
                w.writeheader()
            w.writerow({"id": id_, "template": b64, "hmac": firma})

    # ------------------------------------------------------- validaciones
    @staticmethod
    def _validar_id(id_):
        if not isinstance(id_, str) or not _RE_ID.match(id_) or id_.strip() != id_:
            raise ValueError(f"Identificador no valido: {id_!r}")
        return id_

    @staticmethod
    def _validar_template(tpl):
        if isinstance(tpl, str):
            tpl = decodificar_template(tpl)
        tpl = np.asarray(tpl, dtype=np.float64).ravel()
        if tpl.size != LONG_TEMPLATE:
            raise ValueError(f"Template de longitud {tpl.size}, se esperaba {LONG_TEMPLATE}")
        if not np.all(np.isfinite(tpl)):
            raise ValueError("Template con valores no finitos")
        mag = _partes(tpl)
        if mag.min() < 0 or np.any(np.linalg.norm(mag, axis=1) < 1e-9):
            raise ValueError("Magnitudes de jets no validas")
        if mag.max() > 1e6:
            raise ValueError("Magnitudes fuera de rango")
        return tpl.astype(np.float32)

    # ------------------------------------------------------ extraccion
    def _ajustar(self, can, modo):
        R = self.banco.transformar(fotometria(can))
        return R, ajustar_grafo(R, self.fbg, modo)

    def _template_realineado(self, img, ojos, modo_inicial, iteraciones):
        """Ajusta el grafo y realinea la imagen con las pupilas que localiza el
        propio EBGM, de forma que todas las rutas (con y sin detector) acaban en
        el mismo marco canonico definido por el grafo."""
        h, w = img.shape[:2]
        if modo_inicial == "sin_detector":
            can, M = sin_detector(img), np.array([[TAM / w, 0, 0], [0, TAM / h, 0]])
        else:
            can, M = alinear(img, {"ojos": ojos, "caja": (0, 0, w, h)}, devolver_matriz=True)
        modo = modo_inicial
        for _ in range(iteraciones):
            _, (pos, _, _) = self._ajustar(can, modo)
            pupilas = a_original(M, pos[[_IDX_PA, _IDX_PB]])
            can, M = alinear(img, {"ojos": (tuple(pupilas[0]), tuple(pupilas[1])),
                                   "caja": (0, 0, w, h)}, devolver_matriz=True)
            modo = "detector"
        _, (pos, s_fbg, J) = self._ajustar(can, "detector")
        self.ultima_similitud_fbg = s_fbg
        self.ultima_canonica = can
        self.ultimo_grafo = pos
        return construir_template(pos, J)

    def extraer_template(self, foto):
        img = cargar_imagen(foto)
        cara = elegir_cara(self.detector.detectar(img))
        if cara is None:
            raise ValueError("No se ha detectado ninguna cara")
        self.ultima_deteccion = cara
        ojos = cara.get("ojos") or _ojos_desde_caja(cara["caja"])
        return self._template_realineado(img, ojos, "detector", 1)

    def extraer_template_sd(self, foto):
        """Sin detector: la imagen se trata como recorte de la cara; el EBGM busca
        el grafo con un rango amplio de posicion/escala y despues se realinea con
        las pupilas localizadas (dos iteraciones)."""
        return self._template_realineado(cargar_imagen(foto), None, "sin_detector", 2)

    # ------------------------------------------------------ comparacion
    def comparar(self, t1, t2):
        """Similitud media (coseno) de magnitudes nodo a nodo tras restar el jet
        medio. El resultado esta en [-1, 1]; para caras genuinas queda proximo a 1."""
        m1, m2 = _partes(t1) - self.mu_mag, _partes(t2) - self.mu_mag
        num = (m1 * m2).sum(1)
        den = np.linalg.norm(m1, axis=1) * np.linalg.norm(m2, axis=1) + 1e-12
        return float((num / den).mean())

    def puntuaciones(self, tpl):
        return {i: self.comparar(tpl, t) for i, t in self.bd.items()}

    # ------------------------------------------------------ nucleo comun
    def _inscribir(self, id_, tpl):
        self._validar_id(id_)
        tpl = self._validar_template(tpl)
        if id_ in self.bd:
            raise ValueError(f"El id {id_!r} ya esta inscrito")
        self.bd[id_] = tpl
        self._guardar_fila(id_, tpl)
        return True

    def _verificar(self, id_, tpl):
        tpl = self._validar_template(tpl)
        if id_ not in self.bd:
            return False, 0.0
        s = self.comparar(tpl, self.bd[id_])
        return s >= self.umbral, s

    def _identificar(self, tpl):
        tpl = self._validar_template(tpl)
        if not self.bd:
            return None, 0.0
        p = self.puntuaciones(tpl)
        mejor = max(p, key=p.get)
        return (mejor if p[mejor] >= self.umbral else None), p[mejor]

    # ------------------------------------------------------ API del guion
    def inscribir(self, id, foto):
        return self._inscribir(id, self.extraer_template(foto))

    def verificar(self, id, foto):
        return self._verificar(id, self.extraer_template(foto))

    def identificar(self, foto):
        return self._identificar(self.extraer_template(foto))

    def inscribir_sd(self, id, foto):
        return self._inscribir(id, self.extraer_template_sd(foto))

    def verificar_sd(self, id, foto):
        return self._verificar(id, self.extraer_template_sd(foto))

    def identificar_sd(self, foto):
        return self._identificar(self.extraer_template_sd(foto))

    def inscribir_t(self, id, template):
        return self._inscribir(id, template)

    def verificar_t(self, id, template):
        return self._verificar(id, template)

    def identificar_t(self, template):
        return self._identificar(template)


# ---------------------------------------------------------- utilidades
def construir_template(pos, J):
    return np.abs(J).ravel().astype(np.float32)


def _partes(tpl):
    """Template -> matriz de magnitudes (N_NODOS, 40)."""
    return np.asarray(tpl, dtype=np.float64).reshape(N_NODOS, N_KERNELS)


def codificar_template(tpl):
    return base64.b64encode(np.asarray(tpl, dtype="<f4").tobytes()).decode("ascii")


def decodificar_template(b64):
    raw = base64.b64decode(b64.encode("ascii"), validate=True)
    if len(raw) != LONG_TEMPLATE * 4:
        raise ValueError("Template base64 con longitud incorrecta")
    return np.frombuffer(raw, dtype="<f4").copy()
