"""
Extractor de características EBGM — Elastic Bunch Graph Matching.

Segunda etapa del pipeline:
    cara 112x112 alineada -> [EXTRACTOR EBGM] -> template (grafo de jets de Gabor)

Idea del método
---------------
  - Sobre la cara alineada se coloca un GRAFO de nodos (una rejilla regular).
  - En cada nodo se calcula un JET de Gabor: las respuestas de un banco de
    filtros de Gabor a varias escalas (frecuencias) y orientaciones. Nos
    quedamos con la MAGNITUD de cada respuesta -> un vector por nodo.
  - El TEMPLATE de una persona es el conjunto de jets de todos los nodos.

Normalización por fondo
-----------------------
  La magnitud de Gabor es muy parecida entre caras distintas (todas comparten
  bordes de ojos, nariz, boca...). Para resaltar lo propio de cada identidad se
  resta la "cara media" (el jet medio de una cohorte de fondo) antes de comparar.
  La cara media es un parámetro del MODELO (no de la persona): se calcula una vez
  sobre los 50 famosos comunes y se guarda en models/ebgm_fondo.npy. Por eso el
  template que se guarda en el .csv son los jets en crudo, y `inscribir_t` sólo
  necesita esos jets.

Emparejamiento (matcher)
------------------------
  Similitud entre dos grafos = media, nodo a nodo, del coseno de sus jets
  (tras restar el fondo y normalizar). Con `elastico=True` cada nodo puede
  desplazarse a nodos vecinos del otro grafo (deformación elástica, de ahí
  "elastic graph matching"), con una penalización por desplazamiento.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

import cv2
import numpy as np

RUTA_FONDO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "ebgm_fondo.npy")


# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfigEBGM:
    tam_cara: int = 112          # lado de la cara de entrada (cuadrada)
    n_orientaciones: int = 8     # orientaciones del banco de Gabor
    n_escalas: int = 5           # escalas (frecuencias) del banco
    ksize: int = 31              # tamaño del kernel de Gabor
    margen: int = 16             # margen en px de la rejilla al borde
    filas: int = 9               # nodos por columna
    columnas: int = 8            # nodos por fila
    radio_elastico: float = 1.6  # radio de búsqueda elástica (en pasos de rejilla)
    penaliz_desplaz: float = 0.10  # penalización por desplazamiento normalizado

    @property
    def n_filtros(self) -> int:
        return self.n_orientaciones * self.n_escalas

    @property
    def n_nodos(self) -> int:
        return self.filas * self.columnas


def _construir_kernels(cfg: ConfigEBGM) -> list[tuple[np.ndarray, np.ndarray]]:
    """Lista de (kernel_real, kernel_imag) para cada (escala, orientación)."""
    kernels = []
    lambdas = [4.0 * (np.sqrt(2.0) ** i) for i in range(cfg.n_escalas)]
    for lambd in lambdas:
        sigma = 0.56 * lambd                      # ancho de banda ~1 octava
        for o in range(cfg.n_orientaciones):
            theta = o * np.pi / cfg.n_orientaciones
            comun = dict(ksize=(cfg.ksize, cfg.ksize), sigma=sigma,
                         theta=theta, lambd=lambd, gamma=1.0)
            re = cv2.getGaborKernel(psi=0.0, **comun)            # parte par (cos)
            im = cv2.getGaborKernel(psi=-np.pi / 2.0, **comun)   # parte impar (sin)
            re = re - re.mean()   # media nula -> insensible a iluminación constante
            im = im - im.mean()
            kernels.append((re.astype(np.float32), im.astype(np.float32)))
    return kernels


def _posiciones_nodos(cfg: ConfigEBGM) -> np.ndarray:
    """Posiciones (x, y) de los nodos de la rejilla. Shape (n_nodos, 2)."""
    m, t = cfg.margen, cfg.tam_cara
    ys = np.linspace(m, t - 1 - m, cfg.filas)
    xs = np.linspace(m, t - 1 - m, cfg.columnas)
    return np.array([(x, y) for y in ys for x in xs], dtype=np.float32)


class ExtractorEBGM:
    """Extrae y compara templates EBGM (grafos de jets de Gabor)."""

    def __init__(self, cfg: ConfigEBGM | None = None, cargar_fondo: bool = True):
        self.cfg = cfg or ConfigEBGM()
        self._kernels = _construir_kernels(self.cfg)
        self._pos = _posiciones_nodos(self.cfg)
        self._paso = (self.cfg.tam_cara - 2 * self.cfg.margen) / max(
            self.cfg.filas, self.cfg.columnas)
        self._vecinos = self._precalcular_vecinos()
        # cara media de fondo (jets crudos); None -> no se resta nada
        self.media: np.ndarray | None = None
        if cargar_fondo and os.path.isfile(RUTA_FONDO):
            self.cargar_fondo()

    # ------------------------------------------------------------- extracción
    def _preparar(self, cara: np.ndarray) -> np.ndarray:
        """Cara BGR/gris -> gris float32 de tam_cara x tam_cara."""
        if cara is None or not isinstance(cara, np.ndarray):
            raise ValueError("La cara debe ser un np.ndarray")
        if cara.ndim == 3:
            gris = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)
        elif cara.ndim == 2:
            gris = cara
        else:
            raise ValueError(f"Forma de cara no soportada: {cara.shape}")
        t = self.cfg.tam_cara
        if gris.shape[:2] != (t, t):
            gris = cv2.resize(gris, (t, t), interpolation=cv2.INTER_AREA)
        return gris.astype(np.float32)

    def extraer(self, cara: np.ndarray) -> np.ndarray:
        """Cara 112x112 alineada -> template EBGM (jets crudos).

        Devuelve np.ndarray (n_nodos, n_filtros) float32 con la magnitud de Gabor
        en cada nodo. La normalización (resta de fondo + L2) se hace al comparar.
        """
        gris = self._preparar(cara)
        mapas = np.empty((self.cfg.n_filtros,) + gris.shape, dtype=np.float32)
        for k, (re, im) in enumerate(self._kernels):
            r = cv2.filter2D(gris, cv2.CV_32F, re)
            i = cv2.filter2D(gris, cv2.CV_32F, im)
            mapas[k] = cv2.magnitude(r, i)
        jets = np.empty((self.cfg.n_nodos, self.cfg.n_filtros), dtype=np.float32)
        for n, (x, y) in enumerate(self._pos):
            jets[n] = mapas[:, int(round(y)), int(round(x))]
        return jets

    # ----------------------------------------------------------- fondo/cohorte
    def ajustar_fondo(self, templates: list[np.ndarray] | np.ndarray) -> None:
        """Calcula la cara media (jet medio por nodo) sobre una cohorte."""
        arr = np.stack([self._como_template(t) for t in templates])
        self.media = arr.mean(axis=0).astype(np.float32)

    def guardar_fondo(self, ruta: str = RUTA_FONDO) -> None:
        if self.media is None:
            raise ValueError("No hay fondo calculado; usa ajustar_fondo primero")
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        np.save(ruta, self.media)

    def cargar_fondo(self, ruta: str = RUTA_FONDO) -> None:
        m = np.load(ruta)
        if m.shape != (self.cfg.n_nodos, self.cfg.n_filtros):
            raise ValueError(f"Fondo con forma {m.shape} incompatible con la config")
        self.media = m.astype(np.float32)

    def _normalizar(self, t: np.ndarray) -> np.ndarray:
        """Resta el fondo (si lo hay) y normaliza cada jet a norma 1."""
        v = t if self.media is None else (t - self.media)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return v / n

    # ------------------------------------------------------------ comparación
    def _precalcular_vecinos(self) -> list[tuple[np.ndarray, np.ndarray]]:
        radio_px = self.cfg.radio_elastico * self._paso
        vecinos = []
        for i in range(self.cfg.n_nodos):
            d = np.linalg.norm(self._pos - self._pos[i], axis=1)
            idx = np.where(d <= radio_px)[0]
            pen = 1.0 - self.cfg.penaliz_desplaz * (d[idx] / max(radio_px, 1e-6))
            vecinos.append((idx, pen.astype(np.float32)))
        return vecinos

    def similitud(self, t1, t2, elastico: bool = False) -> float:
        """Similitud entre dos templates EBGM. Mayor = más parecidos.

        Con fondo restado, los valores quedan centrados en ~0: genuinos > 0 y
        la mayoría de impostores <= 0. Con elastico=True cada nodo puede
        emparejarse con nodos vecinos del otro grafo (deformación elástica).
        """
        a = self._normalizar(self._como_template(t1))
        b = self._normalizar(self._como_template(t2))
        if not elastico:
            return float(np.mean(np.sum(a * b, axis=1)))
        C = a @ b.T
        total = 0.0
        for i, (idx, pen) in enumerate(self._vecinos):
            total += float(np.max(C[i, idx] * pen))
        return total / self.cfg.n_nodos

    # --------------------------------------------------- (de)serialización CSV
    def serializar(self, template: np.ndarray) -> str:
        """Template -> cadena base64 (para guardar en el .csv)."""
        return base64.b64encode(
            self._como_template(template).tobytes()).decode("ascii")

    def deserializar(self, cadena: str) -> np.ndarray:
        """Cadena base64 del .csv -> template np.ndarray (n_nodos, n_filtros)."""
        buf = base64.b64decode(cadena.encode("ascii"))
        arr = np.frombuffer(buf, dtype=np.float32)
        return arr.reshape(self.cfg.n_nodos, self.cfg.n_filtros).copy()

    def _como_template(self, t) -> np.ndarray:
        if isinstance(t, str):
            t = self.deserializar(t)
        t = np.asarray(t, dtype=np.float32)
        esperado = (self.cfg.n_nodos, self.cfg.n_filtros)
        if t.shape != esperado:
            raise ValueError(f"Template con forma {t.shape}, se esperaba {esperado}")
        return t


if __name__ == "__main__":
    ext = ExtractorEBGM(cargar_fondo=False)
    cara = (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
    tpl = ext.extraer(cara)
    print("template:", tpl.shape, "| nodos x filtros =",
          ext.cfg.n_nodos, "x", ext.cfg.n_filtros)
    print("similitud consigo mismo:", round(ext.similitud(tpl, tpl), 4))
    print("serializado (40 chars):", ext.serializar(tpl)[:40], "...")
