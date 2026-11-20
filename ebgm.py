"""
Extractor y comparador EBGM — Elastic Bunch Graph Matching (Wiskott et al., 1997).

Segunda y tercera etapa del pipeline:

    cara 112x112 alineada -> [EXTRACTOR EBGM] -> template -> [MATCHER EBGM] -> similitud

Piezas del método (y dónde están en este fichero)
-------------------------------------------------
1. Preprocesado fotométrico (`_preprocesar`): ecualización local de contraste
   (CLAHE). EBGM compara texturas locales; igualar exposición y contraste antes
   de los filtros hace los jets más estables entre fotos. (Tan & Triggs también
   está implementado, pero rindió peor en el ajuste: ver ajuste_ebgm.py.)

2. Banco de wavelets de Gabor (`_construir_kernels`): 7 escalas (longitud de
   onda de 5 a 40 px, en pasos de sqrt(2)) x 8 orientaciones, kernels complejos
   con compensación de DC como en Wiskott. Un JET es el vector de 56 magnitudes
   de las respuestas en un punto. Las escalas grandes, que el EBGM clásico de
   5 escalas no tiene, capturan la forma global de la cara y fueron la mejora
   más grande del ajuste.

3. Grafo facial (`_nodos_grafo`): 31 nodos en puntos fiduciales (ojos,
   cejas, nariz, boca, mentón, pómulos, mandíbula, frente...) definidos sobre la
   plantilla de 5 landmarks del alineado. Simétrico izquierda/derecha. (También
   admite rejilla regular o mixto.)

4. Face Bunch Graph (`ModeloEBGM.bunch`): para cada nodo, el "manojo" de jets de
   ese nodo en una cohorte de caras de referencia. Al extraer el template de
   una cara nueva, cada nodo se desplaza ELÁSTICAMENTE dentro de un radio hasta
   la posición cuyo jet se parece más a algún jet del manojo, con una
   penalización por deformación del grafo (`_localizar_nodos`). Así el grafo se
   ajusta a la geometría real de la cara (la "E" y la "B" de EBGM) y corrige
   errores pequeños del alineado.

5. Template: los jets en las posiciones ajustadas (+ las propias posiciones).

6. Matcher (`similitud`): media ponderada, nodo a nodo, del coseno entre jets
   tras compresión (raíz) y normalización por la cohorte (z-score por nodo y
   filtro). Esa normalización resalta lo que distingue a cada identidad de la
   "cara media". Opcionalmente se añade un término de geometría del grafo.
   Invariancia a espejo: también se compara con el grafo REFLEJADO de la sonda
   (permutando nodos izquierda/derecha y orientaciones θ -> π-θ, exacto en el
   espacio del template) y se toma el máximo. Así una foto en espejo (selfie)
   o con la cara girada hacia el otro lado se reconoce (71 % -> 98 % de rank-1
   en espejo) y además mejora el EER general (ver ajuste_ebgm.py).

Los parámetros aprendidos de la cohorte (media, desviación, manojo) forman el
MODELO (`models/ebgm_modelo.npz`), común a todas las identidades. El template de
cada persona (lo que va al .csv) no depende del modelo salvo en la localización
de nodos, que usa el manojo.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(AQUI, "models", "ebgm_modelo.npz")

# Plantilla de 5 landmarks para la cara de 112x112 (la misma que usa detector.py)
# Orden: ojo (izq. de la imagen), ojo (dcha.), nariz, boca (izq.), boca (dcha.)
LM_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                   [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

MAGIA_TEMPLATE = b"EBGM2"   # cabecera del template serializado


# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfigEBGM:
    tam_cara: int = 112
    # --- preprocesado
    preproceso: str = "clahe"          # "ninguno" | "clahe" | "tantriggs"
    # --- banco de Gabor
    n_escalas: int = 7
    n_orientaciones: int = 8
    lambda_min: float = 5.0            # longitud de onda de la escala más fina (px)
    sigma_rel: float = 0.75            # sigma espacial / lambda (Wiskott: 1.0)
    # --- grafo
    grafo: str = "fiducial"            # "rejilla" | "fiducial" | "mixto"
    rejilla: tuple = (7, 6)            # filas, columnas de la rejilla
    margen_rejilla: float = 18.0
    # --- ajuste elástico con el bunch graph
    radio_elastico: int = 3            # px máximos de desplazamiento por nodo
    paso_elastico: int = 1
    penaliz_elastica: float = 0.05     # coste por desplazamiento (normalizado)
    # --- template / matcher
    simetrizar: bool = False
    compresion: str = "sqrt"           # "ninguna" | "sqrt" | "log"
    normalizacion: str = "zscore"      # "ninguna" | "fondo" | "zscore"
    peso_geometria: float = 0.0        # peso del término de geometría del grafo
    variaciones_inscripcion: int = 0   # micro-variaciones promediadas al inscribir (0..8)
    invariante_espejo: bool = True     # similitud = max(directa, con la sonda reflejada)

    @property
    def n_filtros(self) -> int:
        return self.n_escalas * self.n_orientaciones

    def huella(self) -> str:
        """Hash corto de la configuración: liga templates y modelo a una config."""
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True,
                                       default=list).encode()).hexdigest()[:10]


# ----------------------------------------------------------------------------
# Banco de Gabor
# ----------------------------------------------------------------------------
def _construir_kernels(cfg: ConfigEBGM) -> list[tuple[np.ndarray, np.ndarray]]:
    """Kernels complejos (parte real, parte imaginaria) por (escala, orientación).

    Wavelet de Gabor de Wiskott: envolvente gaussiana de desviación sigma_rel·λ
    por la onda plana exp(i k·x), con compensación de DC (respuesta nula a una
    imagen constante -> insensible al brillo global).
    """
    kernels = []
    for v in range(cfg.n_escalas):
        lambd = cfg.lambda_min * (np.sqrt(2.0) ** v)
        k = 2.0 * np.pi / lambd
        s = cfg.sigma_rel * lambd
        r = int(np.ceil(2.5 * s))
        y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
        g = np.exp(-(x * x + y * y) / (2.0 * s * s))
        g /= g.sum()
        for mu in range(cfg.n_orientaciones):
            th = mu * np.pi / cfg.n_orientaciones
            fase = k * (x * np.cos(th) + y * np.sin(th))
            re = g * np.cos(fase)
            im = g * np.sin(fase)
            re -= g * (re.sum() / g.sum())          # compensación de DC
            kernels.append((re.astype(np.float32), im.astype(np.float32)))
    return kernels


def _permutacion_orientaciones(cfg: ConfigEBGM) -> np.ndarray:
    """Índices de filtro tras reflejar la imagen horizontalmente (θ -> π-θ)."""
    perm = []
    for v in range(cfg.n_escalas):
        for mu in range(cfg.n_orientaciones):
            mu2 = (-mu) % cfg.n_orientaciones
            perm.append(v * cfg.n_orientaciones + mu2)
    return np.array(perm)


# ----------------------------------------------------------------------------
# Grafo facial
# ----------------------------------------------------------------------------
def _nodos_fiduciales() -> np.ndarray:
    """Puntos fiduciales en coordenadas de la cara de 112x112 (simétricos)."""
    (ex1, ey1), (ex2, ey2), (nx, ny), (mx1, my1), (mx2, my2) = LM_112
    ey = (ey1 + ey2) / 2
    my = (my1 + my2) / 2
    cx = 56.0
    lado = [  # (dx respecto al centro, y) — se reflejan en x
        (cx - ex1, ey),            # centro del ojo
        (cx - ex1 + 9, ey),        # comisura externa del ojo
        (cx - ex1 - 8, ey),        # comisura interna del ojo
        (cx - ex1, ey - 13),       # centro de la ceja
        (cx - ex1 + 10, ey - 11),  # extremo externo de la ceja
        (cx - ex1, ey + 11),       # ojera
        (9.0, ny + 2),             # aleta de la nariz
        (cx - mx1, my),            # comisura de la boca
        (29.0, ny + 4),            # pómulo
        (33.0, my + 4),            # mandíbula
        (24.0, ey - 26),           # frente lateral
    ]
    centro = [(cx, ey - 26), (cx, ey - 12), (cx, ey), (cx, ny - 8), (cx, ny),
              (cx, my - 6), (cx, my), (cx, my + 7), (cx, my + 15)]
    pts = []
    for dx, y in lado:
        pts.append((cx - dx, y))
        pts.append((cx + dx, y))
    pts.extend(centro)
    return np.array(pts, dtype=np.float32)


def _nodos_grafo(cfg: ConfigEBGM) -> np.ndarray:
    t = cfg.tam_cara
    partes = []
    if cfg.grafo in ("fiducial", "mixto"):
        partes.append(_nodos_fiduciales() * (t / 112.0))
    if cfg.grafo in ("rejilla", "mixto"):
        f, c = cfg.rejilla
        m = cfg.margen_rejilla * (t / 112.0)
        ys = np.linspace(m, t - 1 - m, f)
        xs = np.linspace(m, t - 1 - m, c)
        partes.append(np.array([(x, y) for y in ys for x in xs], np.float32))
    if not partes:
        raise ValueError(f"grafo desconocido: {cfg.grafo}")
    pts = np.concatenate(partes)
    # quitar duplicados casi coincidentes
    out = []
    for p in pts:
        if all(np.hypot(*(p - q)) > 3.0 for q in out):
            out.append(p)
    return np.clip(np.array(out, np.float32), 2, t - 3)


def _permutacion_espejo(pos: np.ndarray, t: int) -> np.ndarray:
    """Para cada nodo, el índice del nodo simétrico (x -> t-1-x)."""
    refl = pos.copy()
    refl[:, 0] = (t - 1) - refl[:, 0]
    d = np.linalg.norm(pos[None, :, :] - refl[:, None, :], axis=2)
    return d.argmin(axis=1)


# ----------------------------------------------------------------------------
# Modelo aprendido de la cohorte
# ----------------------------------------------------------------------------
@dataclass
class ModeloEBGM:
    """Parámetros comunes aprendidos de una cohorte de caras.

    media, desv: (n_nodos, n_filtros) — estadísticos de los jets comprimidos.
    bunch:       (n_bunch, n_nodos, n_filtros) — jets normalizados (L2) del manojo.
    geo_media:   (n_nodos, 2) — posición media de los nodos tras el ajuste.
    """
    media: np.ndarray
    desv: np.ndarray
    bunch: np.ndarray
    geo_media: np.ndarray
    geo_desv: np.ndarray
    huella_cfg: str
    info: dict = field(default_factory=dict)

    def guardar(self, ruta: str = RUTA_MODELO) -> None:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        tmp = ruta + ".tmp.npz"
        np.savez_compressed(tmp, media=self.media, desv=self.desv,
                            bunch=self.bunch.astype(np.float16),
                            geo_media=self.geo_media, geo_desv=self.geo_desv,
                            huella_cfg=np.array(self.huella_cfg),
                            info=np.array(json.dumps(self.info)))
        os.replace(tmp, ruta)

    @classmethod
    def cargar(cls, ruta: str = RUTA_MODELO) -> "ModeloEBGM":
        with np.load(ruta, allow_pickle=False) as z:
            return cls(media=z["media"].astype(np.float32),
                       desv=z["desv"].astype(np.float32),
                       bunch=z["bunch"].astype(np.float32),
                       geo_media=z["geo_media"].astype(np.float32),
                       geo_desv=z["geo_desv"].astype(np.float32),
                       huella_cfg=str(z["huella_cfg"]),
                       info=json.loads(str(z["info"])))

    @staticmethod
    def sha256(ruta: str = RUTA_MODELO) -> str:
        with open(ruta, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()


# ----------------------------------------------------------------------------
# Extractor + matcher
# ----------------------------------------------------------------------------
class ExtractorEBGM:
    """Extrae templates EBGM de caras alineadas y los compara."""

    def __init__(self, cfg: ConfigEBGM | None = None,
                 modelo: ModeloEBGM | str | None = RUTA_MODELO):
        self.cfg = cfg or ConfigEBGM()
        self._kernels = _construir_kernels(self.cfg)
        self.pos0 = _nodos_grafo(self.cfg)
        self.n_nodos = len(self.pos0)
        self._perm_nodos = _permutacion_espejo(self.pos0, self.cfg.tam_cara)
        self._perm_filtros = _permutacion_orientaciones(self.cfg)
        self.pesos = np.ones(self.n_nodos, np.float32) / self.n_nodos
        # desplazamientos candidatos del ajuste elástico
        r, p = self.cfg.radio_elastico, max(1, self.cfg.paso_elastico)
        d = [(dx, dy) for dy in range(-r, r + 1, p) for dx in range(-r, r + 1, p)
             if dx * dx + dy * dy <= r * r]
        self._desplaz = np.array(d, np.int32) if r > 0 else np.zeros((1, 2), np.int32)
        self.modelo: ModeloEBGM | None = None
        if isinstance(modelo, ModeloEBGM):
            self.fijar_modelo(modelo)
        elif isinstance(modelo, str) and os.path.isfile(modelo):
            self.fijar_modelo(ModeloEBGM.cargar(modelo))

    # ------------------------------------------------------------ dimensiones
    @property
    def forma_template(self) -> tuple[int, int]:
        """Jets (n_nodos x n_filtros) + 2 columnas con la posición (x, y)."""
        return (self.n_nodos, self.cfg.n_filtros + 2)

    def fijar_modelo(self, modelo: ModeloEBGM) -> None:
        if modelo.huella_cfg != self.cfg.huella():
            raise ValueError("El modelo EBGM se entrenó con otra configuración "
                             f"({modelo.huella_cfg} != {self.cfg.huella()}); "
                             "vuelve a ejecutar inscribir_dataset.py")
        self.modelo = modelo

    # ------------------------------------------------------------ preproceso
    def _gris(self, cara: np.ndarray) -> np.ndarray:
        if not isinstance(cara, np.ndarray) or cara.size == 0:
            raise ValueError("La cara debe ser un np.ndarray no vacío")
        if cara.ndim == 3 and cara.shape[2] == 3:
            g = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)
        elif cara.ndim == 3 and cara.shape[2] == 4:
            g = cv2.cvtColor(cara, cv2.COLOR_BGRA2GRAY)
        elif cara.ndim == 2:
            g = cara
        else:
            raise ValueError(f"Forma de cara no soportada: {cara.shape}")
        if g.dtype != np.uint8:
            g = np.clip(g, 0, 255).astype(np.uint8)
        t = self.cfg.tam_cara
        if g.shape != (t, t):
            interp = cv2.INTER_AREA if g.shape[0] > t else cv2.INTER_CUBIC
            g = cv2.resize(g, (t, t), interpolation=interp)
        return g

    def _preprocesar(self, g: np.ndarray) -> np.ndarray:
        modo = self.cfg.preproceso
        if modo == "ninguno":
            return g.astype(np.float32)
        if modo == "clahe":
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            return c.apply(g).astype(np.float32)
        if modo == "tantriggs":
            return _tan_triggs(g)
        raise ValueError(f"preproceso desconocido: {modo}")

    # ------------------------------------------------------------ Gabor
    def _mapas(self, img: np.ndarray) -> np.ndarray:
        """Magnitud de Gabor en toda la imagen: (n_filtros, H, W)."""
        out = np.empty((self.cfg.n_filtros,) + img.shape, np.float32)
        for i, (re, im) in enumerate(self._kernels):
            a = cv2.filter2D(img, cv2.CV_32F, re, borderType=cv2.BORDER_REFLECT)
            b = cv2.filter2D(img, cv2.CV_32F, im, borderType=cv2.BORDER_REFLECT)
            out[i] = cv2.magnitude(a, b)
        return out

    def _comprimir(self, j: np.ndarray) -> np.ndarray:
        c = self.cfg.compresion
        if c == "ninguna":
            return j
        if c == "sqrt":
            return np.sqrt(np.maximum(j, 0))
        if c == "log":
            return np.log1p(np.maximum(j, 0))
        raise ValueError(f"compresion desconocida: {c}")

    @staticmethod
    def _l2(x: np.ndarray, eje: int = -1) -> np.ndarray:
        n = np.linalg.norm(x, axis=eje, keepdims=True)
        return x / np.maximum(n, 1e-9)

    # ------------------------------------------------- ajuste elástico (bunch)
    def _localizar_nodos(self, mapas: np.ndarray) -> np.ndarray:
        """Desplaza cada nodo a la posición cuyo jet mejor encaja en el manojo.

        Similitud de un candidato = max_b cos(jet, bunch_b[nodo]) menos una
        penalización proporcional al desplazamiento al cuadrado. Después se
        regulariza el grafo: el desplazamiento de cada nodo se mezcla con la
        mediana de los de sus vecinos (el grafo se deforma, no se rompe).
        """
        t = self.cfg.tam_cara
        if self.modelo is None or self.cfg.radio_elastico <= 0:
            return self.pos0.copy()
        bunch = self.modelo.bunch                       # (B, N, F) normalizado
        r2 = float(max(1, self.cfg.radio_elastico) ** 2)
        pen = self.cfg.penaliz_elastica * (self._desplaz ** 2).sum(1) / r2
        desp = np.zeros((self.n_nodos, 2), np.float32)
        base = np.round(self.pos0).astype(np.int32)
        for n in range(self.n_nodos):
            cand = np.clip(base[n] + self._desplaz, 0, t - 1)
            jets = mapas[:, cand[:, 1], cand[:, 0]].T            # (C, F)
            jets = self._l2(self._comprimir(jets))
            sim = (jets @ bunch[:, n, :].T).max(axis=1) - pen     # (C,)
            desp[n] = self._desplaz[int(np.argmax(sim))]
        # regularización del grafo con los vecinos más cercanos
        d = np.linalg.norm(self.pos0[:, None] - self.pos0[None], axis=2)
        vec = np.argsort(d, axis=1)[:, 1:6]
        suave = 0.5 * desp + 0.5 * np.median(desp[vec], axis=1)
        return np.clip(self.pos0 + suave, 0, t - 1)

    @staticmethod
    def _muestrear(mapas: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Jets en posiciones sub-píxel (interpolación bilineal)."""
        F, H, W = mapas.shape
        x = np.clip(pos[:, 0], 0, W - 1.001)
        y = np.clip(pos[:, 1], 0, H - 1.001)
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        fx = (x - x0)[None]; fy = (y - y0)[None]
        a = mapas[:, y0, x0]; b = mapas[:, y0, x0 + 1]
        c = mapas[:, y0 + 1, x0]; d = mapas[:, y0 + 1, x0 + 1]
        return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy).T

    def _grafo_imagen(self, g: np.ndarray, localizar: bool = True):
        mapas = self._mapas(self._preprocesar(g))
        pos = self._localizar_nodos(mapas) if localizar else self.pos0.copy()
        return self._muestrear(mapas, pos), pos

    # ------------------------------------------------------------ extracción
    def extraer(self, cara: np.ndarray, localizar: bool = True) -> np.ndarray:
        """Cara alineada -> template EBGM (n_nodos, n_filtros + 2) float32.

        Columnas 0..F-1: magnitudes de Gabor (jet crudo) en cada nodo.
        Columnas F, F+1: posición (x, y) del nodo tras el ajuste elástico.
        """
        g = self._gris(cara)
        jets, pos = self._grafo_imagen(g, localizar)
        if self.cfg.simetrizar:
            jets_e, pos_e = self._grafo_imagen(np.ascontiguousarray(g[:, ::-1]), localizar)
            # reordenar el grafo del espejo para que corresponda nodo a nodo
            jets_e = jets_e[self._perm_nodos][:, self._perm_filtros]
            pos_e = pos_e[self._perm_nodos].copy()
            pos_e[:, 0] = (self.cfg.tam_cara - 1) - pos_e[:, 0]
            jets = 0.5 * (jets + jets_e)
            pos = 0.5 * (pos + pos_e)
        return np.concatenate([jets, pos], axis=1).astype(np.float32)

    def extraer_inscripcion(self, cara: np.ndarray) -> np.ndarray:
        """Template de INSCRIPCIÓN: media de la cara y de micro-variaciones.

        Con una sola foto inscrita, un error de alineado de 1-2 px o unos grados
        en la sonda cuesta mucha similitud. Se promedian los templates de la
        cara desplazada (±2 px), escalada (±4 %) y girada (±4°): el template
        resultante es más estable. Sigue siendo 1 template de la misma forma.
        """
        g = self._gris(cara)
        if self.cfg.variaciones_inscripcion <= 0:
            return self.extraer(g)
        t = self.cfg.tam_cara
        c = (t / 2.0, t / 2.0)
        transf = [(0, 0, 1.0, 0), (2, 0, 1.0, 0), (-2, 0, 1.0, 0), (0, 2, 1.0, 0),
                  (0, -2, 1.0, 0), (0, 0, 1.04, 0), (0, 0, 0.96, 0), (0, 0, 1.0, 4),
                  (0, 0, 1.0, -4)][:1 + self.cfg.variaciones_inscripcion]
        tpls = []
        for dx, dy, esc, ang in transf:
            M = cv2.getRotationMatrix2D(c, ang, esc)
            M[:, 2] += (dx, dy)
            gi = cv2.warpAffine(g, M, (t, t), borderMode=cv2.BORDER_REPLICATE)
            tpls.append(self.extraer(gi))
        return self.promediar(tpls)

    # ------------------------------------------------------------ cohorte
    def entrenar_modelo(self, caras: list[np.ndarray], n_bunch: int | None = None,
                        info: dict | None = None) -> ModeloEBGM:
        """Aprende el modelo (manojo + estadísticos) de una cohorte de caras.

        1) El manojo se forma con los jets de cada cara (y su espejo) en las
           posiciones nominales del grafo.
        2) Con el manojo ya disponible se extraen los templates definitivos
           (con ajuste elástico) y de ellos salen media/desviación por nodo.
        """
        if len(caras) < 5:
            raise ValueError("Se necesitan al menos 5 caras para el modelo")
        grises = [self._gris(c) for c in caras]
        grises += [np.ascontiguousarray(g[:, ::-1]) for g in grises]
        jets = np.stack([self._grafo_imagen(g, localizar=False)[0] for g in grises])
        bunch = self._l2(self._comprimir(jets))
        if n_bunch and len(bunch) > n_bunch:
            idx = np.linspace(0, len(bunch) - 1, n_bunch).astype(int)
            bunch = bunch[idx]
        F = self.cfg.n_filtros
        provisional = ModeloEBGM(
            media=np.zeros((self.n_nodos, F), np.float32),
            desv=np.ones((self.n_nodos, F), np.float32),
            bunch=bunch.astype(np.float32), geo_media=self.pos0.copy(),
            geo_desv=np.ones_like(self.pos0), huella_cfg=self.cfg.huella())
        anterior = self.modelo
        self.modelo = provisional
        try:
            tpls = np.stack([self.extraer(c) for c in caras])
        finally:
            self.modelo = anterior
        j = self._comprimir(tpls[:, :, :F])
        geo = tpls[:, :, F:]
        modelo = ModeloEBGM(
            media=j.mean(0).astype(np.float32),
            desv=(j.std(0) + 1e-3 * j.std(0).mean() + 1e-9).astype(np.float32),
            bunch=bunch.astype(np.float32),
            geo_media=geo.mean(0).astype(np.float32),
            geo_desv=(geo.std(0) + 0.5).astype(np.float32),
            huella_cfg=self.cfg.huella(),
            info=dict(info or {}, n_cohorte=len(caras)))
        return modelo

    # ------------------------------------------------------------ matcher
    def reflejar(self, t) -> np.ndarray:
        """Template de la cara reflejada horizontalmente (exacto, sin re-extraer)."""
        t = self.como_template(t)
        F = self.cfg.n_filtros
        m = t[self._perm_nodos][:, list(self._perm_filtros) + [F, F + 1]].copy()
        m[:, F] = (self.cfg.tam_cara - 1) - m[:, F]
        return m

    def _rep_simple(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        F = self.cfg.n_filtros
        j = self._comprimir(t[:, :F])
        modo = self.cfg.normalizacion
        if modo != "ninguna" and self.modelo is None:
            raise RuntimeError("Falta el modelo EBGM (models/ebgm_modelo.npz); "
                               "ejecuta inscribir_dataset.py")
        if modo == "fondo":
            j = j - self.modelo.media
        elif modo == "zscore":
            j = (j - self.modelo.media) / self.modelo.desv
        elif modo != "ninguna":
            raise ValueError(f"normalizacion desconocida: {modo}")
        geo = t[:, F:]
        if self.modelo is not None:
            geo = (geo - self.modelo.geo_media) / self.modelo.geo_desv
        return self._l2(j), geo

    def representar(self, t) -> tuple:
        """Template -> representación para comparar: (jets, geometría) normalizados
        del template y, si invariante_espejo, también los de su reflejo."""
        t = self.como_template(t)
        r = self._rep_simple(t)
        if self.cfg.invariante_espejo:
            r = r + self._rep_simple(self.reflejar(t))
        return r

    def _nucleo(self, ja, ga, J, G) -> np.ndarray:
        s = np.einsum("nf,knf->kn", ja, J) @ self.pesos
        if self.cfg.peso_geometria > 0:
            s = s - self.cfg.peso_geometria * np.sqrt(
                np.mean(np.sum((G - ga) ** 2, axis=2), axis=1))
        return s

    def similitudes_rep(self, sonda, galeria: list) -> np.ndarray:
        """Similitud de una sonda (representada) con varias entradas de galería
        (representadas). Vectorizado. Si invariante_espejo, máximo entre la
        sonda directa y la sonda reflejada."""
        if not galeria:
            return np.zeros(0, np.float64)
        J = np.stack([g[0] for g in galeria])
        G = np.stack([g[1] for g in galeria])
        s = self._nucleo(sonda[0], sonda[1], J, G)
        if self.cfg.invariante_espejo and len(sonda) == 4:
            s = np.maximum(s, self._nucleo(sonda[2], sonda[3], J, G))
        return s.astype(np.float64)

    def similitud_rep(self, ra, rb) -> float:
        """Similitud entre dos templates ya representados (ra = sonda)."""
        return float(self.similitudes_rep(ra, [rb])[0])

    def similitud(self, t1, t2) -> float:
        """Similitud EBGM entre dos templates. Mayor = más parecidos (máx. 1).
        t1 hace de sonda y t2 de galería (con invariancia a espejo el resultado
        es prácticamente simétrico)."""
        return self.similitud_rep(self.representar(t1), self.representar(t2))

    def similitudes(self, t, galeria: list) -> np.ndarray:
        """Similitud de un template contra muchos (vectorizado)."""
        return self.similitudes_rep(self.representar(t), [self.representar(g) for g in galeria])

    def promediar(self, templates: list) -> np.ndarray:
        """Template medio de varias fotos de la MISMA persona.

        Se promedia en el espacio comprimido (donde el matcher es lineal antes
        de normalizar) y se deshace la compresión: el resultado es un template
        normal, de la misma forma que cualquier otro (1 template por persona).
        """
        T = np.stack([self.como_template(t) for t in templates])
        F = self.cfg.n_filtros
        c = self._comprimir(T[:, :, :F]).mean(0)
        if self.cfg.compresion == "sqrt":
            j = c ** 2
        elif self.cfg.compresion == "log":
            j = np.expm1(c)
        else:
            j = c
        return np.concatenate([j, T[:, :, F:].mean(0)], axis=1).astype(np.float32)

    def puntuacion_cara(self, t) -> float:
        """"Caricidad" del template según el Face Bunch Graph.

        Media, sobre los nodos, de la mejor similitud del jet del nodo con los
        jets del manojo en ese mismo nodo (el criterio con el que EBGM busca
        caras). Separa con margen la BASURA TRIVIAL (imágenes planas, ruido,
        gradientes, patrones periódicos, templates aleatorios) de las caras.
        LÍMITE CONOCIDO (medido en inscribir_dataset.py): con una cohorte de
        49 caras NO separa caras no vistas de caras invertidas/giradas ni de
        texturas naturales (fondos); eso lo filtra el detector en el pipeline
        completo, pero no en _sd/_t.
        """
        if self.modelo is None:
            raise RuntimeError("Falta el modelo EBGM")
        t = self.como_template(t)
        j = self._l2(self._comprimir(t[:, :self.cfg.n_filtros]))     # (N, F)
        s = np.einsum("nf,bnf->bn", j, self.modelo.bunch).max(axis=0)
        return float(s.mean())

    # ------------------------------------------------- validación / formato
    def como_template(self, t) -> np.ndarray:
        """Convierte y VALIDA un template (array o cadena serializada)."""
        if isinstance(t, (str, bytes)):
            t = self.deserializar(t)
        if not isinstance(t, np.ndarray):
            try:
                t = np.asarray(t, dtype=np.float32)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Template no convertible a array: {e}") from None
        if t.dtype == object:
            raise ValueError("Template con tipo object")
        t = t.astype(np.float32, copy=False)
        if t.shape != self.forma_template:
            raise ValueError(f"Template con forma {t.shape}, se esperaba "
                             f"{self.forma_template}")
        if not np.all(np.isfinite(t)):
            raise ValueError("Template con valores NaN/Inf")
        F = self.cfg.n_filtros
        if np.any(t[:, :F] < 0):
            raise ValueError("Template inválido: magnitudes de Gabor negativas")
        if np.any(t[:, F:] < -1) or np.any(t[:, F:] > self.cfg.tam_cara):
            raise ValueError("Template inválido: posiciones de nodo fuera de la cara")
        if float(t[:, :F].max()) > 1e4:
            raise ValueError("Template inválido: magnitudes fuera de rango")
        return t

    def serializar(self, template) -> str:
        """Template -> cadena base64 (cabecera + huella de config + float32)."""
        t = self.como_template(template)
        cab = MAGIA_TEMPLATE + self.cfg.huella().encode("ascii")
        return base64.b64encode(cab + t.tobytes()).decode("ascii")

    def deserializar(self, cadena) -> np.ndarray:
        if isinstance(cadena, str):
            cadena = cadena.encode("ascii")
        try:
            buf = base64.b64decode(cadena, validate=True)
        except Exception:
            raise ValueError("Template serializado: base64 inválido") from None
        cab = MAGIA_TEMPLATE + self.cfg.huella().encode("ascii")
        if not buf.startswith(cab):
            raise ValueError("Template serializado con otra versión/config de EBGM")
        datos = buf[len(cab):]
        n = int(np.prod(self.forma_template)) * 4
        if len(datos) != n:
            raise ValueError("Template serializado con longitud incorrecta")
        return np.frombuffer(datos, np.float32).reshape(self.forma_template).copy()


# ----------------------------------------------------------------------------
def _tan_triggs(g: np.ndarray, gamma: float = 0.2, s0: float = 1.0,
                s1: float = 2.0, alfa: float = 0.1, tau: float = 10.0) -> np.ndarray:
    """Normalización de iluminación de Tan & Triggs (2010)."""
    x = (g.astype(np.float32) / 255.0 + 1e-3) ** gamma
    x = cv2.GaussianBlur(x, (0, 0), s0) - cv2.GaussianBlur(x, (0, 0), s1)
    x = x / (np.mean(np.abs(x) ** alfa) ** (1 / alfa) + 1e-6)
    x = x / (np.mean(np.minimum(tau, np.abs(x)) ** alfa) ** (1 / alfa) + 1e-6)
    x = tau * np.tanh(x / tau)
    return (x * 20.0).astype(np.float32)   # escala ~ la de una imagen 0..255


if __name__ == "__main__":
    ext = ExtractorEBGM(modelo=None)
    cara = (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
    tpl = ext.extraer(cara)
    print("nodos:", ext.n_nodos, "| filtros:", ext.cfg.n_filtros,
          "| template:", tpl.shape, "| huella cfg:", ext.cfg.huella())
