"""Banco de filtros de Gabor y extraccion de jets (Wiskott et al., 1997).

Cada jet tiene 40 coeficientes complejos: 5 frecuencias x 8 orientaciones.
    k_v   = 2^{-(v+2)/2} * pi            v = 0..4
    phi_u = u * pi / 8                   u = 0..7
    sigma = 2*pi
Los filtros se construyen directamente en el dominio de la frecuencia
(transformada analitica del Gabor con termino DC eliminado) y se aplican por FFT.
"""
import numpy as np

N_FREQ = 5
N_ORIENT = 8
N_KERNELS = N_FREQ * N_ORIENT
SIGMA = 2 * np.pi


def _wave_vectors():
    ks = []
    for v in range(N_FREQ):
        kv = 2.0 ** (-(v + 2) / 2.0) * np.pi
        for u in range(N_ORIENT):
            phi = u * np.pi / N_ORIENT
            ks.append((kv * np.cos(phi), kv * np.sin(phi)))
    return np.array(ks, dtype=np.float64)  # (40, 2) -> (kx, ky)


K_VECTORS = _wave_vectors()


class BancoGabor:
    """Aplica los 40 filtros sobre una imagen canonica de tamano fijo."""

    def __init__(self, tam_imagen: int, tam_fft: int = 256):
        self.tam = tam_imagen
        self.n = tam_fft
        wy = np.fft.fftfreq(self.n) * 2 * np.pi
        wx = np.fft.fftfreq(self.n) * 2 * np.pi
        WX, WY = np.meshgrid(wx, wy)
        filtros = []
        for kx, ky in K_VECTORS:
            k2 = kx * kx + ky * ky
            s2 = SIGMA ** 2
            g = np.exp(-s2 * ((WX - kx) ** 2 + (WY - ky) ** 2) / (2 * k2)) \
                - np.exp(-s2 * (WX ** 2 + WY ** 2 + k2) / (2 * k2))
            filtros.append(g)
        self.filtros = np.stack(filtros).astype(np.complex128)  # (40, n, n)

    def transformar(self, img: np.ndarray) -> np.ndarray:
        """img (tam x tam, float) -> respuestas complejas (40, tam, tam)."""
        pad = np.zeros((self.n, self.n), dtype=np.float64)
        pad[: self.tam, : self.tam] = img
        F = np.fft.fft2(pad)
        R = np.fft.ifft2(F[None] * self.filtros)
        return R[:, : self.tam, : self.tam]

    def adjunto(self, coef_mapas: np.ndarray) -> np.ndarray:
        """Operador adjunto de transformar (usado en el ataque de inversion)."""
        pad = np.zeros((N_KERNELS, self.n, self.n), dtype=np.complex128)
        pad[:, : self.tam, : self.tam] = coef_mapas
        F = np.fft.fft2(pad)
        R = np.fft.ifft2(F * np.conj(self.filtros)).sum(0)
        # conj(filtro) porque transformar usa conv. circular con kernel complejo
        return R[: self.tam, : self.tam]


def muestrear_jets(respuestas: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Interpolacion bilineal de las respuestas en posiciones (N,2) [x,y].
    Devuelve (N, 40) complejo."""
    _, h, w = respuestas.shape
    x = np.clip(pos[:, 0], 0, w - 1.001)
    y = np.clip(pos[:, 1], 0, h - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    dx = (x - x0)[:, None]
    dy = (y - y0)[:, None]
    r = respuestas
    j = (r[:, y0, x0].T * (1 - dx) * (1 - dy) + r[:, y0, x0 + 1].T * dx * (1 - dy)
         + r[:, y0 + 1, x0].T * (1 - dx) * dy + r[:, y0 + 1, x0 + 1].T * dx * dy)
    return j


def sim_magnitud(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """S_a: similitud de magnitudes (coseno). a,b (..., 40) magnitudes >= 0."""
    num = (a * b).sum(-1)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1)) + 1e-12
    return num / den


def estimar_desplazamiento(j1: np.ndarray, j2: np.ndarray) -> np.ndarray:
    """Estimacion del desplazamiento d (Wiskott) que maximiza la similitud de fase
    entre dos jets complejos. j1, j2 (40,). Devuelve (dx, dy)."""
    a1, a2 = np.abs(j1), np.abs(j2)
    dphi = np.angle(j1) - np.angle(j2)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    kx, ky = K_VECTORS[:, 0], K_VECTORS[:, 1]
    w = a1 * a2
    Gxx = (w * kx * kx).sum(); Gxy = (w * kx * ky).sum(); Gyy = (w * ky * ky).sum()
    Phx = (w * kx * dphi).sum(); Phy = (w * ky * dphi).sum()
    det = Gxx * Gyy - Gxy * Gxy
    if abs(det) < 1e-12:
        return np.zeros(2)
    dx = (Gyy * Phx - Gxy * Phy) / det
    dy = (-Gxy * Phx + Gxx * Phy) / det
    return np.array([dx, dy])
