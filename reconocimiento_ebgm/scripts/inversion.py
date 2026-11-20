"""Ataque de inversion de template: reconstruye una imagen a partir de un
template leido de la BD (posiciones de los nodos + jets complejos).

La respuesta de Gabor es lineal en la imagen, asi que J = A(I). Se resuelve
    min_I ||A I - J||^2 + l1 ||grad I||^2 + l2 ||I - mu||^2
por gradiente conjugado, donde mu es una cara media obtenida de caras publicas
(el atacante no necesita la BD para construirla)."""
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from ebgm.gabor import BancoGabor, N_KERNELS
from ebgm.normalizacion import TAM


def _pesos_bilineales(pos):
    x = np.clip(pos[:, 0], 0, TAM - 1.001)
    y = np.clip(pos[:, 1], 0, TAM - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - x0, y - y0
    return [(y0, x0, (1 - dx) * (1 - dy)), (y0, x0 + 1, dx * (1 - dy)),
            (y0 + 1, x0, (1 - dx) * dy), (y0 + 1, x0 + 1, dx * dy)]


class OperadorJets:
    def __init__(self, pos, banco=None):
        self.banco = banco or BancoGabor(TAM)
        self.w = _pesos_bilineales(pos)
        self.n = len(pos)

    def directo(self, img):
        R = self.banco.transformar(img)
        return sum(R[:, yy, xx].T * ww[:, None] for yy, xx, ww in self.w)  # (N,40)

    def adjunto(self, c):
        mapas = np.zeros((N_KERNELS, TAM, TAM), dtype=np.complex128)
        for yy, xx, ww in self.w:
            for k in range(N_KERNELS):
                np.add.at(mapas[k], (yy, xx), c[:, k] * ww)
        return np.real(self.banco.adjunto(mapas))


def _lap(img):
    p = np.pad(img, 1, mode="edge")
    return 4 * img - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:]


def invertir(pos, J, mu, l1=0.5, l2=0.05, iters=300):
    op = OperadorJets(pos)
    escala = 1.0 / (np.abs(J).mean() + 1e-9)       # equilibra los terminos

    def normal(v):
        img = v.reshape(TAM, TAM)
        r = escala ** 2 * op.adjunto(op.directo(img)) + l1 * _lap(img) + l2 * img
        return r.ravel()

    b = (escala ** 2 * op.adjunto(J) + l2 * mu).ravel()
    L = LinearOperator((TAM * TAM, TAM * TAM), matvec=normal, dtype=np.float64)
    x, _ = cg(L, b, x0=mu.ravel(), maxiter=iters)
    return x.reshape(TAM, TAM)


def a_uint8(img):
    z = (img - img.mean()) / (img.std() + 1e-9)
    return np.clip(128 + 45 * z, 0, 255).astype(np.uint8)
