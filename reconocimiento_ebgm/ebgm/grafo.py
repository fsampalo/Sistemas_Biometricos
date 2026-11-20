"""Definicion del grafo facial y del Face Bunch Graph (FBG)."""
import numpy as np
from scipy.spatial import Delaunay

# Nodos definidos a partir del esquema de 68 puntos (iBUG 300-W). Cada nodo es
# la media de los indices indicados; los de la frente se generan desplazando
# las cejas hacia arriba.
NODOS = {
    "mandibula_1": [2], "mandibula_2": [5], "menton": [8], "mandibula_4": [11], "mandibula_5": [14],
    "ceja_a_ext": [17], "ceja_a_med": [19], "ceja_a_int": [21],
    "ceja_b_int": [22], "ceja_b_med": [24], "ceja_b_ext": [26],
    "puente_nariz": [27], "punta_nariz": [30], "ala_nariz_a": [31], "base_nariz": [33], "ala_nariz_b": [35],
    "ojo_a_ext": [36], "ojo_a_int": [39], "pupila_a": [37, 38, 40, 41],
    "ojo_b_int": [42], "ojo_b_ext": [45], "pupila_b": [43, 44, 46, 47],
    "boca_a": [48], "labio_sup": [51], "boca_b": [54], "labio_inf": [57], "centro_boca": [62, 66],
    "mejilla_a": [2, 31], "mejilla_b": [14, 35],
    "frente_a": "F19", "frente_c": "F21_22", "frente_b": "F24", "entrecejo": [21, 22],
}
NOMBRES_NODOS = list(NODOS.keys())
N_NODOS = len(NOMBRES_NODOS)


def nodos_desde_68(lm: np.ndarray) -> np.ndarray:
    """lm (68,2) -> posiciones de los nodos (N,2)."""
    pa = lm[[37, 38, 40, 41]].mean(0)
    pb = lm[[43, 44, 46, 47]].mean(0)
    iod = np.linalg.norm(pb - pa)
    arriba = np.array([0.0, -0.45 * iod])
    pos = []
    for nombre in NOMBRES_NODOS:
        spec = NODOS[nombre]
        if spec == "F19":
            pos.append(lm[19] + arriba)
        elif spec == "F24":
            pos.append(lm[24] + arriba)
        elif spec == "F21_22":
            pos.append(lm[[21, 22]].mean(0) + 1.1 * arriba)
        else:
            pos.append(lm[spec].mean(0))
    return np.array(pos)


def aristas_delaunay(pos: np.ndarray) -> np.ndarray:
    tri = Delaunay(pos)
    ar = set()
    for s in tri.simplices:
        for i in range(3):
            a, b = sorted((s[i], s[(i + 1) % 3]))
            ar.add((a, b))
    return np.array(sorted(ar), dtype=int)


class BunchGraph:
    """FBG: posiciones medias de los nodos, aristas y un 'manojo' de jets por nodo."""

    def __init__(self, pos_media, aristas, jets):
        self.pos_media = np.asarray(pos_media, float)          # (N,2)
        self.aristas = np.asarray(aristas, int)                 # (E,2)
        self.jets = np.asarray(jets)                            # (B,N,40) complejo
        mag = np.abs(self.jets)
        self.mag_norm = mag / (np.linalg.norm(mag, axis=-1, keepdims=True) + 1e-12)
        self.vec_aristas = self.pos_media[self.aristas[:, 1]] - self.pos_media[self.aristas[:, 0]]

    def guardar(self, ruta):
        np.savez_compressed(ruta, pos_media=self.pos_media, aristas=self.aristas,
                            jets_re=self.jets.real.astype(np.float32),
                            jets_im=self.jets.imag.astype(np.float32))

    @classmethod
    def cargar(cls, ruta):
        d = np.load(ruta, allow_pickle=False)
        return cls(d["pos_media"], d["aristas"], d["jets_re"] + 1j * d["jets_im"])
