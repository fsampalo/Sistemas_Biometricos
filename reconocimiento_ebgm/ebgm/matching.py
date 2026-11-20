"""Elastic Bunch Graph Matching: localizacion de los nodos en una imagen nueva.

Fases (Wiskott et al., 1997):
 1. Busqueda global rigida: se desplaza y escala el grafo medio maximizando la
    similitud con el FBG (sin distorsion).
 2. Distorsion local: cada nodo se mueve independientemente dentro de una
    ventana maximizando  S_B = max_b S_a(J, J_b) - lambda * coste_elastico.
 3. Refinamiento subpixel por estimacion de desplazamiento con la fase.
"""
import numpy as np
from .gabor import muestrear_jets, estimar_desplazamiento, sim_magnitud

PARAMS = {
    "detector": dict(tx=6, paso=2, escalas=np.linspace(0.9, 1.1, 5), ventana=4, lam=0.6, barridos=2),
    "sin_detector": dict(tx=24, paso=3, escalas=np.linspace(0.6, 1.3, 11), ventana=5, lam=0.6, barridos=3),
}


def _mag_norm(R):
    m = np.abs(R)
    return m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-12)


def ajustar_grafo(R, fbg, modo="detector"):
    """R: respuestas Gabor (40,T,T). Devuelve (posiciones (N,2), similitud FBG)."""
    p = PARAMS[modo]
    _, T, _ = R.shape
    Mn = _mag_norm(R)                         # (40,T,T)
    P0 = fbg.pos_media
    c = P0.mean(0)
    B = fbg.mag_norm                          # (B,N,40)

    def score_global(pos):
        xi = np.clip(np.round(pos[:, 0]).astype(int), 0, T - 1)
        yi = np.clip(np.round(pos[:, 1]).astype(int), 0, T - 1)
        J = Mn[:, yi, xi].T                   # (N,40)
        return np.einsum("nk,bnk->bn", J, B).max(0).mean()

    # --- 1. busqueda global ---
    mejor, mejor_pos = -1.0, P0
    rango = np.arange(-p["tx"], p["tx"] + 1, p["paso"])
    for s in p["escalas"]:
        base = c + s * (P0 - c)
        for ty in rango:
            for tx in rango:
                pos = base + np.array([tx, ty])
                if pos.min() < 0 or pos.max() > T - 1:
                    continue
                v = score_global(pos)
                if v > mejor:
                    mejor, mejor_pos = v, pos
    # refinamiento fino de la traslacion
    for ty in range(-2, 3):
        for tx in range(-2, 3):
            pos = mejor_pos + np.array([tx, ty]) * 0.5 * p["paso"]
            if pos.min() >= 0 and pos.max() <= T - 1:
                v = score_global(pos)
                if v > mejor:
                    mejor, mejor_pos = v, pos
    escala = np.linalg.norm(mejor_pos - mejor_pos.mean(0), axis=1).mean() / \
        np.linalg.norm(P0 - c, axis=1).mean()
    ref_aristas = fbg.vec_aristas * escala
    pos = mejor_pos.copy()

    # vecinos de cada nodo para el coste elastico
    ar = fbg.aristas
    vecinos = [[(e, 0 if ar[e, 0] == n else 1) for e in range(len(ar)) if n in ar[e]]
               for n in range(len(P0))]
    w = p["ventana"]
    d = np.arange(-w, w + 1)
    DX, DY = np.meshgrid(d, d)
    DX, DY = DX.ravel(), DY.ravel()

    # --- 2. distorsion local ---
    for _ in range(p["barridos"]):
        for n in range(len(P0)):
            cx, cy = np.round(pos[n]).astype(int)
            xs = np.clip(cx + DX, 0, T - 1)
            ys = np.clip(cy + DY, 0, T - 1)
            J = Mn[:, ys, xs].T                                   # (W,40)
            sim = (J @ B[:, n, :].T).max(1)                       # (W,)
            cand = np.stack([xs, ys], 1).astype(float)
            coste = np.zeros(len(cand))
            for e, lado in vecinos[n]:
                otro = pos[ar[e, 1 - lado]]
                v = (otro - cand) if lado == 0 else (cand - otro)
                ref = ref_aristas[e]
                coste += ((v - ref) ** 2).sum(1) / ((ref ** 2).sum() + 1e-9)
            coste /= max(len(vecinos[n]), 1)
            k = np.argmax(sim - p["lam"] * coste)
            pos[n] = cand[k]

    # --- 3. refinamiento subpixel con fase ---
    J = muestrear_jets(R, pos)
    for n in range(len(P0)):
        sims = sim_magnitud(np.abs(J[n])[None], np.abs(fbg.jets[:, n]))
        b = int(np.argmax(sims))
        dv = estimar_desplazamiento(fbg.jets[b, n], J[n])
        if np.linalg.norm(dv) <= 1.5:
            pos[n] = np.clip(pos[n] + dv, 0, T - 1.001)
    J = muestrear_jets(R, pos)
    s_fbg = sim_magnitud(np.abs(J)[None], np.abs(fbg.jets)).max(0).mean()
    return pos, float(s_fbg), J
