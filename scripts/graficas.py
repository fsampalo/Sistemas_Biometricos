"""
Gráficas del sistema EBGM desplegado -> salida/graficas/*.png

  01_distribuciones.png  similitudes genuinas vs impostoras y umbral
  02_det.png             curva DET (FAR vs FRR) del protocolo «1 foto vs 1 foto»
  03_fotos_test.png      similitud de cada foto de test del Cigala con su inscripción
  04_cmc.png             identificación 1:N: probabilidad de acierto en el top-k
  05_robustez.png        acierto frente a degradaciones de captura
  06_parejas.png         parejas de famosos más confundibles
  07_ajuste.png          evolución de la configuración en la fase 1
  08_grafo.png           el grafo EBGM ajustado sobre caras reales
  09_gabor.png           el banco de 7x8 wavelets de Gabor

Usa la BD y el umbral reales (como evaluar.py).

Uso:
    python scripts/graficas.py
    python scripts/graficas.py --sin-robustez     # omite la 05 (la más lenta)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raíz del repo

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ajuste_ebgm import aumentos, eer  # noqa: E402
from evaluar import es_de_ajuste  # noqa: E402
from inscribir_dataset import id_desde_fichero  # noqa: E402
from reconocimiento.detector import DetectorCaras, ErrorDeteccion  # noqa: E402
from reconocimiento.rutas import DIR_CIGALA, DIR_CIGALA_TEST, DIR_DATASET, DIR_SALIDA  # noqa: E402
from reconocimiento.sistema import SistemaReconocimiento  # noqa: E402

# ---------------------------------------------------------------- estilo
# Paleta validada (categórica en orden fijo: azul, naranja) y tintas neutras.
AZUL, NARANJA = "#2a78d6", "#eb6834"
SUPERFICIE, TINTA, TINTA2, APAGADO = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
REJILLA, EJE = "#e1e0d9", "#c3c2b7"
DIR_GRAF = os.path.join(DIR_SALIDA, "graficas")

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": EJE, "axes.labelcolor": TINTA2, "axes.titlecolor": TINTA,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": APAGADO, "ytick.color": APAGADO,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": REJILLA,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
    "legend.labelcolor": TINTA2, "font.size": 10, "lines.linewidth": 2,
})


def guardar(fig, nombre):
    os.makedirs(DIR_GRAF, exist_ok=True)
    ruta = os.path.join(DIR_GRAF, nombre)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"  {os.path.relpath(ruta)}")


def linea_umbral(ax, umbral, vertical=True, texto=True):
    (ax.axvline if vertical else ax.axhline)(umbral, color=TINTA, lw=1.2, ls="--", zorder=3)
    if texto:
        if vertical:
            ax.annotate(f"umbral {umbral:.3f}", (umbral, 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -12), textcoords="offset points", color=TINTA, fontsize=9,
                        bbox=dict(fc=SUPERFICIE, ec="none", pad=1.5), zorder=5)
        else:
            ax.annotate(f"umbral {umbral:.3f}", (1, umbral), xycoords=("axes fraction", "data"),
                        xytext=(-4, 4), textcoords="offset points", ha="right", color=TINTA, fontsize=9)


# ---------------------------------------------------------------- datos
def preparar():
    sis = SistemaReconocimiento(auditoria=False, exponer_puntuaciones=True)
    ext, det = sis.extractor, DetectorCaras(politica_multiples="mayor")

    def carpeta(d):
        out = {}
        for r in sorted(glob.glob(os.path.join(d, "*"))):
            try:
                out[os.path.basename(r)] = ext.extraer(det.extraer_cara(r))
            except ErrorDeteccion:
                pass
        return out

    fam = {id_desde_fichero(k): v for k, v in carpeta(DIR_DATASET).items()}
    test = carpeta(DIR_CIGALA_TEST)
    rep = {k: ext.representar(v) for k, v in fam.items()}
    rep_t = {k: ext.representar(v) for k, v in test.items()}
    S = ext.similitud_rep
    kf, kt = list(rep), list(rep_t)
    cig = sis._rep["cigala"]
    imp = np.array([S(rep[kf[i]], rep[kf[j]]) for i in range(len(kf)) for j in range(i + 1, len(kf))]
                   + [S(rep[k], cig) for k in kf]
                   + [S(r, rep[k]) for r in rep_t.values() for k in kf])
    gen_pares = np.array([S(rep_t[x], rep_t[y]) for x in kt for y in kt if x != y])
    sim_bd = {k: S(r, cig) for k, r in rep_t.items()}
    pares_fam = sorted(((S(rep[kf[i]], rep[kf[j]]), kf[i], kf[j])
                        for i in range(len(kf)) for j in range(i + 1, len(kf))), reverse=True)
    return dict(sis=sis, ext=ext, det=det, rep_t=rep_t, imp=imp, gen_pares=gen_pares,
                sim_bd=sim_bd, pares_fam=pares_fam)


# ---------------------------------------------------------------- gráficas
def g_distribuciones(d):
    imp, gen, u = d["imp"], d["gen_pares"], d["sis"].umbral
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(min(imp.min(), gen.min()), max(imp.max(), gen.max()), 45)
    ax.hist(imp, bins, density=True, color=NARANJA, alpha=.85, rwidth=.88,
            label=f"impostores (n={len(imp)})")
    ax.hist(gen, bins, density=True, color=AZUL, alpha=.75, rwidth=.88,
            label=f"genuinos, 1 foto vs 1 foto (n={len(gen)})")
    linea_umbral(ax, u)
    ax.set_xlabel("similitud EBGM")
    ax.set_ylabel("densidad")
    ax.set_title("Genuinos e impostores: cuánto se solapan")
    ax.legend(loc="center right")
    ax.text(0.01, 0.97, f"FAR en el umbral: {np.mean(imp >= u):.1%}\n"
                        f"FRR en el umbral: {np.mean(gen < u):.0%}",
            transform=ax.transAxes, va="top", color=TINTA2, fontsize=9)
    guardar(fig, "01_distribuciones.png")


def g_det(d):
    imp, gen, u = d["imp"], d["gen_pares"], d["sis"].umbral
    thr = np.unique(np.concatenate([imp, gen]))
    far = np.array([(imp >= t).mean() for t in thr])
    frr = np.array([(gen < t).mean() for t in thr])
    e, _ = eer(gen, imp)
    fig, ax = plt.subplots(figsize=(6, 5))
    m = (far > 0) & (frr > 0)
    ax.loglog(far[m] * 100, frr[m] * 100, color=AZUL, label="1 foto vs 1 foto")
    fu, fr = np.mean(imp >= u) * 100, np.mean(gen < u) * 100
    ax.plot([fu], [fr], "o", ms=9, color=AZUL, mec=SUPERFICIE, mew=2, zorder=4)
    ax.annotate(f"punto desplegado\nFAR {fu:.1f} %, FRR {fr:.0f} %", (fu, fr),
                xytext=(10, 8), textcoords="offset points", color=TINTA, fontsize=9)
    ax.plot([e * 100], [e * 100], "s", ms=8, color=TINTA2, mec=SUPERFICIE, mew=2, zorder=4)
    ax.annotate(f"EER {e:.1%}", (e * 100, e * 100), xytext=(-62, -4),
                textcoords="offset points", color=TINTA2, fontsize=9)
    ax.set_xlabel("FAR (%)  — impostores aceptados")
    ax.set_ylabel("FRR (%)  — genuinos rechazados")
    ax.set_title("Curva DET (más abajo e izquierda = mejor)")
    ax.grid(True, which="both", alpha=.6)
    guardar(fig, "02_det.png")


def g_fotos_test(d):
    sim, u = d["sim_bd"], d["sis"].umbral
    ks = sorted(sim, key=sim.get)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(ks) + 1.4))
    cols = [NARANJA if es_de_ajuste(k) else AZUL for k in ks]
    ax.barh(range(len(ks)), [sim[k] for k in ks], color=cols, height=.72)
    ax.set_yticks(range(len(ks)), [os.path.splitext(k)[0] for k in ks], color=TINTA2)
    for i, k in enumerate(ks):
        ax.text(sim[k] + 0.005, i, f"{sim[k]:.2f}", va="center", fontsize=8, color=TINTA2,
                bbox=dict(fc=SUPERFICIE, ec="none", pad=0.5))
    linea_umbral(ax, u)
    ax.set_xlabel("similitud con el Cigala inscrito (8.jpg)")
    acept = sum(v >= u for v in sim.values())
    ax.set_title(f"Fotos de test del Cigala: {acept} de {len(ks)} superan el umbral")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AZUL, label="nuevas (nunca usadas al ajustar)"),
                       Patch(color=NARANJA, label="ref_* (usadas en el ajuste)")],
              loc="lower right")
    ax.grid(axis="y", visible=False)
    guardar(fig, "03_fotos_test.png")


def g_cmc(d):
    sis = d["sis"]
    ids = sis.personas()
    rangos = {}
    for k, r in d["rep_t"].items():
        s = sis._similitudes_rep(r, ids)
        rangos[k] = [ids[i] for i in np.argsort(-s)].index("cigala") + 1
    K = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for nombre, col, sel in (("nuevas", AZUL, lambda k: not es_de_ajuste(k)),
                             ("ref_* (usadas en el ajuste)", NARANJA, es_de_ajuste)):
        v = np.array([rangos[k] for k in rangos if sel(k)])
        if len(v) == 0:
            continue
        cmc = [(v <= k).mean() * 100 for k in K]
        ax.plot(K, cmc, "-o", color=col, ms=8, mec=SUPERFICIE, mew=2, label=f"{nombre} (n={len(v)})")
        ax.annotate(f"{cmc[0]:.0f} %", (1, cmc[0]), xytext=(-30, 0), textcoords="offset points",
                    va="center", color=TINTA2, fontsize=9)
    ax.plot(K, K / len(ids) * 100, color=APAGADO, lw=1.2, ls=":", label="azar")
    ax.set_xticks(K)
    ax.set_xlim(0.3, 10.3)
    ax.set_ylim(0, 105)
    ax.set_xlabel("rango k (galería de 50 personas)")
    ax.set_ylabel("% de fotos con el Cigala en el top-k")
    ax.set_title("Identificación 1:N del Cigala (CMC)")
    ax.legend(loc="center right")
    guardar(fig, "04_cmc.png")


def g_robustez(d):
    sis, det = d["sis"], d["det"]
    por_tipo: dict[str, list] = {}
    for r in sorted(glob.glob(os.path.join(DIR_DATASET, "*"))):
        id_ = id_desde_fichero(r)
        for tipo, img in aumentos(det.cargar_imagen(r)).items():
            try:
                por_tipo.setdefault(tipo, []).append(sis.verificar(id_, img)["verificado"])
            except Exception:
                por_tipo.setdefault(tipo, []).append(False)
    tipos = sorted(por_tipo, key=lambda t: np.mean(por_tipo[t]))
    vals = [np.mean(por_tipo[t]) * 100 for t in tipos]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(range(len(tipos)), vals, color=AZUL, height=.68)
    ax.set_yticks(range(len(tipos)), [t.replace("_", " ") for t in tipos], color=TINTA2)
    for i, v in enumerate(vals):
        ax.text(v - 1, i, f"{v:.0f} %", va="center", ha="right", color="#ffffff", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de los 49 famosos verificados con su foto degradada")
    ax.set_title("Robustez a cambios de captura (misma foto)")
    ax.grid(axis="y", visible=False)
    guardar(fig, "05_robustez.png")


def g_parejas(d):
    u = d["sis"].umbral
    top = d["pares_fam"][:10][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    vals = [s for s, _, _ in top]
    ax.barh(range(len(top)), vals, color=[NARANJA if v >= u else AZUL for v in vals], height=.7)
    ax.set_yticks(range(len(top)), [f"{a}  ~  {b}" for _, a, b in top], color=TINTA2, fontsize=8)
    linea_umbral(ax, u)
    ax.set_xlabel("similitud entre dos famosos distintos")
    ax.set_title("Las 10 parejas de famosos más confundibles")
    from matplotlib.patches import Patch
    h = [Patch(color=NARANJA, label="supera el umbral (falsa aceptación)")]
    if any(v < u for v in vals):
        h.append(Patch(color=AZUL, label="bajo el umbral"))
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.grid(axis="y", visible=False)
    guardar(fig, "06_parejas.png")


def g_ajuste():
    # cifras de scripts/ajuste_ebgm.py (protocolo 1 foto, sin usar el test)
    etapas = ["EBGM\ninicial", "+7 escalas, CLAHE,\nfiducial", "+invariancia\nespejo",
              "+agregación\nrobusta (final)"]
    eer_ = [34.6, 16.7, 14.1, 12.8]
    tar = [2.6, 32.1, 34.6, 41.7]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, vals, tit in ((axs[0], eer_, "EER (%) — menor es mejor"),
                          (axs[1], tar, "TAR a FAR 1 % (%) — mayor es mejor")):
        ax.bar(range(4), vals, color=[APAGADO] * 3 + [AZUL], width=.62)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:.1f}", ha="center", color=TINTA, fontsize=9)
        ax.set_xticks(range(4), etapas, fontsize=8, color=TINTA2)
        ax.set_title(tit, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.15)
    fig.suptitle("Fase 1: cómo mejoró la configuración de EBGM", x=0.01, ha="left",
                 fontweight="bold", color=TINTA)
    guardar(fig, "07_ajuste.png")


def g_grafo(d):
    import cv2
    ext, det = d["ext"], d["det"]
    fotos = [("Cigala (inscripción)", glob.glob(os.path.join(DIR_CIGALA, "*"))[0])]
    fotos += [(id_desde_fichero(r), r) for r in sorted(glob.glob(os.path.join(DIR_DATASET, "*")))[:3]]
    fig, axs = plt.subplots(1, len(fotos), figsize=(3 * len(fotos), 3.4))
    for ax, (nombre, ruta) in zip(axs, fotos):
        cara = det.extraer_cara(ruta)
        g = ext._gris(cara)
        mapas = ext._mapas(ext._preprocesar(g))
        pos = ext._localizar_nodos(mapas)
        ax.imshow(cv2.cvtColor(cara, cv2.COLOR_BGR2RGB))
        ax.scatter(pos[:, 0], pos[:, 1], s=26, c=AZUL, edgecolors="#ffffff", linewidths=1)
        ax.set_title(nombre.replace("_", " "), fontsize=9, fontweight="normal", loc="center")
        ax.axis("off")
    fig.suptitle("Grafo EBGM: los 31 nodos tras el ajuste elástico con el bunch graph",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=TINTA)
    guardar(fig, "08_grafo.png")


def g_gabor(d):
    ext = d["ext"]
    E, O = ext.cfg.n_escalas, ext.cfg.n_orientaciones
    fig, axs = plt.subplots(E, O, figsize=(O * 0.9, E * 0.95))
    for v in range(E):
        for mu in range(O):
            re, _ = ext._kernels[v * O + mu]
            ax = axs[v, mu]
            ax.imshow(re, cmap="gray")
            ax.axis("off")
        axs[v, 0].text(-0.15, 0.5, f"λ={ext.cfg.lambda_min * np.sqrt(2) ** v:.0f}",
                       transform=axs[v, 0].transAxes, ha="right", va="center",
                       fontsize=7, color=TINTA2)
    fig.suptitle("Banco de Gabor: 7 escalas × 8 orientaciones (parte real)", x=0.01, ha="left",
                 fontsize=10, fontweight="bold", color=TINTA)
    guardar(fig, "09_gabor.png")


def main():
    p = argparse.ArgumentParser(description="Gráficas del sistema EBGM")
    p.add_argument("--sin-robustez", action="store_true")
    a = p.parse_args()
    print("Calculando similitudes...")
    d = preparar()
    print(f"Guardando gráficas en {os.path.relpath(DIR_GRAF)}/")
    g_distribuciones(d)
    g_det(d)
    g_fotos_test(d)
    g_cmc(d)
    if not a.sin_robustez:
        g_robustez(d)
    g_parejas(d)
    g_ajuste()
    g_grafo(d)
    g_gabor(d)


if __name__ == "__main__":
    main()
