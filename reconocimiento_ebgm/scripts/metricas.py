"""Calcula metricas y figuras a partir de resultados/evaluacion_bruta.json."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ebgm.sistema import UMBRAL_POR_DEFECTO as U  # noqa: E402

AZUL, NARANJA, TINTA, GRIS = "#2a78d6", "#eb6834", "#1f1f1e", "#8a8984"
DURAS = {"espejo", "gafas_sol", "mascarilla"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRIS, "axes.labelcolor": TINTA,
                     "xtick.color": GRIS, "ytick.color": GRIS, "axes.spines.top": False,
                     "axes.spines.right": False})


def eer(gen, imp):
    us = np.linspace(0.0, 1.0, 2001)
    far = np.array([(imp >= u).mean() for u in us])
    frr = np.array([(gen < u).mean() for u in us])
    i = int(np.argmin(np.abs(far - frr)))
    return (far[i] + frr[i]) / 2, us[i], us, far, frr


def main(dir_res="resultados"):
    R = json.load(open(os.path.join(dir_res, "evaluacion_bruta.json")))
    out = {"umbral": U, "por_aumento": {}, "global": {}}
    for modo in ["detector", "sin_detector"]:
        rr = [r for r in R if r["modo"] == modo]
        nombres = list(dict.fromkeys(r["aum"] for r in rr))
        for a in nombres:
            sub = [r for r in rr if r["aum"] == a]
            okk = [r for r in sub if "fallo" not in r]
            g = np.array([r["genuino"] for r in okk])
            imp = np.array([v for r in okk for v in r["impostores"].values()])
            fallos = len(sub) - len(okk)
            out["por_aumento"][f"{modo}/{a}"] = dict(
                n=len(sub), fallos_deteccion=fallos, gen_mediana=float(np.median(g)),
                gen_min=float(g.min()), imp_max=float(imp.max()),
                frr=float(((g < U).sum() + fallos) / len(sub)),
                rank1=float(np.mean([r["rank1"] == r["id"] for r in okk] + [False] * fallos)),
                ident_correcta_umbral=float(np.mean(
                    [(r["rank1"] == r["id"] and r["genuino"] >= U) for r in okk] + [False] * fallos)))
        for nombre, filtro in [("todas", lambda r: True), ("sin_oclusion_ni_espejo", lambda r: r["aum"] not in DURAS)]:
            okk = [r for r in rr if "fallo" not in r and filtro(r)]
            g = np.array([r["genuino"] for r in okk])
            imp = np.array([v for r in okk for v in r["impostores"].values()])
            e, ue, us, far, frr = eer(g, imp)
            out["global"][f"{modo}/{nombre}"] = dict(
                n_gen=len(g), n_imp=len(imp), eer=float(e), umbral_eer=float(ue),
                far_umbral=float((imp >= U).mean()), frr_umbral=float((g < U).mean()),
                imp_max=float(imp.max()))
            if modo == "detector" and nombre == "todas":
                gd, impd, fard, frrd = g, imp, far, frr
    json.dump(out, open(os.path.join(dir_res, "metricas.json"), "w"), indent=1)

    # Figura 1: distribuciones de puntuaciones (ruta con detector)
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    bins = np.linspace(-0.2, 1.0, 97)
    ax.hist(impd, bins=bins, color=NARANJA, alpha=0.85, density=True, label="Impostores",
            edgecolor="white", linewidth=0.5)
    ax.hist(gd, bins=bins, color=AZUL, alpha=0.85, density=True, label="Genuinos",
            edgecolor="white", linewidth=0.5)
    ax.axvline(U, color=TINTA, lw=1, ls="--")
    ax.text(U + 0.01, ax.get_ylim()[1] * 0.9, f"umbral = {U}", color=TINTA, fontsize=8)
    ax.set_xlabel("Similitud S_a media (template sonda vs. template inscrito)")
    ax.set_ylabel("Densidad")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_res, "distribuciones.pdf"))
    fig.savefig(os.path.join(dir_res, "distribuciones.png"), dpi=160)

    # Figura 2: FAR y FRR frente al umbral
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    us = np.linspace(0.0, 1.0, 2001)
    ax.plot(us, fard * 100, color=NARANJA, lw=2, label="FAR")
    ax.plot(us, frrd * 100, color=AZUL, lw=2, label="FRR")
    ax.axvline(U, color=TINTA, lw=1, ls="--")
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("Umbral")
    ax.set_ylabel("Tasa de error (%)")
    ax.legend(frameon=False)
    ax.grid(color="#e6e5e0", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_res, "far_frr.pdf"))
    fig.savefig(os.path.join(dir_res, "far_frr.png"), dpi=160)
    print(json.dumps(out["global"], indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
