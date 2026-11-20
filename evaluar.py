"""
Evaluación del sistema de reconocimiento facial EBGM.

Calcula las métricas biométricas estándar y calibra el umbral de decisión:
  - Distribución de similitudes genuinas vs impostoras.
  - FAR / FRR, curva DET y EER (Equal Error Rate).
  - Umbral en el punto de EER.
  - Tasa de identificación rank-1 (1:N) para las fotos del cigala.

Protocolo (con los datos disponibles):
  - Galería: los 50 famosos comunes + 1 foto del cigala (la inscrita).
  - Genuinos: el resto de fotos del cigala contra la galería (misma persona).
  - Impostores: cada famoso contra los demás famosos (personas distintas) y las
    fotos del cigala contra los famosos.

Nota: sólo el cigala tiene varias fotos, así que las métricas de "mismo sujeto"
se estiman sobre él. Es una evaluación limitada por el dataset (1 foto/famoso),
y así se reporta.

Uso:
    python evaluar.py
    python evaluar.py --cigala-enrolada fotos_cigala/1.jpg --guardar-umbral
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from detector import DetectorCaras, ErrorDeteccion
from ebgm import ExtractorEBGM

AQUI = os.path.dirname(os.path.abspath(__file__))


def cargar_templates(det, ext, patron):
    out = {}
    for ruta in sorted(glob.glob(patron)):
        try:
            out[os.path.basename(ruta)] = ext.extraer(det.extraer_cara(ruta))
        except ErrorDeteccion:
            pass
    return out


def metricas(genuinos, impostores):
    g = np.asarray(genuinos); im = np.asarray(impostores)
    thr = np.linspace(min(g.min(), im.min()), max(g.max(), im.max()), 1001)
    far = np.array([(im >= t).mean() for t in thr])   # falsos aceptados
    frr = np.array([(g < t).mean() for t in thr])      # falsos rechazados
    j = int(np.argmin(np.abs(far - frr)))
    eer = (far[j] + frr[j]) / 2.0
    dprime = (g.mean() - im.mean()) / np.sqrt(0.5 * (g.var() + im.var()) + 1e-12)
    return dict(umbral_eer=float(thr[j]), eer=float(eer), dprime=float(dprime),
                gen_media=float(g.mean()), imp_media=float(im.mean()),
                thr=thr, far=far, frr=frr)


def histograma_ascii(g, im, ancho=50):
    lo = min(g.min(), im.min()); hi = max(g.max(), im.max())
    bins = np.linspace(lo, hi, 21)
    hg, _ = np.histogram(g, bins); hi_, _ = np.histogram(im, bins)
    hg = hg / hg.max(); hi_ = hi_ / hi_.max()
    print(f"  {'similitud':>10}  genuinos (#) / impostores (o)")
    for k in range(len(bins) - 1):
        c = (bins[k] + bins[k + 1]) / 2
        linea = "#" * int(hg[k] * ancho) + "o" * int(hi_[k] * ancho)
        print(f"  {c:>10.3f}  {linea}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.path.join(AQUI, "Dataset"))
    p.add_argument("--cigala-dir", default=os.path.join(AQUI, "fotos_cigala"))
    p.add_argument("--cigala-test", default=os.path.join(AQUI, "fotos_cigala_test"),
                   help="carpeta con fotos NUEVAS del cigala (genuinos no vistos)")
    p.add_argument("--guardar-umbral", action="store_true",
                   help="escribe el umbral EER en umbral.txt")
    a = p.parse_args()

    det = DetectorCaras(politica_multiples="mayor")
    ext = ExtractorEBGM(cargar_fondo=False)

    print("Extrayendo templates...")
    fam = cargar_templates(det, ext, os.path.join(a.dataset, "*"))
    cig = cargar_templates(det, ext, os.path.join(a.cigala_dir, "*.jpg"))
    # Fondo con los famosos comunes (como en producción)
    ext.ajustar_fondo(list(fam.values()))

    # El cigala se inscribe como CENTROIDE de sus fotos de referencia (igual que
    # inscribir_dataset): centroide en espacio normalizado + media de fondo.
    centroide = np.mean([ext._normalizar(t) for t in cig.values()], axis=0)
    galeria = dict(fam); galeria["cigala"] = (ext.media + centroide).astype(np.float32)

    # Genuinos: fotos NUEVAS del cigala (no usadas al inscribir). Si no hay, se
    # cae a leave-one-out sobre las de referencia (menos honesto, se avisa).
    cig_test = cargar_templates(det, ext, os.path.join(a.cigala_test, "*"))
    if cig_test:
        print(f"  famosos: {len(fam)} | referencias cigala: {len(cig)} | "
              f"prueba cigala (nuevas): {len(cig_test)}")
        genuinos = [ext.similitud(galeria["cigala"], t) for t in cig_test.values()]
        probes_cig = cig_test
    else:
        print(f"  famosos: {len(fam)} | cigala: {len(cig)} (sin set de prueba: leave-one-out)")
        genuinos = [ext.similitud(galeria["cigala"], t) for t in cig.values()]
        probes_cig = cig

    # --- Impostores: famoso vs otros famosos + famosos vs cigala ---
    impostores = []
    claves_fam = list(fam)
    for i, ki in enumerate(claves_fam):
        for kj in claves_fam[i + 1:]:
            impostores.append(ext.similitud(fam[ki], fam[kj]))
    for kf in fam:
        impostores.append(ext.similitud(galeria["cigala"], fam[kf]))

    m = metricas(genuinos, impostores)

    print("\n=== Distribución de similitudes ===")
    histograma_ascii(np.array(genuinos), np.array(impostores))

    print("\n=== Métricas de verificación (1:1) ===")
    print(f"  genuinos:   media={m['gen_media']:+.3f}  (n={len(genuinos)})")
    print(f"  impostores: media={m['imp_media']:+.3f}  (n={len(impostores)})")
    print(f"  d-prime (separación): {m['dprime']:.2f}")
    print(f"  EER: {m['eer']*100:.1f}%  en umbral = {m['umbral_eer']:+.4f}")

    # --- Identificación 1:N (rank-1) de las fotos del cigala ---
    aciertos = 0; total = 0
    for k, t in probes_cig.items():
        total += 1
        ranking = sorted(((id_, ext.similitud(g, t)) for id_, g in galeria.items()),
                         key=lambda kv: kv[1], reverse=True)
        if ranking[0][0] == "cigala":
            aciertos += 1
    print("\n=== Métricas de identificación (1:N, galería de 50) ===")
    print(f"  rank-1 (cigala): {aciertos}/{total} = {aciertos/total*100:.0f}%")
    print(f"  (azar ≈ {100/len(galeria):.0f}%)")

    if a.guardar_umbral:
        with open(os.path.join(AQUI, "umbral.txt"), "w") as f:
            f.write(f"{m['umbral_eer']:.4f}\n")
        print(f"\nUmbral EER guardado en umbral.txt: {m['umbral_eer']:.4f}")

    print("\nNota: EBGM es un método clásico basado en jets de Gabor. En fotos no")
    print("controladas (poses, gafas, décadas de diferencia) su capacidad es")
    print("limitada; funciona mejor en caras frontales de condiciones similares.")


if __name__ == "__main__":
    main()
