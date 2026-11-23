"""
Banco de experimentos para AJUSTAR la configuración de EBGM (fase 1).

Compara variantes de EBGM con un protocolo que NO toca el conjunto de test
(`fotos_cigala_test/`), que se reserva para la evaluación final (evaluar.py).

Datos de ajuste
---------------
  * Famosos del docente (Dataset/, 1 foto por persona): impostores reales y,
    con AUMENTOS sintéticos de cada foto (iluminación, desenfoque, JPEG, giro,
    espejo, ruido, luz lateral), genuinos "de captura".
  * Fotos de referencia del cigala (fotos_cigala/, 13): todos los PARES entre
    ellas son genuinos reales con 1 sola foto inscrita (pose, gafas, 20 años
    de diferencia...), que es el caso real de los 49 famosos.

Métricas que se imprimen por configuración
------------------------------------------
  EER_cig / d'_cig : genuinos = pares de fotos del cigala; impostores = pares
                     de famosos + cigala vs famosos.
  EER_aug          : genuinos = aumento vs original del mismo famoso.
  R1_cig           : rank-1 1:N usando 1 foto del cigala como galería y otra
                     como sonda (galería de 50: famosos + cigala), promediado.
  R1_aug           : rank-1 1:N de las fotos aumentadas contra la galería.

Uso:
    python ajuste_ebgm.py              # tabla de variantes
    python ajuste_ebgm.py --rapido     # solo unas pocas variantes
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import replace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raíz del repo

from reconocimiento.detector import DetectorCaras, ErrorDeteccion
from reconocimiento.ebgm import ConfigEBGM, ExtractorEBGM
from reconocimiento.rutas import DIR_CIGALA, DIR_DATASET, DIR_SALIDA

CACHE = os.path.join(DIR_SALIDA, "cache_caras_ajuste.npz")


# ----------------------------------------------------------------- aumentos
def aumentos(img: np.ndarray) -> dict[str, np.ndarray]:
    """Variaciones de captura de una misma foto (deterministas)."""
    h, w = img.shape[:2]
    f = img.astype(np.float32) / 255.0
    out = {}
    out["oscura"] = np.clip(255 * f ** 1.8, 0, 255).astype(np.uint8)
    out["clara"] = np.clip(255 * f ** 0.55, 0, 255).astype(np.uint8)
    peq = cv2.resize(img, (max(32, w // 3), max(32, h // 3)), interpolation=cv2.INTER_AREA)
    out["baja_res"] = cv2.GaussianBlur(cv2.resize(peq, (w, h)), (0, 0), 1.0)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 15])
    out["jpeg15"] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1.0)
    out["giro12"] = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    out["espejo"] = img[:, ::-1].copy()
    rng = np.random.default_rng(0)
    ruido = f * 0.75 + 0.1 + rng.normal(0, 0.04, f.shape)
    out["ruido"] = np.clip(255 * ruido, 0, 255).astype(np.uint8)
    rampa = np.linspace(0.35, 1.25, w, dtype=np.float32)[None, :, None]
    out["luz_lateral"] = np.clip(img * rampa, 0, 255).astype(np.uint8)
    return out


def construir_cache(forzar: bool = False) -> dict:
    if os.path.isfile(CACHE) and not forzar:
        z = np.load(CACHE, allow_pickle=False)
        return {k: z[k] for k in z.files}
    det = DetectorCaras(politica_multiples="mayor")
    fam, fam_id, aug, aug_id, aug_tipo = [], [], [], [], []
    for ruta in sorted(glob.glob(os.path.join(DIR_DATASET, "*"))):
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        try:
            img = det.cargar_imagen(ruta)
            fam.append(det.extraer_cara(img)); fam_id.append(nombre)
        except ErrorDeteccion as e:
            print("  [famoso sin cara]", nombre, e)
            continue
        for tipo, im2 in aumentos(img).items():
            try:
                aug.append(det.extraer_cara(im2)); aug_id.append(nombre); aug_tipo.append(tipo)
            except ErrorDeteccion:
                pass

    def carpeta(c):
        caras, nombres = [], []
        for ruta in sorted(glob.glob(os.path.join(c, "*"))):
            try:
                caras.append(det.extraer_cara(ruta)); nombres.append(os.path.basename(ruta))
            except ErrorDeteccion:
                pass
        return caras, nombres

    cig, cig_n = carpeta(DIR_CIGALA)
    d = dict(fam=np.stack(fam), fam_id=np.array(fam_id), aug=np.stack(aug),
             aug_id=np.array(aug_id), aug_tipo=np.array(aug_tipo),
             cig=np.stack(cig), cig_n=np.array(cig_n))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **d)
    return d


# ----------------------------------------------------------------- métricas
def eer(gen, imp) -> tuple[float, float]:
    gen = np.asarray(gen); imp = np.asarray(imp)
    thr = np.unique(np.concatenate([gen, imp]))
    far = np.array([(imp >= t).mean() for t in thr])
    frr = np.array([(gen < t).mean() for t in thr])
    j = int(np.argmin(np.abs(far - frr)))
    return float((far[j] + frr[j]) / 2), float(thr[j])


def dprime(gen, imp) -> float:
    gen = np.asarray(gen); imp = np.asarray(imp)
    return float((gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))


def evaluar_config(cfg: ConfigEBGM, datos: dict, verbose: bool = False) -> dict:
    """Galería = template de INSCRIPCIÓN (extraer_inscripcion); sonda = extraer."""
    ext = ExtractorEBGM(cfg, modelo=None)
    t0 = time.time()
    ext.fijar_modelo(ext.entrenar_modelo(list(datos["fam"])))
    P_fam = [ext.representar(ext.extraer(c)) for c in datos["fam"]]
    P_cig = [ext.representar(ext.extraer(c)) for c in datos["cig"]]
    P_aug = [ext.representar(ext.extraer(c)) for c in datos["aug"]]
    t_ext = (time.time() - t0) / (len(P_fam) * 2 + len(P_cig) + len(P_aug))
    if cfg.variaciones_inscripcion > 0:
        G_fam = [ext.representar(ext.extraer_inscripcion(c)) for c in datos["fam"]]
        G_cig = [ext.representar(ext.extraer_inscripcion(c)) for c in datos["cig"]]
    else:
        G_fam, G_cig = P_fam, P_cig

    S = ext.similitud_rep
    nf, nc = len(P_fam), len(P_cig)
    M_ff = np.array([[S(P_fam[i], G_fam[j]) for j in range(nf)] for i in range(nf)])
    M_cf = np.array([[S(P_cig[i], G_fam[j]) for j in range(nf)] for i in range(nc)])
    M_cc = np.array([[S(P_cig[i], G_cig[j]) for j in range(nc)] for i in range(nc)])
    off_f = ~np.eye(nf, dtype=bool); off_c = ~np.eye(nc, dtype=bool)
    imp_ff = list(M_ff[off_f]); imp_cf = list(M_cf.ravel())
    gen_cc = list(M_cc[off_c])
    idx = {n: k for k, n in enumerate(datos["fam_id"])}
    M_af = np.array([[S(P_aug[k], g) for g in G_fam] for k in range(len(P_aug))])
    gen_aug = [M_af[k, idx[n]] for k, n in enumerate(datos["aug_id"])]

    imp = imp_ff + imp_cf
    e_c, thr_c = eer(gen_cc, imp)
    e_a, _ = eer(gen_aug, imp_ff)
    # rank-1 cigala: galería = famosos + cigala[j]; sonda = cigala[i]
    r1c = np.mean([M_cc[i, j] > M_cf[i].max() for i in range(nc) for j in range(nc) if i != j])
    r1a = np.mean([int(np.argmax(M_af[k])) == idx[n] for k, n in enumerate(datos["aug_id"])])
    # TAR del cigala (1 foto) con el umbral a FAR=1 %
    thr1 = np.quantile(imp, 0.99)
    return dict(EER_cig=e_c, dp_cig=dprime(gen_cc, imp), EER_aug=e_a,
                dp_aug=dprime(gen_aug, imp_ff), R1_cig=float(r1c), R1_aug=float(r1a),
                thr=thr_c, ms=t_ext * 1000, TAR1=float(np.mean(np.array(gen_cc) >= thr1)),
                gen_c=float(np.mean(gen_cc)), imp=float(np.mean(imp)))


# Las rondas 1-4 se definieron como cambios sobre la configuración INICIAL de
# la reescritura (no sobre la final). Se fija aquí para que sean reproducibles.
BASE_RONDAS_1_4 = dict(preproceso="tantriggs", n_escalas=5, lambda_min=4.0, sigma_rel=1.0,
                       grafo="mixto", simetrizar=True, invariante_espejo=False)

VARIANTES = {
    # nombre: cambios sobre BASE_RONDAS_1_4 (rondas 1-4) o sobre el defecto (R5)
    "antiguo(rejilla9x8,sin preproc,fondo,sin elast.)": dict(
        preproceso="ninguno", grafo="rejilla", rejilla=(9, 8), margen_rejilla=16,
        sigma_rel=0.56, radio_elastico=0, simetrizar=False, compresion="ninguna",
        normalizacion="fondo"),
    "DEFECTO": dict(),
    "sin tantriggs": dict(preproceso="ninguno"),
    "clahe": dict(preproceso="clahe"),
    "sigma 0.56λ": dict(sigma_rel=0.56),
    "sigma 0.75λ": dict(sigma_rel=0.75),
    "solo rejilla": dict(grafo="rejilla"),
    "solo fiducial": dict(grafo="fiducial"),
    "sin elástico": dict(radio_elastico=0),
    "radio 5": dict(radio_elastico=5),
    "radio 2": dict(radio_elastico=2),
    "penaliz 0.2": dict(penaliz_elastica=0.2),
    "sin simetrizar": dict(simetrizar=False),
    "compresión log": dict(compresion="log"),
    "sin compresión": dict(compresion="ninguna"),
    "norm fondo": dict(normalizacion="fondo"),
    "norm ninguna": dict(normalizacion="ninguna"),
    "geometría 0.05": dict(peso_geometria=0.05),
    "6 escalas": dict(n_escalas=6),
    "lambda_min 3": dict(lambda_min=3.0),
    # ---- 2ª ronda: combinaciones de lo que mejora en la 1ª
    "R2 clahe+6esc": dict(preproceso="clahe", n_escalas=6),
    "R2 clahe+6esc+fiducial": dict(preproceso="clahe", n_escalas=6, grafo="fiducial"),
    "R2 clahe+6esc+pen0.005": dict(preproceso="clahe", n_escalas=6, penaliz_elastica=0.005),
    "R2 clahe+6esc+pen0.005+r5": dict(preproceso="clahe", n_escalas=6,
                                     penaliz_elastica=0.005, radio_elastico=5),
    "R2 clahe+6esc+sigma.75": dict(preproceso="clahe", n_escalas=6, sigma_rel=0.75),
    "R2 clahe+6esc+log": dict(preproceso="clahe", n_escalas=6, compresion="log"),
    "R2 clahe+6esc+sinsim": dict(preproceso="clahe", n_escalas=6, simetrizar=False),
    "R2 clahe+7esc": dict(preproceso="clahe", n_escalas=7),
    # ---- 3ª ronda: más escalas bajas (forma global), sin simetrizar (x2 rápido)
    "R3 7esc": dict(preproceso="clahe", n_escalas=7, simetrizar=False),
    "R3 8esc": dict(preproceso="clahe", n_escalas=8, simetrizar=False),
    "R3 7esc lmin5": dict(preproceso="clahe", n_escalas=7, lambda_min=5.0, simetrizar=False),
    "R3 7esc sigma.75": dict(preproceso="clahe", n_escalas=7, sigma_rel=0.75, simetrizar=False),
    "R3 7esc fiducial": dict(preproceso="clahe", n_escalas=7, grafo="fiducial", simetrizar=False),
    "R3 7esc tantriggs": dict(n_escalas=7, simetrizar=False),
    "R3 7esc 12orient": dict(preproceso="clahe", n_escalas=7, n_orientaciones=12, simetrizar=False),
    "R3 7esc rejilla9x8": dict(preproceso="clahe", n_escalas=7, rejilla=(9, 8), simetrizar=False),
    # ---- 4ª ronda: combinación final + inscripción con micro-variaciones
    "R4 base(7esc)": dict(preproceso="clahe", n_escalas=7, simetrizar=False),
    "R4 fid+s.75": dict(preproceso="clahe", n_escalas=7, grafo="fiducial",
                        sigma_rel=0.75, simetrizar=False),
    "R4 fid+s.75+lmin5": dict(preproceso="clahe", n_escalas=7, grafo="fiducial",
                              sigma_rel=0.75, lambda_min=5.0, simetrizar=False),
    "R4 fid+s.75+var4": dict(preproceso="clahe", n_escalas=7, grafo="fiducial",
                             sigma_rel=0.75, simetrizar=False, variaciones_inscripcion=4),
    "R4 fid+s.75+var8": dict(preproceso="clahe", n_escalas=7, grafo="fiducial",
                             sigma_rel=0.75, simetrizar=False, variaciones_inscripcion=8),
    # ---- 5ª ronda: invariancia a espejo en el matcher (por defecto desde aquí)
    "R5 FINAL (defecto)": dict(agregacion="media"),
    "R5 FINAL sin invariancia espejo": dict(agregacion="media", invariante_espejo=False),
    # ---- 6ª ronda: agregación robusta de los nodos. Probado y DESCARTADO en
    # esta ronda (experimentos aparte, no incluidos aquí): T-norm con la
    # cohorte (peor), pesos de Fisher por nodo aprendidos de los aumentos (sin
    # efecto) o del cigala (mejoran, pero bajan los ojos por sus gafas de sol
    # y suben boca/mentón por su barba: son pesos de UNA persona), WCCN por
    # nodo (+1-3 puntos, ruidoso) y aumentos de pose para WCCN (peor).
    "R6 mitad + quita 6": dict(agregacion="mitad_recortada", nodos_descartados=6),
    "R6 mejor mitad (sin recortar)": dict(agregacion="mitad_recortada", nodos_descartados=0),
    "R6 mitad + quita 2": dict(agregacion="mitad_recortada", nodos_descartados=2),
    "R6 FINAL mitad + quita 3 (defecto)": dict(),
    "R6 mitad + quita 4": dict(agregacion="mitad_recortada", nodos_descartados=4),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--rehacer-cache", action="store_true")
    p.add_argument("--solo", nargs="*", help="nombres de variantes a probar")
    a = p.parse_args()
    datos = construir_cache(a.rehacer_cache)
    print(f"famosos={len(datos['fam'])}  aumentos={len(datos['aug'])}  "
          f"cigala(ref)={len(datos['cig'])}\n")
    nombres = list(VARIANTES)
    if a.rapido:
        nombres = nombres[:2]
    if a.solo:
        nombres = [n for n in nombres if n in a.solo]
    cab = (f"{'variante':50s} {'EER_cig':>8s} {'dp_cig':>7s} {'R1_cig':>7s} "
           f"{'EER_aug':>8s} {'dp_aug':>7s} {'R1_aug':>7s} {'TAR@1%':>7s} {'ms':>6s}")
    print(cab); print("-" * len(cab))
    for n in nombres:
        base = {} if n.startswith(("R5", "R6")) else BASE_RONDAS_1_4
        cfg = replace(ConfigEBGM(), **{**base, **VARIANTES[n]})
        r = evaluar_config(cfg, datos)
        print(f"{n:50s} {r['EER_cig']*100:7.1f}% {r['dp_cig']:7.2f} {r['R1_cig']*100:6.1f}% "
              f"{r['EER_aug']*100:7.1f}% {r['dp_aug']:7.2f} {r['R1_aug']*100:6.1f}% "
              f"{r['TAR1']*100:6.1f}% {r['ms']:6.0f}", flush=True)


if __name__ == "__main__":
    main()
