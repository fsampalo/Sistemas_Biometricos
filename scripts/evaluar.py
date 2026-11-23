"""
Evaluación (fase 2) del sistema de reconocimiento facial EBGM DESPLEGADO.

Usa la BD y el umbral reales (bd_sistema.csv + models/calibracion.json), es
decir, mide lo mismo que se encontrará el grupo rival.

Bloques
-------
  1. Detector: tasa de fallo de adquisición (FTA) por conjunto.
  2. Verificación 1:1
       - Genuinos de TEST (fotos_cigala_test/, nunca inscritas) contra el
         cigala de la BD (inscrito con 1 foto). Se separan las fotos NUEVAS
         (nunca usadas al ajustar) de las ref_* (usadas para ajustar en la fase 1).
       - Protocolo "1 foto contra 1 foto": todos los pares entre fotos de test.
       - Impostores: famoso vs famoso, famosos vs cigala inscrito, test del
         cigala vs famosos.
     -> EER, d', FAR/FRR en el umbral desplegado, TAR a FAR fijo, curva DET.
  3. Identificación 1:N (galería de 50): CMC (rank-1, rank-5) de las fotos de
     test y de fotos degradadas de cada famoso.
  4. Robustez por tipo de degradación (iluminación, baja resolución, JPEG,
     giro, espejo, ruido, luz lateral) usando el sistema completo.
  5. Riesgos: parejas de famosos más confundibles (gemelos / parecidos) y
     "zoo" de Doddington (identidades que atraen impostores).
  6. Tiempos por operación.

Uso:
    python scripts/evaluar.py         # informe en consola (gráficas: scripts/graficas.py)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raíz del repo

from ajuste_ebgm import aumentos, dprime, eer
from inscribir_dataset import id_desde_fichero
from reconocimiento.detector import DetectorCaras, ErrorDeteccion
from reconocimiento.rutas import DIR_CIGALA_TEST, DIR_DATASET
from reconocimiento.sistema import SistemaReconocimiento


def titulo(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def es_de_ajuste(nombre: str) -> bool:
    """Fotos de test del cigala que se usaron para ajustar la configuración."""
    return os.path.basename(nombre).startswith("ref_")


def tar_a_far(gen, imp, far):
    thr = np.quantile(imp, 1 - far)
    return float(np.mean(np.asarray(gen) >= thr)), float(thr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DIR_DATASET)
    p.add_argument("--cigala-test", default=DIR_CIGALA_TEST)
    p.add_argument("--sin-robustez", action="store_true", help="omite el bloque 4 (lento)")
    a = p.parse_args()

    # modo evaluación: necesitamos similitudes y rankings
    sis = SistemaReconocimiento(auditoria=False, exponer_puntuaciones=True)
    ext = sis.extractor
    det = DetectorCaras(politica_multiples="mayor")
    print(f"BD: {len(sis.personas())} personas | umbral={sis.umbral:.4f} | "
          f"umbral_cara={sis.umbral_cara:.3f}")
    if sis.filas_rechazadas:
        print(f"AVISO: {len(sis.filas_rechazadas)} filas de la BD rechazadas")

    # ------------------------------------------------------------- 1. detector
    titulo("1. DETECTOR — fallo de adquisición (FTA)")

    def extraer_carpeta(patron):
        tpls, fallos = {}, []
        for r in sorted(glob.glob(patron)):
            try:
                tpls[os.path.basename(r)] = ext.extraer(det.extraer_cara(r))
            except ErrorDeteccion as e:
                fallos.append((os.path.basename(r), str(e)))
        return tpls, fallos

    fam, f1 = extraer_carpeta(os.path.join(a.dataset, "*"))
    test, f3 = extraer_carpeta(os.path.join(a.cigala_test, "*"))
    for nombre, ok, f in (("famosos", fam, f1), ("cigala TEST", test, f3)):
        n = len(ok) + len(f)
        print(f"  {nombre:12s}: {len(f)}/{n} sin cara (FTA={len(f)/max(n,1):.0%})")
        for nom, e in f:
            print(f"      - {nom}: {e}")

    rep = {k: ext.representar(v) for k, v in fam.items()}
    rep_test = {k: ext.representar(v) for k, v in test.items()}
    rep_cig_bd = sis._rep["cigala"]
    S = ext.similitud_rep
    # las fotos ref_* se usaron para AJUSTAR la configuración (fase 1); las
    # demás de test nunca se han usado: se informa de ambos grupos por separado
    grupos = {"TODAS": list(rep_test),
              "NUEVAS (nunca usadas al ajustar)": [k for k in rep_test if not es_de_ajuste(k)],
              "de ajuste (ref_*)": [k for k in rep_test if es_de_ajuste(k)]}

    # ------------------------------------------------------- 2. verificación
    titulo("2. VERIFICACIÓN 1:1")
    kf = list(rep)
    imp_ff = [S(rep[kf[i]], rep[kf[j]]) for i in range(len(kf)) for j in range(i + 1, len(kf))]
    imp_fc = [S(rep[k], rep_cig_bd) for k in kf]
    imp_tf = [S(r, rep[k]) for r in rep_test.values() for k in kf]
    imp = np.array(imp_ff + imp_fc + imp_tf)
    print(f"  impostores: n={len(imp)}  media={imp.mean():+.3f}  máx={imp.max():+.3f}  "
          f"FAR en el umbral desplegado={np.mean(imp >= sis.umbral):.2%}")

    def informe(nombre, gen):
        gen = np.asarray(gen)
        if len(gen) == 0:
            return
        e, thr_e = eer(gen, imp)
        tar1, _ = tar_a_far(gen, imp, 0.01)
        print(f"\n  [{nombre}]  genuinos n={len(gen)} media={gen.mean():+.3f}")
        print(f"     d'={dprime(gen, imp):.2f}   EER={e:.1%} (umbral EER {thr_e:+.3f})")
        print(f"     en el umbral desplegado {sis.umbral:+.3f}:  "
              f"aceptadas {int(np.sum(gen >= sis.umbral))}/{len(gen)}  "
              f"(FRR={np.mean(gen < sis.umbral):.1%})   TAR @ FAR=1%: {tar1:.1%}")

    for g, ks in grupos.items():
        informe(f"TEST {g} vs cigala de la BD", [S(rep_test[k], rep_cig_bd) for k in ks])
    # protocolo «1 foto contra 1 foto»: cada foto de test hace de inscrita y
    # las demás de sonda (todos los pares ordenados)
    kt = list(rep_test)
    informe("pares entre fotos de TEST (1 foto vs 1 foto)",
            [S(rep_test[x], rep_test[y]) for x in kt for y in kt if x != y])
    print("\n  Similitud de cada foto de TEST con el cigala de la BD:")
    for k, r in rep_test.items():
        s = S(r, rep_cig_bd)
        print(f"     {k:28s} {s:+.3f}  {'ACEPTA' if s >= sis.umbral else 'rechaza'}"
              f"{'' if not es_de_ajuste(k) else '   (de ajuste)'}")
    print("\n  Gráficas: python scripts/graficas.py  ->  salida/graficas/")

    # ----------------------------------------------------- 3. identificación
    titulo("3. IDENTIFICACIÓN 1:N (galería de 50)")
    ids_bd = sis.personas()

    def rango(rep_sonda, verdadero):
        s = sis._similitudes_rep(rep_sonda, ids_bd)
        orden = [ids_bd[k] for k in np.argsort(-s)]
        return orden.index(verdadero) + 1, float(s.max()), orden[0]

    rs = {k: rango(r, "cigala") for k, r in rep_test.items()}
    for g, ks in grupos.items():
        if not ks:
            continue
        v = [rs[k] for k in ks]
        r1 = np.mean([r[0] == 1 for r in v]); r5 = np.mean([r[0] <= 5 for r in v])
        ac = np.mean([r[0] == 1 and r[1] >= sis.umbral for r in v])
        otro = np.mean([r[0] != 1 and r[1] >= sis.umbral for r in v])
        print(f"  cigala TEST {g:34s} n={len(v):2d}  rank-1={r1:4.0%}  rank-5={r5:4.0%}  "
              f"identificado bien={ac:4.0%}  como OTRA persona={otro:4.0%}")
    print(f"  [azar: rank-1 ≈ {1/len(ids_bd):.0%}, rank-5 ≈ {5/len(ids_bd):.0%}]")
    for k, (rk, s, primero) in rs.items():
        print(f"     {k:28s} rango={rk:2d}  sim_max={s:+.3f}  "
              f"-> {primero if s >= sis.umbral else 'nadie (bajo umbral)'}")

    # --------------------------------------------------------- 4. robustez
    if not a.sin_robustez:
        titulo("4. ROBUSTEZ — fotos degradadas de cada famoso por el sistema completo")
        por_tipo: dict[str, list] = {}
        fta: dict[str, int] = {}
        for r in sorted(glob.glob(os.path.join(a.dataset, "*"))):
            id_ = id_desde_fichero(r)
            if id_ not in sis._rep:
                continue
            for tipo, img in aumentos(det.cargar_imagen(r)).items():
                try:
                    res = sis.identificar(img)
                    ver = sis.verificar(id_, img)
                    por_tipo.setdefault(tipo, []).append(
                        (res.get("ranking", [[None]])[0][0] == id_, ver["verificado"]))
                except Exception:
                    fta[tipo] = fta.get(tipo, 0) + 1
        print(f"  {'degradación':14s} {'rank-1':>7s} {'verifica':>9s} {'FTA':>5s}")
        for tipo, v in por_tipo.items():
            v = np.array(v)
            print(f"  {tipo:14s} {v[:,0].mean():7.0%} {v[:,1].mean():9.0%} {fta.get(tipo,0):5d}")

    # ----------------------------------------------------------- 5. riesgos
    titulo("5. RIESGOS — parejas más confundibles y zoo de Doddington")
    pares = sorted(((S(rep[kf[i]], rep[kf[j]]), kf[i], kf[j])
                    for i in range(len(kf)) for j in range(i + 1, len(kf))), reverse=True)
    print("  Parejas de famosos más parecidas para el sistema (posibles falsas aceptaciones):")
    for s, x, y in pares[:8]:
        marca = "  <-- SUPERA EL UMBRAL" if s >= sis.umbral else ""
        print(f"     {s:+.3f}  {x}  ~  {y}{marca}")
    atraccion = {}
    for id_ in ids_bd:
        s = sis._similitudes_rep(sis._rep[id_], [k for k in ids_bd if k != id_])
        atraccion[id_] = float(np.sort(s)[-3:].mean())
    top = sorted(atraccion.items(), key=lambda kv: -kv[1])[:5]
    print("  Identidades que más 'atraen' a otras (media top-3, candidatas a 'lobo/cordero'):")
    for k, v in top:
        print(f"     {v:+.3f}  {k}")
    print(f"  cigala: {atraccion.get('cigala', float('nan')):+.3f} "
          "(si fuera muy alto, el cigala sería un 'imán' de falsos positivos)")
    print("  Nota: el Dataset incluye GEMELOS (Twin Melody, gemelos Weasley, Olsen):")
    print("  su gemelo NO inscrito pasará casi seguro por la persona inscrita. Es un")
    print("  límite conocido de cualquier biometría facial (y más de EBGM).")

    # ------------------------------------------------------------ 6. tiempos
    titulo("6. TIEMPOS")
    foto = sorted(glob.glob(os.path.join(a.dataset, "*")))[0]
    cara = det.extraer_cara(foto)
    tpl = ext.extraer(cara)
    for nombre, f in (("verificar (foto)", lambda: sis.verificar("cigala", foto)),
                      ("identificar (foto)", lambda: sis.identificar(foto)),
                      ("identificar_sd (cara)", lambda: sis.identificar_sd(cara)),
                      ("identificar_t (template)", lambda: sis.identificar_t(tpl))):
        t0 = time.perf_counter()
        for _ in range(5):
            f()
        print(f"  {nombre:26s} {(time.perf_counter()-t0)/5*1000:7.1f} ms")


if __name__ == "__main__":
    main()
