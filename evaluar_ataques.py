"""
Evaluación de seguridad del sistema EBGM — escenarios de ataque.

El enunciado pide analizar qué pasa cuando el sistema es atacado y el atacante:
  (A) se salta alguna parte del sistema,
  (B) consigue LEER la base de datos,
  (C) consigue ESCRIBIR en la base de datos.

Este script DEMUESTRA cada ataque sobre una copia de la BD y comenta la
vulnerabilidad y la mitigación. El objetivo de la práctica no es un sistema sin
defectos (imposible), sino conocer los nuestros antes que el grupo rival.

Uso:
    python evaluar_ataques.py
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile

import numpy as np

from detector import DetectorCaras
from ebgm import ExtractorEBGM
from sistema import SistemaReconocimiento

AQUI = os.path.dirname(os.path.abspath(__file__))


def titulo(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def bd_temporal() -> str:
    """Copia la BD real a un fichero temporal para no estropear la de verdad."""
    orig = os.path.join(AQUI, "bd_sistema.csv")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    tmp.close()
    shutil.copy(orig, tmp.name)
    return tmp.name


def main():
    det = DetectorCaras()
    ext = ExtractorEBGM()

    # ----------------------------------------------------------------- (A)
    titulo("(A) EL ATACANTE SE SALTA PARTE DEL SISTEMA (métodos _sd / _t)")
    print("""El sistema expone inscribir/verificar/identificar en 3 variantes. Las
variantes _sd (sin detector) y _t (sólo template) existen para interoperar,
pero si quedan EXPUESTAS al atacante, le dejan inyectar datos saltándose las
defensas del pipeline (el detector valida la entrada; el extractor fija el
formato).""")
    bd = bd_temporal()
    sis = SistemaReconocimiento(ruta_bd=bd)

    # A.1 - inyección de template robado: autenticación perfecta sin la cara
    print("\n A.1  Replay de template: el atacante tiene el template de 'cigala'")
    print("      (p.ej. porque leyó la BD) y lo presenta por verificar_t:")
    tpl_victima = sis._bd["cigala"]
    r = sis.verificar_t("cigala", tpl_victima)
    print(f"      verificar_t('cigala', template_robado) -> {r['verificado']} "
          f"(sim={r['similitud']:.3f})  ⇒ SUPLANTACIÓN TOTAL")

    # A.2 - entrada que el detector habría rechazado, colada por _sd
    print("\n A.2  Entrada no validada por _sd: imagen de ruido (sin cara).")
    print("      inscribir/verificar exigen que el detector encuentre una cara;")
    print("      _sd se la salta y acepta cualquier imagen 112x112:")
    ruido = (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
    import cv2
    tmp_png = os.path.join(tempfile.gettempdir(), "ruido_ataque.png")
    cv2.imwrite(tmp_png, ruido)
    r = sis.identificar_sd(tmp_png)
    print(f"      identificar_sd(ruido) -> id={r['id']} sim={r['similitud']:.3f} "
          f"(el pipeline normal lo habría rechazado: 'no se detecta cara')")
    print("""
   Mitigación: no exponer _sd/_t al exterior (son sólo para intercambio entre
   grupos / pruebas); exigir detección+calidad en el camino real; antispoofing
   (liveness); y firmar/verificar la procedencia de cada template.""")

    # ----------------------------------------------------------------- (B)
    titulo("(B) EL ATACANTE LEE LA BASE DE DATOS")
    print("""La BD es un .csv con los templates en claro (base64). Si el atacante la
lee, obtiene los rasgos biométricos de todos los inscritos.""")
    with open(bd, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    print(f"\n B.1  Exfiltración: {len(filas)} identidades legibles. Ejemplo:")
    print(f"      {filas[0]['id']},{filas[0]['template'][:48]}...")

    print("\n B.2  Con un template robado el atacante suplanta por _t (ver A.1)")
    print("      o lo reinyecta en otro sistema que use el mismo formato.")

    print("\n B.3  ¿Se puede 'invertir' el template para reconstruir la cara?")
    print("      El template EBGM son MAGNITUDES de Gabor (se pierde la fase y la")
    print("      posición exacta), así que no se recupera la foto; PERO para")
    print("      SUPLANTAR no hace falta: basta reinyectar el propio template.")
    print("""
   Mitigación: cifrar la BD en reposo; control de acceso; almacenar los
   templates transformados (biometric template protection: biohashing,
   fuzzy vault) para que lo filtrado no sea reutilizable; y detectar replays.""")

    # ----------------------------------------------------------------- (C)
    titulo("(C) EL ATACANTE ESCRIBE EN LA BASE DE DATOS")
    print("""Si el atacante puede ESCRIBIR el .csv, controla a quién reconoce el
sistema. Mostramos tres sub-ataques sobre la BD temporal.""")

    # C.1 - sustitución de identidad: el atacante pone SU template bajo 'cigala'
    print("\n C.1  Sustitución de identidad: el atacante sobrescribe el template")
    print("      de 'cigala' con el SUYO. A partir de ahí, él ES 'cigala'.")
    cara_atacante = det.extraer_cara(os.path.join(AQUI, "Dataset", "adele_2016.jpg"))
    tpl_atacante = ext.extraer(cara_atacante)
    antes = sis.verificar_t("cigala", tpl_atacante)
    sis._bd["cigala"] = tpl_atacante          # (simula escritura directa en el .csv)
    sis._guardar_bd()
    sis.recargar()
    despues = sis.verificar_t("cigala", tpl_atacante)
    print(f"      sim(atacante, 'cigala')  antes={antes['similitud']:+.3f} -> "
          f"después de escribir={despues['similitud']:+.3f} (=1.0: ya ES 'cigala')")

    # C.2 - puerta trasera: alta de una identidad nueva controlada por el atacante
    print("\n C.2  Puerta trasera: el atacante da de alta 'admin_falso' con su cara.")
    sis.inscribir_t("admin_falso", tpl_atacante)
    print(f"      personas tras el alta: {len(sis.personas())} "
          f"(incluye 'admin_falso': {'admin_falso' in sis.personas()})")

    # C.3 - denegación de servicio: corromper filas -> se ignoran al cargar
    print("\n C.3  Sabotaje/DoS: el atacante corrompe filas del .csv.")
    with open(bd, "a", encoding="utf-8") as f:
        f.write("cigala,ESTO_NO_ES_BASE64_valido!!!\n")  # fila basura
    sis.recargar()
    print(f"      el cargador ignora filas corruptas sin caerse; personas={len(sis.personas())}")
    print("""
   Mitigación: BD de sólo-lectura para el motor / permisos estrictos; firmar
   (HMAC) cada fila y verificar su integridad al cargar; registro de auditoría
   de altas y cambios; y copias de seguridad para recuperar la BD legítima.""")

    os.remove(bd)
    titulo("RESUMEN")
    print("""Defectos CONOCIDOS por nosotros (que es el objetivo):
  - _sd/_t permiten saltarse detector/extractor -> no exponerlos fuera.
  - Replay de template = suplantación total -> liveness + protección de template.
  - BD en claro -> cifrado + control de acceso + plantillas revocables.
  - Escritura en BD = control de identidades -> integridad (HMAC) + auditoría.
Ninguno es 'un sistema sin defectos'; son riesgos acotados y con mitigación.""")


if __name__ == "__main__":
    main()
