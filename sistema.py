"""
Sistema de reconocimiento facial — Práctica Sistemas Biométricos (EBGM).

Pipeline completo:
    foto -> [DETECTOR YuNet] -> cara 112x112 -> [EXTRACTOR EBGM] -> template -> [MATCHER] -> BD (.csv)

La clase `SistemaReconocimiento` implementa los 9 métodos que exige el enunciado,
en tres variantes según qué parte del pipeline se omite:

  - normal : foto -> detector -> extractor -> matcher     (inscribir / verificar / identificar)
  - _sd    : foto (ya recortada) -> extractor -> matcher   (omite el DETECTOR)
  - _t     : template -> matcher                           (omite detector y EXTRACTOR)

Base de datos: un .csv con una fila por persona inscrita:
    id,template          (template = jets EBGM crudos, en base64)
"""

from __future__ import annotations

import csv
import os

import numpy as np

from detector import DetectorCaras, ErrorDeteccion
from ebgm import ExtractorEBGM, ConfigEBGM

AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_BD_POR_DEFECTO = os.path.join(AQUI, "bd_sistema.csv")
RUTA_UMBRAL = os.path.join(AQUI, "umbral.txt")


def _umbral_calibrado(por_defecto: float = 0.047) -> float:
    """Lee el umbral calibrado (umbral.txt, generado por evaluar.py) si existe."""
    try:
        with open(RUTA_UMBRAL) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return por_defecto


class SistemaReconocimiento:
    def __init__(self, ruta_bd: str = RUTA_BD_POR_DEFECTO,
                 umbral: float | None = None,
                 cfg: ConfigEBGM | None = None,
                 politica_multiples: str = "mayor"):
        """
        ruta_bd:  fichero .csv de la base de datos (se crea si no existe).
        umbral:   similitud mínima para aceptar una coincidencia (verificar/identificar).
                  Calibrado con calibrar.py; por defecto cerca del EER.
        """
        self.ruta_bd = ruta_bd
        self.umbral = _umbral_calibrado() if umbral is None else umbral
        self.detector = DetectorCaras(politica_multiples=politica_multiples)
        self.extractor = ExtractorEBGM(cfg)
        self._bd: dict[str, np.ndarray] = {}
        self._cargar_bd()

    # ================================================================= BD CSV
    def _cargar_bd(self) -> None:
        self._bd.clear()
        if not os.path.isfile(self.ruta_bd):
            return
        with open(self.ruta_bd, newline="", encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                try:
                    self._bd[fila["id"]] = self.extractor.deserializar(fila["template"])
                except Exception:
                    # Fila corrupta/manipulada: se ignora (ver análisis de ataques).
                    continue

    def _guardar_bd(self) -> None:
        tmp = self.ruta_bd + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "template"])
            for id_, tpl in self._bd.items():
                w.writerow([id_, self.extractor.serializar(tpl)])
        os.replace(tmp, self.ruta_bd)

    def personas(self) -> list[str]:
        return list(self._bd)

    def recargar(self) -> None:
        """Vuelve a leer la BD del disco (p.ej. tras un ataque de escritura)."""
        self._cargar_bd()

    # ============================================================ extracción
    def _template_desde_foto(self, foto, inscribiendo: bool = False) -> np.ndarray:
        """foto -> detector -> extractor -> template."""
        if inscribiendo:
            # al inscribir conviene ser estricto si hay varias caras
            det = DetectorCaras(politica_multiples="error")
            cara = det.extraer_cara(foto)
        else:
            cara = self.detector.extraer_cara(foto)
        return self.extractor.extraer(cara)

    def _template_desde_cara(self, foto) -> np.ndarray:
        """cara ya recortada -> extractor -> template (omite detector)."""
        cara = DetectorCaras.cargar_imagen(foto)
        return self.extractor.extraer(cara)

    # ============================================================ matching
    def _inscribir_template(self, id_: str, template: np.ndarray) -> None:
        if not isinstance(id_, str) or not id_.strip():
            raise ValueError("El id debe ser un texto no vacío")
        self._bd[id_] = self.extractor._como_template(template)
        self._guardar_bd()

    def _verificar_template(self, id_: str, template: np.ndarray) -> dict:
        if id_ not in self._bd:
            return {"verificado": False, "similitud": None,
                    "motivo": f"'{id_}' no está inscrito"}
        s = self.extractor.similitud(self._bd[id_], template)
        return {"verificado": s >= self.umbral, "similitud": s, "umbral": self.umbral}

    def _identificar_template(self, template: np.ndarray) -> dict:
        if not self._bd:
            return {"id": None, "similitud": None, "motivo": "BD vacía"}
        tpl = self.extractor._como_template(template)
        ranking = sorted(
            ((id_, self.extractor.similitud(t, tpl)) for id_, t in self._bd.items()),
            key=lambda kv: kv[1], reverse=True)
        mejor_id, mejor_s = ranking[0]
        aceptado = mejor_s >= self.umbral
        return {"id": mejor_id if aceptado else None,
                "similitud": mejor_s, "umbral": self.umbral,
                "aceptado": aceptado, "ranking": ranking[:5]}

    # ============================================================= API (9 métodos)
    # ---- pipeline completo -------------------------------------------------
    def inscribir(self, id: str, foto) -> None:
        """Inscribe a una persona a partir de su nombre y una foto."""
        self._inscribir_template(id, self._template_desde_foto(foto, inscribiendo=True))

    def verificar(self, id: str, foto) -> dict:
        """Verifica (1:1) que la foto corresponde a la persona 'id'."""
        return self._verificar_template(id, self._template_desde_foto(foto))

    def identificar(self, foto) -> dict:
        """Identifica (1:N) a quién corresponde la foto."""
        return self._identificar_template(self._template_desde_foto(foto))

    # ---- sin detector (_sd) -------------------------------------------------
    def inscribir_sd(self, id: str, foto) -> None:
        """Inscribe a partir de una foto ya recortada (omite el detector)."""
        self._inscribir_template(id, self._template_desde_cara(foto))

    def verificar_sd(self, id: str, foto) -> dict:
        """Verifica a partir de una foto ya recortada (omite el detector)."""
        return self._verificar_template(id, self._template_desde_cara(foto))

    def identificar_sd(self, foto) -> dict:
        """Identifica a partir de una foto ya recortada (omite el detector)."""
        return self._identificar_template(self._template_desde_cara(foto))

    # ---- solo template (_t) -------------------------------------------------
    def inscribir_t(self, id: str, template) -> None:
        """Inscribe a partir de un template (omite detector y extractor)."""
        self._inscribir_template(id, template)

    def verificar_t(self, id: str, template) -> dict:
        """Verifica a partir de un template (omite detector y extractor)."""
        return self._verificar_template(id, template)

    def identificar_t(self, template) -> dict:
        """Identifica a partir de un template (omite detector y extractor)."""
        return self._identificar_template(template)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sistema de reconocimiento facial EBGM")
    p.add_argument("accion", choices=["inscribir", "verificar", "identificar", "listar"])
    p.add_argument("--id")
    p.add_argument("--foto")
    p.add_argument("--bd", default=RUTA_BD_POR_DEFECTO)
    a = p.parse_args()

    sis = SistemaReconocimiento(ruta_bd=a.bd)
    try:
        if a.accion == "listar":
            print(f"{len(sis.personas())} personas:", ", ".join(sis.personas()))
        elif a.accion == "inscribir":
            sis.inscribir(a.id, a.foto)
            print(f"Inscrito: {a.id}")
        elif a.accion == "verificar":
            print(sis.verificar(a.id, a.foto))
        elif a.accion == "identificar":
            r = sis.identificar(a.foto)
            print(r)
    except ErrorDeteccion as e:
        print("Error de detección:", e)
