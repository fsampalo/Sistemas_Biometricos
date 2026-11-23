"""
Sistema de reconocimiento facial — Práctica Sistemas Biométricos (EBGM).

Pipeline completo:
    foto -> [DETECTOR YuNet] -> cara 112x112 alineada -> [EXTRACTOR EBGM] -> template
         -> [MATCHER EBGM] -> BD (.csv)

La clase `SistemaReconocimiento` implementa los 9 métodos del enunciado, en tres
variantes según qué parte del pipeline se omite:

  - normal : foto -> detector -> extractor -> matcher   (inscribir/verificar/identificar)
  - _sd    : cara ya recortada -> extractor -> matcher   (omite el DETECTOR)
  - _t     : template -> matcher                         (omite detector y EXTRACTOR)

Base de datos (.csv), una fila por persona:
    id,template,mac
  template = "aesgcm:" + AES-256-GCM(jets EBGM serializados), AAD = id | huella
  mac      = HMAC-SHA256(clave, id | template cifrado | huella del modelo)

Defensas incluidas (ver evaluar_ataques.py e INFORME.md):
  * Confidencialidad: los templates se guardan CIFRADOS; leer el .csv no
    da templates reutilizables.
  * Integridad por fila (HMAC): filas añadidas/alteradas sin la clave se
    rechazan al cargar. La BD queda ligada al modelo EBGM con el que se creó
    (y por tanto a sus umbrales). MAC global en <bd>.mac: detecta borrados.
  * Validación de ids (longitud, caracteres, sin inyección de fórmulas CSV) y
    de templates (forma, NaN/Inf, rangos).
  * Control de "caricidad" EBGM: la entrada de _sd y _t debe parecerse a una
    cara según el bunch graph; ruido, imágenes planas o templates basura se
    rechazan aunque se salten el detector.
  * inscribir no sobrescribe un id existente salvo que se pida explícitamente.
  * Aviso de posible duplicado (misma cara inscrita con otro nombre).
  * Registro de auditoría de altas y bajas.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import time
import unicodedata

import base64

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .detector import DetectorCaras, ErrorDeteccion
from .ebgm import ConfigEBGM, ExtractorEBGM, ModeloEBGM, RUTA_MODELO

from .rutas import DIR_MODELOS, DIR_SALIDA, RUTA_BD

RUTA_BD_POR_DEFECTO = RUTA_BD
RUTA_CLAVE_POR_DEFECTO = os.path.join(DIR_MODELOS, "clave_bd.key")
RUTA_AUDITORIA = os.path.join(DIR_SALIDA, "auditoria.log")
RUTA_CALIBRACION = os.path.join(DIR_MODELOS, "calibracion.json")  # solo informe

# Umbral de decisión (similitud EBGM). Lo calibra inscribir_dataset.py y se
# guarda DENTRO del modelo (models/ebgm_modelo.npz), cuyo hash va en el HMAC de
# cada fila de la BD: alterar el umbral invalida la BD. Este es el de reserva.
UMBRAL_POR_DEFECTO = 0.20
# Por debajo de esta "caricidad" la entrada no se considera una cara.
UMBRAL_CARA_POR_DEFECTO = 0.80

PREFIJO_CIFRADO = "aesgcm:"   # celda template = prefijo + base64(nonce | cifrado)
MAX_PERSONAS = 10_000
MAX_LONG_ID = 64
_PATRON_ID = re.compile(r"^[\w][\w .,'()\-]*$", re.UNICODE)


class ErrorSistema(Exception):
    """Error controlado del sistema (entrada inválida, id inexistente...)."""


class ErrorCalidad(ErrorSistema):
    """La entrada no parece una cara (control de caricidad EBGM)."""


def validar_id(id_) -> str:
    if not isinstance(id_, str):
        raise ErrorSistema("El id debe ser un texto")
    id_ = unicodedata.normalize("NFC", id_.strip())
    if not id_ or len(id_) > MAX_LONG_ID:
        raise ErrorSistema(f"El id debe tener entre 1 y {MAX_LONG_ID} caracteres")
    if not _PATRON_ID.match(id_):
        # Bloquea, entre otros, ids que empiezan por = + - @ (inyección de
        # fórmulas al abrir el .csv en una hoja de cálculo), saltos de línea,
        # comas de separación de campos raras, comillas, etc.
        raise ErrorSistema(f"id con caracteres no permitidos: {id_!r}")
    return id_


class SistemaReconocimiento:
    def __init__(self, ruta_bd: str = RUTA_BD_POR_DEFECTO,
                 umbral: float | None = None,
                 cfg: ConfigEBGM | None = None,
                 ruta_modelo: str = RUTA_MODELO,
                 clave: bytes | None = None,
                 ruta_clave: str = RUTA_CLAVE_POR_DEFECTO,
                 politica_multiples: str = "mayor",
                 control_cara: bool = True,
                 exponer_puntuaciones: bool = False,
                 estricto: bool = True,
                 auditoria: bool = True):
        """
        ruta_bd:      fichero .csv de la base de datos (se crea si no existe).
        umbral:       similitud mínima para aceptar (por defecto, la calibrada).
        clave:        clave HMAC de la BD. Si no se da: variable de entorno
                      EBGM_CLAVE_BD (hex) o fichero ruta_clave (se crea si falta).
        control_cara: rechaza entradas que no parecen una cara (_sd, _t, inscribir).
        exponer_puntuaciones: por defecto False: verificar/identificar solo
                      devuelven la DECISIÓN (sin similitudes ni ranking), lo que
                      quita al atacante el gradiente del hill-climbing. True solo
                      para evaluar (curvas DET, rankings) en un entorno controlado.
        estricto:     si la integridad de la BD falla (filas alteradas, borradas
                      o restauradas), verificar/identificar se niegan a operar
                      (falla en cerrado) hasta que se reconstruya la BD.
        """
        if not os.path.isfile(ruta_modelo):
            raise FileNotFoundError(
                f"Falta el modelo EBGM ({ruta_modelo}). Ejecuta: python scripts/inscribir_dataset.py")
        self.extractor = ExtractorEBGM(cfg, modelo=ruta_modelo)
        self.huella_modelo = ModeloEBGM.sha256(ruta_modelo)[:16]
        cal = self.extractor.modelo.info
        self.umbral = float(umbral) if umbral is not None else float(
            cal.get("umbral", UMBRAL_POR_DEFECTO))
        self.umbral_cara = float(cal.get("umbral_cara", UMBRAL_CARA_POR_DEFECTO))
        if not (np.isfinite(self.umbral) and np.isfinite(self.umbral_cara)):
            raise ErrorSistema("Umbrales del modelo no válidos")
        self.control_cara = control_cara
        self.exponer_puntuaciones = exponer_puntuaciones
        self.auditoria = auditoria
        self.estricto = estricto
        self.detector = DetectorCaras(politica_multiples=politica_multiples)
        self._det_estricto = DetectorCaras(politica_multiples="error")
        self.ruta_bd = ruta_bd
        self._clave = clave if clave is not None else self._obtener_clave(ruta_clave)
        self._bd: dict[str, np.ndarray] = {}
        self._rep: dict[str, tuple] = {}
        self.filas_rechazadas: list[tuple[int, str]] = []
        self._cargar_bd()

    # ============================================================ clave HMAC
    @staticmethod
    def _obtener_clave(ruta: str) -> bytes:
        env = os.environ.get("EBGM_CLAVE_BD")
        if env:
            try:
                return bytes.fromhex(env)
            except ValueError:
                raise ErrorSistema("EBGM_CLAVE_BD debe ser hexadecimal") from None
        if os.path.isfile(ruta):
            with open(ruta, "rb") as f:
                clave = bytes.fromhex(f.read().decode().strip())
            if len(clave) < 16:
                raise ErrorSistema("Clave de la BD demasiado corta")
            return clave
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        clave = secrets.token_bytes(32)
        fd = os.open(ruta, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(clave.hex())
        return clave

    def _mac(self, id_: str, tpl_b64: str) -> str:
        msg = f"{id_}\x1f{tpl_b64}\x1f{self.huella_modelo}".encode("utf-8")
        return hmac.new(self._clave, msg, hashlib.sha256).hexdigest()

    # ======================================================= cifrado en reposo
    # El HMAC da integridad, no confidencialidad: con templates en claro, leer
    # el .csv bastaría para hacer replay por _t, enlazar personas entre BDs o
    # intentar reconstruir la cara a partir de las magnitudes de Gabor. Por eso
    # la columna template va cifrada con AES-256-GCM; la clave se deriva de la
    # clave maestra (separada de la del HMAC) y el id + huella del modelo van
    # como datos asociados (AAD), así que un template cifrado no se puede mover
    # a otra fila.
    def _clave_cifrado(self) -> bytes:
        return hmac.new(self._clave, b"ebgm-cifrado-templates-v1", hashlib.sha256).digest()

    def _aad(self, id_: str) -> bytes:
        return f"{id_}\x1f{self.huella_modelo}".encode("utf-8")

    def _cifrar(self, id_: str, tpl_b64: str) -> str:
        nonce = secrets.token_bytes(12)
        ct = AESGCM(self._clave_cifrado()).encrypt(nonce, tpl_b64.encode("ascii"), self._aad(id_))
        return PREFIJO_CIFRADO + base64.b64encode(nonce + ct).decode("ascii")

    def _descifrar(self, id_: str, celda: str) -> str:
        if not celda.startswith(PREFIJO_CIFRADO):
            raise ValueError("template sin cifrar en la BD")
        try:
            raw = base64.b64decode(celda[len(PREFIJO_CIFRADO):], validate=True)
            return AESGCM(self._clave_cifrado()).decrypt(
                raw[:12], raw[12:], self._aad(id_)).decode("ascii")
        except (InvalidTag, ValueError) as e:
            raise ValueError(f"template cifrado inválido ({type(e).__name__})") from None

    # ================================================================= BD CSV
    def _cargar_bd(self) -> None:
        self._bd.clear()
        self._rep.clear()
        self.filas_rechazadas = []
        if not os.path.isfile(self.ruta_bd):
            # BD vacía legítima solo si tampoco hay .mac (si lo hay, alguien
            # ha borrado el .csv entero)
            self.integridad_global = not os.path.exists(self.ruta_bd + ".mac")
            if not self.integridad_global:
                self._auditar("bd_borrada", ruta=self.ruta_bd)
            return
        csv.field_size_limit(10_000_000)
        with open(self.ruta_bd, newline="", encoding="utf-8") as f:
            lector = csv.DictReader(f)
            if lector.fieldnames is None or not {"id", "template", "mac"} <= set(lector.fieldnames):
                raise ErrorSistema(f"Cabecera de BD inválida en {self.ruta_bd}: "
                                   f"{lector.fieldnames}")
            for n, fila in enumerate(lector, start=2):
                try:
                    id_ = validar_id(fila["id"])
                    if not hmac.compare_digest(self._mac(id_, fila["template"]),
                                               fila["mac"] or ""):
                        raise ValueError("HMAC incorrecto (fila alterada o ajena)")
                    if id_ in self._bd:
                        raise ValueError("id duplicado; se conserva la primera fila")
                    if len(self._bd) >= MAX_PERSONAS:
                        raise ValueError("límite de personas alcanzado")
                    tpl = self.extractor.como_template(self._descifrar(id_, fila["template"]))
                    self._bd[id_] = tpl
                    self._rep[id_] = self.extractor.representar(tpl)
                except Exception as e:  # fila corrupta/manipulada: se aparta
                    self.filas_rechazadas.append((n, str(e)))
        if self.filas_rechazadas:
            self._auditar("bd_filas_rechazadas", detalle=self.filas_rechazadas[:20])
        # MAC global (fichero .mac): detecta filas BORRADAS o la BD entera
        # sustituida, que el HMAC por fila no puede ver.
        self.integridad_global = self._comprobar_mac_global()
        if not self.integridad_global:
            self._auditar("bd_mac_global_incorrecto", n=len(self._bd))

    def _mac_global(self) -> str:
        filas = "".join(f"{k}{self._mac(k, self.extractor.serializar(v))}"
                            for k, v in sorted(self._bd.items()))
        msg = f"{len(self._bd)}{filas}{self.huella_modelo}".encode("utf-8")
        return hmac.new(self._clave, msg, hashlib.sha256).hexdigest()

    def _comprobar_mac_global(self) -> bool:
        try:
            with open(self.ruta_bd + ".mac", encoding="ascii") as f:
                guardado = f.read().strip()
        except OSError:
            return not self._bd and not self.filas_rechazadas
        return (not self.filas_rechazadas) and hmac.compare_digest(guardado, self._mac_global())

    def _guardar_bd(self) -> None:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["id", "template", "mac"])
        for id_, tpl in self._bd.items():
            cifrado = self._cifrar(id_, self.extractor.serializar(tpl))
            w.writerow([id_, cifrado, self._mac(id_, cifrado)])
        d = os.path.dirname(os.path.abspath(self.ruta_bd))
        os.makedirs(d, exist_ok=True)
        tmp = f"{self.ruta_bd}.{os.getpid()}.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            f.write(buf.getvalue())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.ruta_bd)   # escritura atómica
        with open(tmp, "w", encoding="ascii") as f:
            f.write(self._mac_global())
        os.replace(tmp, self.ruta_bd + ".mac")
        self.integridad_global = True

    def _auditar(self, evento: str, **datos) -> None:
        if not self.auditoria:
            return
        try:
            os.makedirs(os.path.dirname(RUTA_AUDITORIA), exist_ok=True)
            with open(RUTA_AUDITORIA, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "evento": evento, **datos},
                                   ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def personas(self) -> list[str]:
        return list(self._bd)

    def recargar(self) -> None:
        """Vuelve a leer la BD del disco."""
        self._cargar_bd()

    def eliminar(self, id: str) -> bool:
        id_ = validar_id(id)
        if self._bd.pop(id_, None) is None:
            return False
        self._rep.pop(id_, None)
        self._guardar_bd()
        self._auditar("baja", id=id_)
        return True

    # ============================================================ extracción
    def _comprobar_cara(self, tpl: np.ndarray) -> float:
        c = self.extractor.puntuacion_cara(tpl)
        if self.control_cara and c < self.umbral_cara:
            raise ErrorCalidad(f"La entrada no parece una cara (caricidad {c:.3f} "
                               f"< {self.umbral_cara:.3f})")
        return c

    def _template_desde_foto(self, foto, inscribiendo: bool = False) -> np.ndarray:
        """foto -> detector -> extractor -> template."""
        if inscribiendo:
            tpl = self.extractor.extraer_inscripcion(self._det_estricto.extraer_cara(foto))
        else:
            tpl = self.extractor.extraer(self.detector.extraer_cara(foto))
        self._comprobar_cara(tpl)
        return tpl

    def _template_desde_cara(self, foto, inscribiendo: bool = False) -> np.ndarray:
        """cara ya recortada (112x112 alineada) -> extractor -> template."""
        cara = DetectorCaras.cargar_imagen(foto)
        h, w = cara.shape[:2]
        if max(h, w) > 4 * min(h, w):
            raise ErrorCalidad(f"Proporción de recorte absurda para una cara ({w}x{h})")
        tpl = (self.extractor.extraer_inscripcion(cara) if inscribiendo
               else self.extractor.extraer(cara))
        self._comprobar_cara(tpl)
        return tpl

    def _template_externo(self, template) -> np.ndarray:
        try:
            tpl = self.extractor.como_template(template)
        except ValueError as e:
            raise ErrorSistema(f"Template inválido: {e}") from None
        self._comprobar_cara(tpl)
        return tpl

    # ============================================================ matching
    def _inscribir_template(self, id_, template: np.ndarray, metodo: str,
                            sobrescribir: bool) -> dict:
        self._exigir_integridad()   # no re-firmar una BD manipulada
        id_ = validar_id(id_)
        if id_ in self._bd and not sobrescribir:
            raise ErrorSistema(f"'{id_}' ya está inscrito (usa sobrescribir=True "
                               "para re-inscribirlo)")
        if id_ not in self._bd and len(self._bd) >= MAX_PERSONAS:
            raise ErrorSistema("Límite de personas inscritas alcanzado")
        rep = self.extractor.representar(template)
        # ¿esta cara ya está inscrita con otro nombre?
        duplicado = None
        otros = [k for k in self._bd if k != id_]
        if otros:
            s = self._similitudes_rep(rep, otros)
            k = int(np.argmax(s))
            if s[k] >= self.umbral:
                duplicado = {"id": otros[k], "similitud": float(s[k])}
        self._bd[id_] = template
        self._rep[id_] = rep
        self._guardar_bd()
        self._auditar("alta", id=id_, metodo=metodo, sobrescrito=sobrescribir,
                      posible_duplicado=duplicado,
                      sha=hashlib.sha256(template.tobytes()).hexdigest()[:16])
        # la auditoría guarda la similitud; la respuesta solo si se exponen puntuaciones
        if duplicado is not None and not self.exponer_puntuaciones:
            duplicado = {"id": duplicado["id"]}
        return {"inscrito": id_, "posible_duplicado": duplicado}

    def _similitudes_rep(self, rep, ids: list[str]) -> np.ndarray:
        return self.extractor.similitudes_rep(rep, [self._rep[k] for k in ids])

    def _exigir_integridad(self) -> None:
        if self.estricto and not self.integridad_global:
            raise ErrorSistema(
                "Integridad de la BD comprometida (filas alteradas, borradas o "
                f"restauradas; {len(self.filas_rechazadas)} filas rechazadas). El "
                "sistema no opera hasta reconstruir la BD (python scripts/inscribir_dataset.py)")

    def _verificar_template(self, id_, template: np.ndarray) -> dict:
        self._exigir_integridad()
        id_ = validar_id(id_)
        rep = self.extractor.representar(template)
        if id_ not in self._bd:
            # Misma respuesta (y mismo trabajo) que un rechazo normal: antes
            # se respondía "'x' no está inscrito", lo que permitía enumerar
            # quién está en la BD. El motivo queda solo en la auditoría.
            self._similitudes_rep(rep, list(self._bd)[:1])
            self._auditar("verificar_id_desconocido", id=id_)
            r = {"verificado": False}
            if self.exponer_puntuaciones:
                r.update(similitud=None, umbral=self.umbral, motivo="id no inscrito")
            return r
        s = float(self._similitudes_rep(rep, [id_])[0])
        r = {"verificado": bool(s >= self.umbral)}
        if self.exponer_puntuaciones:
            r.update(similitud=s, umbral=self.umbral)
        return r

    def _identificar_template(self, template: np.ndarray) -> dict:
        self._exigir_integridad()
        if not self._bd:
            return {"id": None, "aceptado": False, "similitud": None, "motivo": "BD vacía"}
        ids = list(self._bd)
        s = self._similitudes_rep(self.extractor.representar(template), ids)
        orden = np.argsort(-s)
        mejor = int(orden[0])
        aceptado = bool(s[mejor] >= self.umbral)
        r = {"id": ids[mejor] if aceptado else None, "aceptado": aceptado}
        if self.exponer_puntuaciones:
            r.update(similitud=float(s[mejor]), umbral=self.umbral,
                     ranking=[(ids[k], float(s[k])) for k in orden[:5]])
        return r

    # ============================================================= API (9 métodos)
    # ---- pipeline completo -------------------------------------------------
    def inscribir(self, id: str, foto, sobrescribir: bool = False) -> dict:
        """Inscribe a una persona a partir de su nombre y una foto."""
        return self._inscribir_template(
            id, self._template_desde_foto(foto, inscribiendo=True), "inscribir", sobrescribir)

    def verificar(self, id: str, foto) -> dict:
        """Verifica (1:1) que la foto corresponde a la persona 'id'."""
        return self._verificar_template(id, self._template_desde_foto(foto))

    def identificar(self, foto) -> dict:
        """Identifica (1:N) a quién corresponde la foto."""
        return self._identificar_template(self._template_desde_foto(foto))

    # ---- sin detector (_sd) -------------------------------------------------
    def inscribir_sd(self, id: str, foto, sobrescribir: bool = False) -> dict:
        """Inscribe a partir de una cara ya recortada (omite el detector)."""
        return self._inscribir_template(
            id, self._template_desde_cara(foto, inscribiendo=True), "inscribir_sd", sobrescribir)

    def verificar_sd(self, id: str, foto) -> dict:
        """Verifica a partir de una cara ya recortada (omite el detector)."""
        return self._verificar_template(id, self._template_desde_cara(foto))

    def identificar_sd(self, foto) -> dict:
        """Identifica a partir de una cara ya recortada (omite el detector)."""
        return self._identificar_template(self._template_desde_cara(foto))

    # ---- solo template (_t) -------------------------------------------------
    def inscribir_t(self, id: str, template, sobrescribir: bool = False) -> dict:
        """Inscribe a partir de un template (omite detector y extractor)."""
        return self._inscribir_template(
            id, self._template_externo(template), "inscribir_t", sobrescribir)

    def verificar_t(self, id: str, template) -> dict:
        """Verifica a partir de un template (omite detector y extractor)."""
        return self._verificar_template(id, self._template_externo(template))

    def identificar_t(self, template) -> dict:
        """Identifica a partir de un template (omite detector y extractor)."""
        return self._identificar_template(self._template_externo(template))

    # ---- utilidades ----------------------------------------------------------
    def extraer_template(self, foto) -> np.ndarray:
        """foto -> template (para usar con los métodos _t)."""
        return self._template_desde_foto(foto)

    def extraer_cara(self, foto) -> np.ndarray:
        """foto -> cara 112x112 alineada (para usar con los métodos _sd)."""
        return self.detector.extraer_cara(foto)


def _main():
    import argparse
    p = argparse.ArgumentParser(description="Sistema de reconocimiento facial EBGM")
    p.add_argument("accion", choices=["inscribir", "verificar", "identificar",
                                      "listar", "eliminar"])
    p.add_argument("--id")
    p.add_argument("--foto")
    p.add_argument("--bd", default=RUTA_BD_POR_DEFECTO)
    p.add_argument("--sobrescribir", action="store_true")
    p.add_argument("--puntuaciones", action="store_true",
                   help="muestra similitudes y ranking (modo evaluación)")
    a = p.parse_args()

    sis = SistemaReconocimiento(ruta_bd=a.bd, exponer_puntuaciones=a.puntuaciones)
    try:
        if a.accion == "listar":
            print(f"{len(sis.personas())} personas:", ", ".join(sis.personas()))
            if not sis.integridad_global:
                print("AVISO: el MAC global de la BD no cuadra (filas borradas/alteradas)")
            if sis.filas_rechazadas:
                print(f"AVISO: {len(sis.filas_rechazadas)} filas rechazadas:",
                      sis.filas_rechazadas[:5])
        elif a.accion == "inscribir":
            print(sis.inscribir(a.id, a.foto, sobrescribir=a.sobrescribir))
        elif a.accion == "verificar":
            print(sis.verificar(a.id, a.foto))
        elif a.accion == "identificar":
            print(sis.identificar(a.foto))
        elif a.accion == "eliminar":
            print("eliminado" if sis.eliminar(a.id) else "no existía")
    except (ErrorDeteccion, ErrorSistema) as e:
        print("Error:", e)
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
