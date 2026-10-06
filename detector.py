"""
Módulo de detección facial — Práctica Sistemas Biométricos.

Primera etapa del pipeline:  imagen -> [DETECTOR] -> cara recortada/alineada -> extractor -> template

Detector: YuNet, red neuronal ligera de OpenCV (cv2.FaceDetectorYN).
Devuelve caja + 5 landmarks (ojos, nariz, comisuras) -> permite alinear.

Uso rápido:
    from detector import DetectorCaras
    det = DetectorCaras()
    cara = det.extraer_cara("foto.jpg")   # np.ndarray 112x112 BGR alineada, o lanza excepción
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

RUTA_MODELO_YUNET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "face_detection_yunet_2023mar.onnx",
)

# Posiciones de referencia de los 5 landmarks para una cara de 112x112
# (plantilla estándar de ArcFace/InsightFace). Orden: ojo dcho, ojo izq, nariz, boca dcha, boca izq
PLANTILLA_LANDMARKS_112 = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]], dtype=np.float32)

# Límites de entrada (el sistema puede ser atacado: no aceptar cualquier cosa)
TAM_MAX_LADO = 4096
TAM_MIN_LADO = 32


def guardar_imagen(ruta: str, img: np.ndarray) -> None:
    """cv2.imwrite no admite rutas con tildes en Windows: codificar y escribir con numpy."""
    ok, buf = cv2.imencode(os.path.splitext(ruta)[1] or ".jpg", img)
    if not ok:
        raise ErrorDeteccion(f"No se pudo codificar la imagen para {ruta}")
    buf.tofile(ruta)


class ErrorDeteccion(Exception):
    """Error controlado del detector (imagen inválida, sin cara, varias caras...)."""


@dataclass
class Deteccion:
    caja: tuple[int, int, int, int]          # x, y, ancho, alto
    confianza: float
    landmarks: np.ndarray = field(repr=False)  # (5, 2)

    @property
    def area(self) -> int:
        return self.caja[2] * self.caja[3]


class DetectorCaras:
    def __init__(self, umbral_confianza: float = 0.8,
                 tam_min_cara: int = 40, tam_salida: int = 112,
                 politica_multiples: str = "mayor"):
        """
        umbral_confianza:   score mínimo para aceptar una detección
        tam_min_cara:       lado mínimo en píxeles de una cara válida
        tam_salida:         tamaño del recorte devuelto (cuadrado)
        politica_multiples: qué hacer si hay varias caras en la foto
                            "mayor"  -> se queda con la más grande
                            "error"  -> lanza ErrorDeteccion (más estricto, recomendable en inscribir)
        """
        if politica_multiples not in ("mayor", "error"):
            raise ValueError("politica_multiples debe ser 'mayor' o 'error'")
        self.umbral = umbral_confianza
        self.tam_min = tam_min_cara
        self.tam_salida = tam_salida
        self.politica = politica_multiples

        if not os.path.isfile(RUTA_MODELO_YUNET):
            raise FileNotFoundError(
                f"No se encuentra el modelo YuNet en {RUTA_MODELO_YUNET}. "
                "Ejecuta download_models.py")
        # Cargar desde buffer: OpenCV no abre rutas con tildes/ñ en Windows
        modelo = np.fromfile(RUTA_MODELO_YUNET, np.uint8)
        self._yunet = cv2.FaceDetectorYN.create(
            "onnx", modelo, np.empty(0, np.uint8), (320, 320),
            score_threshold=umbral_confianza, nms_threshold=0.3, top_k=50)

    # ------------------------------------------------------------------ carga
    @staticmethod
    def cargar_imagen(foto) -> np.ndarray:
        """Acepta ruta (str/Path), bytes o np.ndarray. Devuelve imagen BGR uint8 validada."""
        if isinstance(foto, np.ndarray):
            img = foto
        elif isinstance(foto, (bytes, bytearray)):
            img = cv2.imdecode(np.frombuffer(foto, np.uint8), cv2.IMREAD_COLOR)
        else:
            ruta = os.fspath(foto)
            if not os.path.isfile(ruta):
                raise ErrorDeteccion(f"No existe el fichero: {ruta}")
            # imdecode en vez de imread: funciona con rutas con tildes en Windows
            img = cv2.imdecode(np.fromfile(ruta, np.uint8), cv2.IMREAD_COLOR)

        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            raise ErrorDeteccion("La imagen no se pudo decodificar")
        if img.dtype != np.uint8:
            raise ErrorDeteccion(f"Tipo de imagen no soportado: {img.dtype}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif not (img.ndim == 3 and img.shape[2] == 3):
            raise ErrorDeteccion(f"Forma de imagen no soportada: {img.shape}")

        h, w = img.shape[:2]
        if min(h, w) < TAM_MIN_LADO:
            raise ErrorDeteccion(f"Imagen demasiado pequeña ({w}x{h})")
        if max(h, w) > TAM_MAX_LADO:  # reescalar en vez de rechazar
            f = TAM_MAX_LADO / max(h, w)
            img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        return img

    # ------------------------------------------------------------- detección
    def detectar(self, foto) -> list[Deteccion]:
        """Devuelve TODAS las caras válidas, ordenadas de mayor a menor."""
        img = self.cargar_imagen(foto)
        dets = [d for d in self._detectar_yunet(img) if min(d.caja[2], d.caja[3]) >= self.tam_min]
        return sorted(dets, key=lambda d: d.area, reverse=True)

    def _detectar_yunet(self, img) -> list[Deteccion]:
        h, w = img.shape[:2]
        self._yunet.setInputSize((w, h))
        _, caras = self._yunet.detect(img)
        if caras is None:
            return []
        out = []
        for c in caras:
            # c = [x, y, w, h, 5 landmarks (x,y)*5, score]
            x, y, bw, bh = c[:4]
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
            if x1 <= x0 or y1 <= y0:
                continue
            out.append(Deteccion((x0, y0, x1 - x0, y1 - y0), float(c[14]),
                                 c[4:14].reshape(5, 2).astype(np.float32)))
        return out

    def detectar_principal(self, foto) -> tuple[np.ndarray, Deteccion]:
        """Devuelve (imagen, detección de la cara principal) aplicando la política de múltiples caras."""
        img = self.cargar_imagen(foto)
        dets = self.detectar(img)
        if not dets:
            raise ErrorDeteccion("No se ha detectado ninguna cara")
        if len(dets) > 1 and self.politica == "error":
            raise ErrorDeteccion(f"Se han detectado {len(dets)} caras; se esperaba una")
        return img, dets[0]

    # -------------------------------------------------------------- recorte
    def recortar(self, img: np.ndarray, det: Deteccion, alinear: bool = True) -> np.ndarray:
        """Recorte cuadrado de tam_salida x tam_salida. Con alinear=True aplica una
        transformación de similitud para que ojos/nariz/boca queden en posiciones fijas."""
        n = self.tam_salida
        if alinear:
            ref = PLANTILLA_LANDMARKS_112 * (n / 112.0)
            M, _ = cv2.estimateAffinePartial2D(det.landmarks, ref, method=cv2.LMEDS)
            if M is not None:
                return cv2.warpAffine(img, M, (n, n), borderMode=cv2.BORDER_REPLICATE)
        # Sin alineado: caja ampliada a cuadrado con un pequeño margen
        x, y, w, h = det.caja
        lado = int(max(w, h) * 1.2)
        cx, cy = x + w // 2, y + h // 2
        x0, y0 = cx - lado // 2, cy - lado // 2
        H, W = img.shape[:2]
        pad = max(0, -x0, -y0, x0 + lado - W, y0 + lado - H)
        if pad:
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            x0, y0 = x0 + pad, y0 + pad
        return cv2.resize(img[y0:y0 + lado, x0:x0 + lado], (n, n), interpolation=cv2.INTER_AREA)

    def extraer_cara(self, foto, alinear: bool = True) -> np.ndarray:
        """Punto de entrada para inscribir/verificar/identificar: foto -> cara normalizada."""
        img, det = self.detectar_principal(foto)
        return self.recortar(img, det, alinear)

    # ------------------------------------------------------------- utilidad
    def dibujar(self, foto, ruta_salida: str | None = None) -> np.ndarray:
        img = self.cargar_imagen(foto).copy()
        for i, d in enumerate(self.detectar(img)):
            x, y, w, h = d.caja
            color = (0, 200, 0) if i == 0 else (0, 165, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{d.confianza:.2f}", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            for px, py in d.landmarks:
                cv2.circle(img, (int(px), int(py)), 2, (0, 0, 255), -1)
        if ruta_salida:
            guardar_imagen(ruta_salida, img)
        return img


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Prueba del detector de caras")
    p.add_argument("imagenes", nargs="+")
    p.add_argument("--salida", default="salida")
    a = p.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    det = DetectorCaras()
    for ruta in a.imagenes:
        base = os.path.splitext(os.path.basename(ruta))[0]
        try:
            dets = det.detectar(ruta)
            print(f"{ruta}: {len(dets)} cara(s) -> " +
                  ", ".join(f"{d.caja} conf={d.confianza:.2f}" for d in dets))
            det.dibujar(ruta, os.path.join(a.salida, f"{base}_det.jpg"))
            guardar_imagen(os.path.join(a.salida, f"{base}_cara.jpg"), det.extraer_cara(ruta))
        except ErrorDeteccion as e:
            print(f"{ruta}: ERROR -> {e}")
