"""Detector de caras intercambiable.

Por defecto se usa YuNet (OpenCV), que devuelve la caja y 5 puntos (ojos, nariz,
comisuras). Si no esta disponible se recurre a Haar. Para usar el detector del
grupo basta con pasar al SistemaEBGM un objeto con el metodo
    detectar(img_bgr) -> lista de dicts {"caja": (x, y, w, h), "score": float,
                                          "ojos": ((xd, yd), (xi, yi)) | None}
"""
import os
import cv2
import numpy as np

DIR_MODELOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelos")


class DetectorYuNet:
    def __init__(self, umbral: float = 0.6, ruta: str | None = None):
        ruta = ruta or os.path.join(DIR_MODELOS, "yunet.onnx")
        self.det = cv2.FaceDetectorYN.create(ruta, "", (320, 320), umbral, 0.3, 5000)

    def detectar(self, img):
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, caras = self.det.detect(img)
        res = []
        if caras is None:
            return res
        for f in caras:
            x, y, cw, ch = f[:4]
            # YuNet: f[4:6] ojo derecho (lado izquierdo de la imagen), f[6:8] ojo izquierdo
            res.append({"caja": (float(x), float(y), float(cw), float(ch)),
                        "score": float(f[-1]),
                        "ojos": ((float(f[4]), float(f[5])), (float(f[6]), float(f[7])))})
        return res


class DetectorHaar:
    def __init__(self, ruta: str | None = None):
        ruta = ruta or os.path.join(DIR_MODELOS, "haarcascade_frontalface_default.xml")
        self.det = cv2.CascadeClassifier(ruta)

    def detectar(self, img):
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        cajas = self.det.detectMultiScale(gris, 1.1, 5, minSize=(40, 40))
        return [{"caja": tuple(map(float, c)), "score": 1.0, "ojos": None} for c in cajas]


def detector_por_defecto():
    try:
        return DetectorYuNet()
    except Exception:  # pragma: no cover
        return DetectorHaar()


def elegir_cara(detecciones):
    """Politica de seleccion cuando hay varias caras: la de mayor area.
    (Elegir por confianza falla en adam_pearson_2024.jpg, donde una cara del
    fondo tiene mas score que la persona protagonista).
    Para no elegir falsos positivos grandes de baja confianza, primero se
    descartan las detecciones con score < max(0.75, mejor_score - 0.15)."""
    if not detecciones:
        return None
    mejor = max(d["score"] for d in detecciones)
    cand = [d for d in detecciones if d["score"] >= min(mejor, max(0.75, mejor - 0.15))]
    return max(cand, key=lambda d: d["caja"][2] * d["caja"][3])
