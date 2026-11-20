"""Transformaciones para generar muestras de prueba a partir de la unica foto
disponible por persona. Todas operan sobre la imagen completa (antes del detector)."""
import cv2
import numpy as np


def _ojos(det, img):
    from ebgm.detector import elegir_cara
    d = elegir_cara(det.detectar(img))
    return d


def brillo(img, det=None):
    return np.clip(img.astype(int) + 45, 0, 255).astype(np.uint8)


def oscuro(img, det=None):
    return np.clip(img.astype(float) * 0.45, 0, 255).astype(np.uint8)


def contraste(img, det=None):
    m = img.mean()
    return np.clip((img.astype(float) - m) * 0.55 + m, 0, 255).astype(np.uint8)


def gamma(img, det=None):
    return (255 * (img / 255.0) ** 1.8).astype(np.uint8)


def desenfoque(img, det=None):
    s = max(img.shape[:2]) / 400
    return cv2.GaussianBlur(img, (0, 0), 1.5 * s)


def ruido(img, det=None):
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, 14, img.shape), 0, 255).astype(np.uint8)


def jpeg(img, det=None):
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 12])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def baja_resolucion(img, det=None):
    d = _ojos(det, img)
    s = 40.0 / max(d["caja"][2], 1)    # cara de ~40 px de ancho
    peq = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.resize(peq, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def rotacion(img, det=None, ang=12):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def perspectiva(img, det=None):
    """Simula un ligero giro de cabeza / foto tomada de lado."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0.06 * h], [w, 0], [w, h], [0, 0.94 * h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderMode=cv2.BORDER_REFLECT)


def espejo(img, det=None):
    return cv2.flip(img, 1)


def gris(img, det=None):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def gafas_sol(img, det):
    d = _ojos(det, img)
    out = img.copy()
    (x1, y1), (x2, y2) = d["ojos"]
    iod = np.hypot(x2 - x1, y2 - y1)
    for (x, y) in d["ojos"]:
        cv2.ellipse(out, (int(x), int(y)), (int(0.42 * iod), int(0.28 * iod)), 0, 0, 360, (15, 15, 15), -1)
    cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (15, 15, 15), max(2, int(0.06 * iod)))
    return out


def mascarilla(img, det):
    d = _ojos(det, img)
    x, y, w, h = map(int, d["caja"])
    out = img.copy()
    (x1, y1), (x2, y2) = d["ojos"]
    top = int(max(y1, y2) + 0.45 * np.hypot(x2 - x1, y2 - y1))
    cv2.rectangle(out, (x, top), (x + w, y + h + h // 10), (235, 235, 235), -1)
    return out


AUMENTOS = {
    "original": lambda img, det=None: img.copy(),
    "brillo": brillo, "oscuro": oscuro, "contraste": contraste, "gamma": gamma,
    "escala_grises": gris, "desenfoque": desenfoque, "ruido": ruido, "jpeg_q12": jpeg,
    "baja_resolucion": baja_resolucion, "rotacion_12": rotacion, "perspectiva": perspectiva,
    "espejo": espejo, "gafas_sol": gafas_sol, "mascarilla": mascarilla,
}
