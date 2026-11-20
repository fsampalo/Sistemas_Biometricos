"""Normalizacion geometrica y fotometrica de la cara a una imagen canonica."""
import cv2
import numpy as np

TAM = 144                       # lado de la imagen canonica
OJO_A = np.array([50.0, 54.0])  # ojo que aparece a la izquierda de la imagen
OJO_B = np.array([94.0, 54.0])  # ojo que aparece a la derecha de la imagen
MAX_LADO = 2000                 # limite de tamano de entrada (evita consumo de memoria)


def cargar_imagen(foto):
    """Acepta ruta o array (BGR o gris). Devuelve BGR uint8 limitado en tamano."""
    if isinstance(foto, (str, bytes)) or hasattr(foto, "__fspath__"):
        img = cv2.imread(str(foto), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"No se puede leer la imagen: {foto}")
    else:
        img = np.asarray(foto)
        if img.ndim == 2:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGRA2BGR)
        elif img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Formato de imagen no soportado")
        img = img.astype(np.uint8)
    if img.shape[0] < 16 or img.shape[1] < 16:
        raise ValueError("Imagen demasiado pequena")
    lado = max(img.shape[:2])
    if lado > MAX_LADO:
        s = MAX_LADO / lado
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def _ojos_desde_caja(caja):
    """Estimacion de los ojos a partir de la caja si el detector no da puntos
    (relaciones medias medidas con YuNet sobre el dataset)."""
    x, y, w, h = caja
    cy = y + 0.40 * h
    return (x + 0.32 * w, cy), (x + 0.68 * w, cy)


def alinear(img_bgr, deteccion, devolver_matriz=False):
    """Transformacion de semejanza que lleva los ojos a su posicion canonica."""
    ojos = deteccion.get("ojos") or _ojos_desde_caja(deteccion["caja"])
    a, b = np.array(ojos[0], float), np.array(ojos[1], float)
    if a[0] > b[0]:
        a, b = b, a
    src = np.stack([a, b])
    dst = np.stack([OJO_A, OJO_B])
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    if M is None:  # dos puntos -> calculo directo
        v1, v2 = b - a, OJO_B - OJO_A
        s = np.linalg.norm(v2) / (np.linalg.norm(v1) + 1e-9)
        ang = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        R = s * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        t = OJO_A - R @ a
        M = np.hstack([R, t[:, None]])
    can = cv2.warpAffine(img_bgr, M, (TAM, TAM), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return (can, M) if devolver_matriz else can


def a_original(M, pts):
    """Lleva puntos del marco canonico a la imagen original (inversa de M)."""
    Mi = cv2.invertAffineTransform(M)
    pts = np.asarray(pts, float)
    return pts @ Mi[:, :2].T + Mi[:, 2]


def sin_detector(img_bgr):
    """Ruta *_sd: se asume que la imagen ya es (aprox.) un recorte de la cara;
    solo se reescala al tamano canonico. La busqueda global de EBGM corrige
    desplazamiento y escala."""
    return cv2.resize(img_bgr, (TAM, TAM), interpolation=cv2.INTER_AREA)


def fotometria(img_bgr):
    """Gris + estandarizacion (media 0, desv. 1) ponderada por una ventana
    eliptica centrada en la cara (es una operacion lineal + escala)."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) if img_bgr.ndim == 3 \
        else img_bgr.astype(np.float64)
    m = (g * PESO).sum() / PESO.sum()
    s = np.sqrt((((g - m) ** 2) * PESO).sum() / PESO.sum()) + 1e-6
    return (g - m) / s


def _peso_eliptico():
    yy, xx = np.mgrid[0:TAM, 0:TAM]
    cx, cy = TAM / 2, TAM * 0.55
    r = ((xx - cx) / (TAM * 0.42)) ** 2 + ((yy - cy) / (TAM * 0.5)) ** 2
    return np.clip(1.2 - r, 0, 1)


PESO = _peso_eliptico()
