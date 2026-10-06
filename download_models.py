"""Descarga los modelos necesarios en ./models (YuNet de OpenCV Zoo)."""

import hashlib
import os
import urllib.request

DIR_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELOS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
}


def descargar(nombre: str, url: str) -> None:
    destino = os.path.join(DIR_MODELOS, nombre)
    if os.path.isfile(destino) and os.path.getsize(destino) > 0:
        print(f"[ok] {nombre} ya existe")
        return
    print(f"[..] Descargando {nombre}")
    tmp = destino + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, destino)
    with open(destino, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"[ok] {nombre} ({os.path.getsize(destino)} bytes, sha256={sha[:16]}...)")


if __name__ == "__main__":
    os.makedirs(DIR_MODELOS, exist_ok=True)
    for nombre, url in MODELOS.items():
        descargar(nombre, url)
