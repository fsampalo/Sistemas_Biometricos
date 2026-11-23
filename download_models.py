"""Descarga los modelos necesarios en ./models (YuNet de OpenCV Zoo).

El fichero se verifica contra un SHA-256 fijado: si alguien sustituye el
detector (en el repositorio remoto o en el disco), no se usa.
"""

import hashlib
import os
import sys
import urllib.request

DIR_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELOS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
}


def sha256(ruta: str) -> str:
    with open(ruta, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def descargar(nombre: str, url: str, sha_esperado: str) -> None:
    destino = os.path.join(DIR_MODELOS, nombre)
    if os.path.isfile(destino) and sha256(destino) == sha_esperado:
        print(f"[ok] {nombre} ya existe (sha256 verificado)")
        return
    print(f"[..] Descargando {nombre}")
    tmp = destino + ".part"
    urllib.request.urlretrieve(url, tmp)
    sha = sha256(tmp)
    if sha != sha_esperado:
        os.remove(tmp)
        sys.exit(f"[ERROR] {nombre}: sha256 {sha} no coincide con el esperado")
    os.replace(tmp, destino)
    print(f"[ok] {nombre} ({os.path.getsize(destino)} bytes, sha256 verificado)")


if __name__ == "__main__":
    os.makedirs(DIR_MODELOS, exist_ok=True)
    for nombre, (url, sha) in MODELOS.items():
        descargar(nombre, url, sha)
