# Reconocimiento facial — Sistemas Biométricos

Práctica de la asignatura Sistemas Biométricos. Este repositorio contiene la primera etapa del pipeline de reconocimiento facial: el **detector de caras**.

```
imagen -> [DETECTOR] -> cara recortada y alineada (112x112) -> extractor -> template
```

## Detector

`detector.py` usa **YuNet** (`cv2.FaceDetectorYN`), una red neuronal ligera de OpenCV. Por cada cara devuelve:

- la caja (`x, y, ancho, alto`) y la confianza de la detección;
- 5 landmarks: ojo derecho, ojo izquierdo, nariz, comisura derecha y comisura izquierda de la boca.

Con los landmarks, la cara se **alinea** mediante una transformación de similitud con la plantilla estándar de ArcFace/InsightFace. Así, ojos, nariz y boca quedan siempre en las mismas posiciones del recorte de 112x112.

Además, el detector valida la entrada: rechaza imágenes corruptas o demasiado pequeñas (< 32 px), reescala las demasiado grandes (> 4096 px) y descarta las caras con un lado menor de 40 px.

## Instalación

Requiere Python 3.9 o superior.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python download_models.py           # descarga YuNet en models/
```

Si PowerShell no deja activar el entorno: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Prueba rápida

```powershell
python detector.py (Get-ChildItem fotos_cigala\*.jpg).FullName
```

En Linux/macOS: `python detector.py fotos_cigala/*.jpg`.

Por cada imagen, se muestran en consola las caras detectadas y se guardan en `salida/`:

| Fichero | Contenido |
|---|---|
| `N_det.jpg` | Imagen original con la caja, la confianza y los landmarks dibujados |
| `N_cara.jpg` | Recorte de 112x112 alineado (la entrada del extractor) |

Con la opción `--salida <carpeta>` se cambia la carpeta de destino.

## Uso desde código

```python
from detector import DetectorCaras, ErrorDeteccion

det = DetectorCaras()                     # umbral_confianza=0.8, tam_salida=112
try:
    cara = det.extraer_cara("foto.jpg")   # np.ndarray (112, 112, 3) BGR
except ErrorDeteccion as e:
    print("No válida:", e)
```

`extraer_cara` acepta una ruta, `bytes` o un `np.ndarray`.

Parámetros de `DetectorCaras`:

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `umbral_confianza` | `0.8` | Confianza mínima para aceptar una cara |
| `tam_min_cara` | `40` | Lado mínimo de la cara, en píxeles |
| `tam_salida` | `112` | Tamaño del recorte cuadrado |
| `politica_multiples` | `"mayor"` | Si hay varias caras: `"mayor"` se queda con la más grande; `"error"` lanza `ErrorDeteccion` (recomendable al inscribir) |

Otros métodos útiles:

- `detectar(foto)`: lista de todas las caras, de mayor a menor.
- `recortar(img, det, alinear=False)`: recorte sin alinear.
- `dibujar(foto, ruta_salida)`: imagen con las detecciones dibujadas.

## Estructura

```
├── detector.py          # Detector de caras (YuNet) + recorte/alineado
├── download_models.py   # Descarga los modelos en models/
├── requirements.txt
├── fotos_cigala/        # Imágenes de prueba
├── models/              # (generado, no versionado)
└── salida/              # (generado, no versionado)
```

## Notas

- En Windows, OpenCV no puede abrir rutas con tildes o "ñ". Por eso el modelo y las imágenes se leen y escriben a través de buffers de NumPy: funciona en cualquier carpeta.
- Modelo: [YuNet 2023mar](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), de OpenCV Zoo.
