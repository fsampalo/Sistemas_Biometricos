# Reconocimiento facial EBGM — Sistemas Biométricos

Práctica de la asignatura. Sistema de reconocimiento facial completo cuyo
**método de reconocimiento es EBGM (Elastic Bunch Graph Matching)**: grafo de
nodos sobre la cara + *jets* de Gabor en cada nodo + emparejamiento de grafos.

Nuestro famoso elegido es **El Cigala**.

```
foto -> [DETECTOR YuNet] -> cara 112x112 alineada -> [EXTRACTOR EBGM] -> template -> [MATCHER] -> BD (.csv)
```

| Etapa | Fichero | Qué hace |
|---|---|---|
| Detector | [`detector.py`](detector.py) | YuNet (OpenCV). Detecta la cara y la **alinea** a 112×112 con 5 landmarks. |
| Extractor | [`ebgm.py`](ebgm.py) | **EBGM**: rejilla de nodos + jets de Gabor (magnitud). El template. |
| Matcher + BD | [`sistema.py`](sistema.py) | Los 9 métodos del enunciado + base de datos en `.csv`. |

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_models.py        # descarga YuNet en models/
```

El `Dataset/` de famosos (lo da el docente) debe estar en la raíz del repo
(`Dataset/adele_2016.jpg`, …). Se extrae de `Dataset.tar.gz`.

## Puesta en marcha

```bash
python inscribir_dataset.py      # inscribe 49 famosos + el cigala y calibra el fondo
python demo.py                   # demuestra los 9 métodos funcionando
```

> El comprimido del docente trae **49** famosos; con el cigala la BD tiene **50**
> identidades. (El enunciado menciona 50 + 1 = 51.)

## El método: EBGM

1. **Grafo.** Sobre la cara alineada se coloca una rejilla de nodos (9×8 = 72).
   Como las caras llegan alineadas, las posiciones de los nodos son fijas.
2. **Jets de Gabor.** En cada nodo se calcula la respuesta de un banco de filtros
   de Gabor (5 escalas × 8 orientaciones = 40). Se toma la **magnitud** → un
   vector de 40 por nodo. El conjunto de 72 jets es el **template**.
3. **Normalización por fondo.** Se resta la "cara media" (jet medio de los 49
   famosos, `models/ebgm_fondo.npy`) para resaltar lo propio de cada identidad.
4. **Matcher.** La similitud entre dos grafos es la media, nodo a nodo, del
   **coseno** de sus jets. Con `elastico=True` cada nodo puede desplazarse a
   nodos vecinos del otro grafo (deformación elástica, la "E" de EBGM).

**Inscripción del cigala por centroide.** Nuestro famoso se inscribe como el
**centroide** (media en espacio normalizado) de varias fotos de referencia, no
con una sola. Así la plantilla es robusta a pose/edad **sin dejar de ser un único
template por persona** (igual de justo que los famosos). Esto sube el rank-1 en
fotos nuevas de ~0 % a ~80 % sin aumentar los falsos positivos (ver Evaluación).

## Los 9 métodos

Tres operaciones × tres variantes según en qué punto del pipeline se entra:

| Método | Entra por | Omite |
|---|---|---|
| `inscribir(id, foto)` / `verificar(id, foto)` / `identificar(foto)` | foto | — (pipeline completo) |
| `inscribir_sd` / `verificar_sd` / `identificar_sd` | cara ya recortada | **detector** |
| `inscribir_t` / `verificar_t` / `identificar_t` | template | **detector y extractor** |

```python
from sistema import SistemaReconocimiento
sis = SistemaReconocimiento()

sis.inscribir("ana", "ana.jpg")          # alta por foto
sis.verificar("ana", "otra_ana.jpg")     # 1:1 -> {'verificado': bool, 'similitud': ...}
sis.identificar("quien.jpg")             # 1:N -> {'id': ..., 'ranking': [...]}
```

## Base de datos (`.csv`)

Una fila por persona: `id,template`, con el template (jets EBGM crudos) en
**base64**. La escribe/lee `sistema.py`; filas corruptas se ignoran al cargar.

## Evaluación

```bash
python evaluar.py                 # distribución, FAR/FRR, EER, rank-1
```

Protocolo **honesto**: el cigala se inscribe con sus fotos de referencia
(`fotos_cigala/`) y se evalúa con fotos **nuevas que el sistema nunca vio**
(`fotos_cigala_test/`, descargadas de Wikimedia Commons: 2014, 2024, San Isidro).
Impostores: cada famoso contra los demás y contra el cigala.

| Métrica | 1 sola foto | **Centroide (desplegado)** |
|---|---|---|
| Rank-1 en fotos nuevas (galería 50) | 0/5 | **4/5 (80 %)** |
| d-prime (separación genuino/impostor) | 0.8 | **3.3** |
| EER (verificación 1:1) | ~40 % | **~1–2 %** (n pequeño) |
| FAR zero-effort (famosos que pasan como cigala) | 18/49 | **0/49** |

**Lectura honesta.** Con **una sola foto** inscrita, EBGM clásico apenas
generaliza a fotos nuevas no controladas. Inscribiendo al cigala como
**centroide** de varias fotos la cosa cambia mucho (rank-1 0→80 %) y de forma
**justa** (sigue siendo 1 template/persona, sin inflar nada). Aun así quedan
límites del método: 2 de 7 fotos nuevas ni detectan cara (encuadre muy apaisado),
y un perfil se confunde con otro famoso. El EER ~1–2 % es optimista por tener
pocas fotos genuinas, pero la separación (d-prime 3.3) es real y visible en el
histograma. Durante el desarrollo se probaron otras variantes de EBGM (fase con
desplazamiento de Wiskott, blanqueo PCA, *bunch* con `max`); el *bunch* con `max`
se **descartó** a propósito: inflaba las puntuaciones y convertía al cigala en un
"imán" de falsos positivos (12/49) que el rival explotaría.

## Seguridad (análisis de ataques)

```bash
python evaluar_ataques.py
```

Cubre los supuestos del enunciado, con demostración y mitigación de cada uno:

- **(A) Saltarse parte del sistema** — `_sd`/`_t` permiten inyectar entradas sin
  pasar por el detector/extractor (ruido que el detector habría rechazado;
  *replay* de un template = suplantación total con similitud 1.0).
- **(B) Leer la BD** — templates en claro: exfiltración de rasgos biométricos y
  reutilización por *replay*. (Las magnitudes de Gabor no permiten reconstruir la
  foto, pero sí reinyectar el template.)
- **(C) Escribir en la BD** — sustitución de identidad (el atacante se vuelve
  "cigala"), puerta trasera (alta de identidad propia) y sabotaje/DoS.

Mitigaciones propuestas: no exponer `_sd`/`_t`, *liveness*/antispoofing, cifrado y
control de acceso a la BD, protección de plantilla (biohashing/fuzzy vault),
integridad por fila (HMAC) y auditoría.

## Docker

```bash
docker build -t ebgm .
docker run --rm ebgm                      # demo de los 9 métodos
docker run --rm ebgm python evaluar.py
docker run --rm ebgm python evaluar_ataques.py
```

El `build` instala dependencias, descarga el modelo e inscribe la BD, de modo
que el contenedor arranca listo para usar.

## Estructura

```
├── detector.py            # Etapa 1: detección + alineado (YuNet)
├── ebgm.py                # Etapa 2: extractor EBGM (jets de Gabor) + matcher
├── sistema.py             # Etapa 3: los 9 métodos + BD .csv
├── inscribir_dataset.py   # Inscribe famosos + cigala (centroide) y calibra el fondo
├── demo.py                # Demostración de los 9 métodos
├── evaluar.py             # Métricas (EER, rank-1, DET)
├── evaluar_ataques.py     # Análisis de seguridad (ataques del enunciado)
├── download_models.py     # Descarga YuNet
├── requirements.txt
├── Dockerfile
├── fotos_cigala/          # Fotos de referencia del cigala (inscripción)
├── fotos_cigala_test/     # Fotos NUEVAS del cigala (evaluación, no inscritas)
├── Dataset/               # Famosos del docente (no versionado)
├── models/                # YuNet + ebgm_fondo.npy (generado)
└── salida/                # Recortes/figuras (generado)
```

## Detector (detalle)

`detector.py` usa **YuNet** (`cv2.FaceDetectorYN`). Por cara devuelve la caja, la
confianza y 5 landmarks (ojos, nariz, comisuras), con los que **alinea** la cara
(transformación de similitud a la plantilla de ArcFace) a un recorte de 112×112.
Valida la entrada (rechaza imágenes corruptas o < 32 px, reescala > 4096 px,
descarta caras < 40 px). Prueba rápida:

```bash
python detector.py fotos_cigala/*.jpg     # guarda recortes en salida/
```

Parámetros de `DetectorCaras`: `umbral_confianza=0.8`, `tam_min_cara=40`,
`tam_salida=112`, `politica_multiples="mayor"` (o `"error"` al inscribir).

## Notas

- En Windows, OpenCV no abre rutas con tildes/ñ: el modelo y las imágenes se leen
  vía buffers de NumPy, así funciona en cualquier carpeta.
- Modelo detector: [YuNet 2023mar](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
