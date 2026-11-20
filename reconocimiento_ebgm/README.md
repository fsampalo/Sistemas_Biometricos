# Sistema de reconocimiento facial — EBGM

Reconocimiento facial mediante **Elastic Bunch Graph Matching** (Wiskott et al., 1997)
para la práctica de Sistemas Biométricos. El sistema inscribe a 51 personas (50 fotos
comunes del docente + **Diego Ramón Jiménez Salazar «El Cigala»**, elección del grupo)
en una base de datos `.csv` y ofrece los nueve métodos que exige el guion.

## Puesta en marcha

### Con Docker (recomendado)
```bash
docker build -t ebgm .
docker run --rm ebgm listar                                   # lista las personas inscritas
docker run --rm -v "$PWD/foto.jpg:/app/foto.jpg" ebgm identificar --foto foto.jpg
```

### Sin Docker
```bash
pip install -r requirements.txt
python cli.py identificar --foto datos/dataset/keanu_reeves_2021.png
python cli.py verificar --id keanu_reeves --foto foto.jpg
python cli.py verificar --id keanu_reeves --foto recorte.jpg --sd   # omite el detector
```

## API (clase `ebgm.sistema.SistemaEBGM`)

| Método | Descripción |
|---|---|
| `inscribir(id, foto)` / `verificar(id, foto)` / `identificar(foto)` | Flujo completo (detector + extractor + comparador). |
| `inscribir_sd` / `verificar_sd` / `identificar_sd` | Omiten el detector; la foto debe ser un **recorte** de la cara. |
| `inscribir_t` / `verificar_t` / `identificar_t` | Trabajan directamente con un **template** (omiten detector y extractor). |

`verificar*` devuelve `(aceptado, similitud)`; `identificar*` devuelve `(id | None, similitud)`.

## Cómo está construido

```
foto ─▶ detector (YuNet) ─▶ normalización (alineado por ojos, 144×144, z-score)
     ─▶ 40 filtros de Gabor (5 frec × 8 orient) ─▶ EBGM (ajuste elástico del grafo)
     ─▶ template = 33 jets de magnitud (1320 floats) ─▶ comparación (coseno con resta de media)
```

- **`modelos/fbg.npz`** — Face Bunch Graph (33 nodos, 87 aristas) construido con
  `scripts/construir_bunch.py`. El etiquetado automático de nodos usa el modelo LBF de
  68 puntos **solo offline**; en ejecución no hace falta.
- **`datos/bd.csv`** — base de datos: `id`, `template` (base64) y `hmac` opcional.

## Reproducir los experimentos

```bash
python scripts/construir_bunch.py --lbf modelos/lbfmodel.yaml   # (solo si se regenera el FBG)
python scripts/inscribir_dataset.py       # inscribe las 50 fotos -> datos/bd.csv
python scripts/evaluar.py                  # FAR/FRR/EER, rank-1 -> resultados/
python scripts/metricas.py                 # tablas y figuras
python scripts/ataques.py                  # batería de vulnerabilidades -> resultados/ataques.json
```

La memoria con el análisis completo y el estudio de vulnerabilidades está en
`memoria/memoria.pdf`.

## Estructura

```
ebgm/         detector, normalización, Gabor, grafo, matching EBGM, sistema (API)
scripts/      construcción del FBG, inscripción, evaluación, métricas, ataques, inversión
datos/        bd.csv y dataset de imágenes
modelos/      fbg.npz, yunet.onnx, haarcascade
resultados/   métricas, figuras y resultados de los ataques
memoria/      memoria LaTeX
```
