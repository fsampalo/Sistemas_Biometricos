# Reconocimiento facial EBGM — Sistemas Biométricos

Práctica de la asignatura. El sistema de reconocimiento facial usa como método
**EBGM (Elastic Bunch Graph Matching)**: un grafo de nodos fiduciales sobre la
cara, jets de Gabor en cada nodo, un *Face Bunch Graph* para ajustar el grafo a
cada cara y un emparejamiento de grafos para comparar.

Nuestro famoso elegido es **El Cigala**. La BD tiene **50 personas**: los 49
famosos del docente (`Dataset.tar.gz`) más el Cigala.

```
foto -> [DETECTOR YuNet] -> cara 112x112 alineada -> [EXTRACTOR EBGM] -> template -> [MATCHER EBGM] -> BD (.csv)
```

| Etapa | Fichero | Qué hace |
|---|---|---|
| Detector | [`detector.py`](detector.py) | YuNet (OpenCV). Detecta la cara y la **alinea** a 112×112 con 5 landmarks. Lee JPG/PNG/WebP/AVIF/HEIC. |
| Extractor + matcher | [`ebgm.py`](ebgm.py) | **EBGM**: CLAHE → Gabor 7×8 → grafo fiducial de 31 nodos → ajuste elástico con el *bunch* → jets → similitud. |
| Sistema + BD | [`sistema.py`](sistema.py) | Los **9 métodos** del enunciado, BD `.csv` con integridad HMAC y defensas. |
| Modelo / BD | [`inscribir_dataset.py`](inscribir_dataset.py) | Entrena el modelo EBGM, inscribe las 50 personas y calibra los umbrales. |
| Ajuste (fase 1) | [`ajuste_ebgm.py`](ajuste_ebgm.py) | Banco de experimentos con el que se eligió la configuración. |
| Evaluación (fase 2) | [`evaluar.py`](evaluar.py) | EER, d', FAR/FRR, CMC, robustez, riesgos y tiempos del sistema desplegado. |
| Seguridad (fase 2) | [`evaluar_ataques.py`](evaluar_ataques.py) | 22 ataques ejecutados de verdad (saltar etapas, leer BD, escribir BD). |

## Puesta en marcha

### Con Docker (recomendado)

```bash
docker build -t ebgm .
```

```bash
docker run --rm ebgm
```

```bash
docker run --rm ebgm python evaluar.py
```

```bash
docker run --rm ebgm python evaluar_ataques.py
```

El build extrae `Dataset.tar.gz`, descarga YuNet (verificando su sha256), entrena
el modelo e inscribe a las 50 personas: el contenedor arranca listo.

### En local

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python download_models.py         # YuNet en models/ (sha256 verificado)
tar -xzf Dataset.tar.gz           # famosos del docente -> Dataset/
python inscribir_dataset.py       # modelo EBGM + BD de 50 personas + umbrales
python demo.py                    # los 9 métodos funcionando
```

## Los 9 métodos

```python
from sistema import SistemaReconocimiento
sis = SistemaReconocimiento()

sis.inscribir("ana", "ana.jpg")            # -> {'inscrito': 'ana', 'posible_duplicado': None}
sis.verificar("ana", "otra_ana.jpg")       # -> {'verificado': True}
sis.identificar("quien.jpg")               # -> {'id': 'ana', 'aceptado': True}

# modo evaluación (similitudes + ranking top-5), p. ej. para curvas DET:
ev = SistemaReconocimiento(exponer_puntuaciones=True)
ev.verificar("ana", "otra_ana.jpg")        # -> {'verificado': True, 'similitud': 0.31, 'umbral': 0.23}

cara = sis.extraer_cara("foto.jpg")        # recorte 112x112 alineado (entrada de los _sd)
sis.inscribir_sd("ana", cara)              # omite el detector
sis.verificar_sd("ana", cara)
sis.identificar_sd(cara)

tpl = sis.extraer_template("foto.jpg")     # template EBGM (entrada de los _t)
sis.inscribir_t("ana", tpl)                # omite detector y extractor
sis.verificar_t("ana", tpl)
sis.identificar_t(tpl)
```

| Método | Entra por | Omite |
|---|---|---|
| `inscribir` / `verificar` / `identificar` | foto (ruta, bytes o `np.ndarray` BGR) | — |
| `inscribir_sd` / `verificar_sd` / `identificar_sd` | cara recortada y alineada 112×112 | detector |
| `inscribir_t` / `verificar_t` / `identificar_t` | template (`np.ndarray` (31, 58) o cadena base64 del .csv) | detector y extractor |

- `identificar` es de **conjunto abierto**: devuelve `id=None` si la mejor
  similitud no supera el umbral (la persona no está inscrita).
- `inscribir*` **no sobrescribe** un id existente salvo con `sobrescribir=True`,
  y avisa de `posible_duplicado` si esa cara ya está inscrita con otro nombre.
- Los errores controlados lanzan `ErrorDeteccion` (no hay cara, varias caras al
  inscribir, imagen inválida) o `ErrorSistema` / `ErrorCalidad` (id inválido,
  template inválido, entrada que no parece una cara, BD comprometida).
- **Por defecto no se devuelven similitudes ni ranking**, solo la decisión:
  así se frena el *hill-climbing*. `exponer_puntuaciones=True` las devuelve
  (modo evaluación). Otras opciones del constructor: `estricto=False` (no bloquear si la
  integridad de la BD falla), `umbral=...` (otro punto de operación).

Línea de comandos (añade `--puntuaciones` para ver similitudes): `python sistema.py identificar --foto x.jpg`,
`python sistema.py verificar --id cigala --foto x.jpg`, `python sistema.py listar`.

## El método: EBGM

1. **Preprocesado.** CLAHE (ecualización local de contraste).
2. **Wavelets de Gabor.** 7 escalas (λ = 5…40 px) × 8 orientaciones, kernels
   complejos con compensación de DC (Wiskott). El **jet** de un punto es el
   vector de 56 magnitudes.
3. **Grafo fiducial.** 31 nodos en ojos, comisuras, cejas, ojeras, nariz,
   aletas, boca, mentón, pómulos, mandíbula y frente, simétricos izquierda/derecha.
4. **Face Bunch Graph.** Para cada nodo, el «manojo» de jets de esa posición en
   las 49 caras de la cohorte (más sus espejos). Al extraer, cada nodo se
   desplaza elásticamente (±3 px) hasta donde su jet mejor encaja en el manojo,
   con penalización de deformación y regularización con sus vecinos.
5. **Template.** Los 31 jets más la posición de cada nodo: matriz (31, 58).
6. **Matcher.** Media, nodo a nodo, del coseno entre jets comprimidos (√) y
   normalizados por la cohorte (z-score por nodo y filtro). Es **invariante a
   espejo**: se compara también con el grafo reflejado de la sonda (permutación
   exacta de nodos y orientaciones) y se toma el máximo.

El **modelo** (`models/ebgm_modelo.npz`: manojo, media/desviación de la cohorte y
umbrales) es común a todas las identidades. El Cigala se inscribe como el
**template medio** de sus 13 fotos de referencia (1 template, igual que el resto).

### Cómo se eligió la configuración (fase 1)

`python ajuste_ebgm.py` compara variantes **sin tocar el test**
(`fotos_cigala_test/` solo se usa en `evaluar.py`). Genuinos de ajuste: los 156
pares ordenados entre las 13 fotos del Cigala (caso real «1 foto inscrita») y 392
fotos degradadas de los famosos. Impostores: famosos entre sí y Cigala contra famosos.

| Ajuste (1 foto inscrita) | EBGM inicial (5 escalas, rejilla 9×8) | **EBGM final** |
|---|---|---|
| EER del Cigala | 34,6 % | **14,1 %** |
| d' del Cigala | 0,65 | **2,01** |
| Rank-1 del Cigala (galería de 50) | 19 % | **46 %** |
| TAR @ FAR 1 % del Cigala | 2,6 % | **34,6 %** |
| EER de las degradaciones | 4,6 % | **0,3 %** |
| Rank-1 de fotos en espejo | — | **98 %** (71 % sin invariancia) |

Lo que más aportó: **7 escalas** (las escalas grandes capturan la forma de la
cara), **CLAHE** (mejor que Tan-Triggs), **grafo fiducial** y **matcher
invariante a espejo**. Lo que se probó y **se descartó** por no mejorar:
Tan-Triggs, simetrizar el template, rejilla densa, término de geometría,
compresión logarítmica, 8 escalas, 12 orientaciones y micro-variaciones al
inscribir (están implementados y se pueden activar en `ConfigEBGM`).

## Resultados (fase 2) — `python evaluar.py`

Sistema desplegado: umbral **0,231** (FAR objetivo 1 %).

| Métrica | Resultado |
|---|---|
| FTA del detector | 0/49 famosos, 0/13 refs. del Cigala, 2/7 fotos de test (fotos de concierto con caras de ~20 px) |
| Test del Cigala vs Cigala inscrito | **5/5 aceptadas**, rank-1 **100 %**, d' 5,4 |
| Test del Cigala vs **1 sola foto** | EER 15 %, d' 1,9, TAR@FAR1 % ≈ 25 % |
| FAR real en el umbral desplegado | 1,3 % |
| Robustez (iluminación, baja resolución, JPEG 15, giro 12°, ruido, luz lateral) | **100 %** rank-1 y verificación |
| Robustez espejo | 98 % |
| Tiempo `verificar` / `identificar` (foto) | ~50 ms |

**Lectura honesta.** Con varias fotos de inscripción el sistema reconoce muy
bien. Con **una sola foto** (el caso de los 49 famosos) y fotos muy distintas
(otra década, perfil, gafas de sol), EBGM, que es un método clásico, se queda en
un TAR de ~25–35 % a FAR 1 %. Para el rival: una foto nueva de un famoso muy
distinta de la del Dataset tiene bastante probabilidad de ser rechazada.

## Seguridad — `python evaluar_ataques.py`

Defensas: templates **cifrados en reposo** (AES-256-GCM); HMAC por fila + MAC global (detecta filas alteradas, añadidas,
intercambiadas, borradas o restauradas); BD ligada al hash del modelo, que lleva
dentro los umbrales; modo estricto que falla en cerrado; validación de ids
(incluida la inyección de fórmulas CSV) y de templates; control de «caricidad»
EBGM en `_sd`/`_t`; no sobrescritura; aviso de duplicados; auditoría
(`salida/auditoria.log`); sha256 fijado del detector; escritura atómica.

**16 de 23 ataques bloqueados**; los **7 restantes son vulnerabilidades
CONOCIDAS**, cada una con su mitigación (detalle en [`INFORME.md`](INFORME.md)).

## Estructura

```
├── detector.py            # Etapa 1: detección + alineado (YuNet)
├── ebgm.py                # Etapas 2-3: extractor y matcher EBGM
├── sistema.py             # Los 9 métodos + BD .csv segura
├── inscribir_dataset.py   # Modelo EBGM + inscripción de 50 personas + calibración
├── ajuste_ebgm.py         # Fase 1: experimentos de ajuste
├── evaluar.py             # Fase 2: métricas del sistema desplegado
├── evaluar_ataques.py     # Fase 2: ataques
├── demo.py                # Demostración de los 9 métodos
├── download_models.py     # Descarga YuNet (sha256 verificado)
├── Dataset.tar.gz         # 49 famosos del docente (no versionado)
├── fotos_cigala/          # 13 fotos de referencia del Cigala (inscripción)
├── fotos_cigala_test/     # 7 fotos NUEVAS del Cigala (solo evaluación)
├── models/                # YuNet, ebgm_modelo.npz, calibracion.json (+ clave_bd.key generada, no versionada)
└── salida/                # det.png, auditoria.log, caché del ajuste
```

## Notas

- La BD (`bd_sistema.csv`, `.mac`) y la clave **se generan** con
  `python inscribir_dataset.py` (o en el `docker build`); no se versionan.
  Comparativa de ramas y cambios de esta rama: [`COMPARATIVA.md`](COMPARATIVA.md).

- `models/clave_bd.key` se genera la primera vez y **no se versiona**. En
  producción, pasa la clave por entorno: `EBGM_CLAVE_BD=<64 hex>`.
- Si se cambia `ConfigEBGM`, hay que re-ejecutar `inscribir_dataset.py`: los
  templates llevan la huella de la configuración y no se mezclan.
- Rutas con tildes/ñ en Windows: imágenes y modelos se leen por buffers.
