# Informe — Sistema de reconocimiento facial EBGM

Asignatura: Sistemas Biométricos · Método asignado: **EBGM (Elastic Bunch Graph
Matching)** · Famoso propio: **El Cigala**.

Pipeline: `foto → [DETECTOR YuNet] → cara 112×112 → [EXTRACTOR EBGM] → template → [MATCHER] → BD (.csv)`.

Este informe resume **qué funciona**, **qué falla** y **qué queda pendiente**,
con datos medidos (no estimados). Las cifras salen de `evaluar.py` y
`evaluar_ataques.py`, evaluando con fotos del cigala **que el sistema no vio**
al inscribirlo (`fotos_cigala_test/`, descargadas de Wikimedia Commons).

---

## 1. Lo que FUNCIONA

### 1.1 Sistema y requisitos del enunciado
| Requisito | Estado |
|---|---|
| Lenguaje Python | ✅ |
| Repositorio fácil de ejecutar + **Docker** listo | ✅ (imagen `ebgm`, arranca con la BD ya inscrita) |
| Base de datos en **.csv** | ✅ (`id,template` en base64) |
| **9 métodos** (`inscribir/verificar/identificar` × normal/`_sd`/`_t`) | ✅ todos, probados y coherentes entre variantes |
| Personas inscritas | ✅ 49 famosos + cigala = **50** (ver *falla* 3.4) |
| Evaluación propia + supuestos de ataque | ✅ `evaluar.py` y `evaluar_ataques.py` |

### 1.2 Reconocimiento (tras inscribir al cigala como centroide)
El cigala se inscribe como **centroide** (media justa, en espacio normalizado) de
sus 13 fotos de referencia: una sola plantilla por persona, igual que los famosos.

| Métrica (sobre fotos NUEVAS, galería de 50) | 1 sola foto | **Centroide (desplegado)** |
|---|---|---|
| Rank-1 (identificación 1:N) | 0/5 | **4/5 (80 %)** · azar ≈ 2 % |
| d-prime (separación genuino/impostor) | 0.8 | **3.3** |
| EER (verificación 1:1) | ~40 % | **~1–2 %** (optimista, pocas genuinas) |
| FAR zero-effort (famosos que pasan como cigala) | 18/49 | **0/49** |

- La mejora es **legítima**: el centroide **no infla** las puntuaciones ni crea
  "imán" de falsos positivos (0/49 famosos se confunden con el cigala).
- Las similitudes genuinas (0.27–0.43) quedan bien separadas de las impostoras
  (centradas en ~0); el umbral desplegado es **0.277**.

### 1.3 Robustez ya implementada
- El **detector valida la entrada**: rechaza imágenes corruptas o < 32 px,
  reescala > 4096 px, descarta caras < 40 px; al **inscribir** usa política
  estricta (error si hay varias caras).
- El cargador de la BD **ignora filas corruptas** sin caerse (tolerancia a
  sabotaje parcial del .csv).
- Funciona con rutas con tildes/ñ (lectura por buffers de NumPy, útil en Windows).

---

## 2. Lo que FALLA en el reconocimiento (límites del método)

| # | Fallo | Evidencia |
|---|---|---|
| 2.1 | **No detecta cara** en encuadres muy apaisados o formatos atípicos | 2 de 7 fotos nuevas (`koerner_2014` 960×310, `elcigala`) |
| 2.2 | **Perfiles / pose extrema** se confunden con otro famoso | 1 de 5 fotos nuevas (un perfil → `christian_bale`) |
| 2.3 | EBGM clásico (jets de Gabor + coseno) **generaliza poco** a fotos no controladas | con 1 sola foto el rank-1 era 0/5; el método asume cara frontal alineada |
| 2.4 | EER optimista | solo 5 fotos genuinas de prueba; la separación (d-prime 3.3) sí es sólida |

> Es coherente con el enunciado: el objetivo no es un sistema perfecto. EBGM
> rinde en caras frontales de condiciones parecidas y se degrada con pose/edad.

---

## 3. Puntos DÉBILES de seguridad (los conocemos nosotros)

El objetivo de la práctica es que el grupo rival **no** encuentre fallos que no
conozcamos. Auditoría realizada (los 3 del enunciado **y** extras verificados):

### 3.1 Los 3 supuestos del enunciado (en `evaluar_ataques.py`)
- **Saltarse el sistema** (`_sd`/`_t`): *replay* de un template = **suplantación
  total** (similitud 1.0); entrada no validada colada por `_sd`.
- **Leer la BD**: templates **en claro** → exfiltración y reutilización por replay.
- **Escribir la BD**: sustitución de identidad (el atacante se vuelve "cigala"),
  puerta trasera (alta propia) y sabotaje/DoS.

### 3.2 Debilidades extra encontradas en la auditoría
| # | Debilidad | Verificado |
|---|---|---|
| a | **Hill-climbing**: como `verificar/identificar` **devuelven la similitud**, con solo consultas se forja un template que pasa | sí — ~750 consultas → sim 0.30, aceptado, sin conocer el real |
| b | **Basura aceptada**: ruido / imagen negra o blanca se identifican por `_sd` | sí — sim 0.58–0.70 |
| c | `_sd` acepta **cualquier tamaño** (redimensiona en silencio) | sí |
| d | **Template NaN/Inf no validado** al inscribir | sí — contamina la BD |
| e | `inscribir` **sobrescribe** un id existente sin avisar (secuestro) | sí |
| f | `politica_multiples="mayor"` en verificar/identificar → **cara distractora** | por diseño |
| g | El fichero de fondo `ebgm_fondo.npy` **no está protegido** (altera todo) | por diseño |
| h | **Sin límite de altas** (DoS) y escritura del .csv no atómica (TOCTOU) | por diseño |
| i | Respuestas **filtran** ranking+scores (alimenta el hill-climbing y revela inscritos) | por diseño |
| j | Template **irrevocable** y **sin liveness** (ataque de foto impresa/pantalla) | por diseño |

### 3.3 Mitigaciones propuestas
No exponer `_sd`/`_t` al exterior; *liveness*/antispoofing; **cifrar** la BD y
control de acceso; **protección de plantilla** (biohashing / fuzzy vault, para que
lo filtrado no sea reutilizable); **integridad por fila (HMAC)** + auditoría;
validar entradas (cara real, tamaño, template finito); no devolver scores crudos;
límite de altas y escritura atómica. *(Parte del FAR ya se mitigó al calibrar el
umbral en 0.277: el impostor sin esfuerzo pasó de 18/49 a 0/49.)*

### 3.4 Otros
- El `Dataset.tar.gz` trae **49** famosos, no 50; con el cigala la BD tiene **50**
  identidades (el enunciado menciona 51). A revisar con el docente.

---

## 4. PENDIENTE (fases del enunciado)
- **Fase 3** — evaluar el sistema de un grupo **rival** (cuando lo entreguen; las
  herramientas de ataque ya sirven).
- **Fase 4** — **exposición** de resultados/conclusiones (este informe es la base).
- Decisiones del grupo: ¿aplicar las defensas de 3.3?, ¿subir al repositorio
  (aún no hay commit)?, ¿resolver el 50 vs 51 con el docente?

---

## 5. Cómo reproducir
```bash
python inscribir_dataset.py      # inscribe famosos + cigala (centroide) y calibra
python demo.py                   # los 9 métodos funcionando
python evaluar.py                # métricas (rank-1, EER, d-prime)
python evaluar_ataques.py        # análisis de seguridad
# o con Docker:
docker build -t ebgm . && docker run --rm ebgm
```
