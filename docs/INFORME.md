# Informe — Sistema de reconocimiento facial EBGM

Asignatura: Sistemas Biométricos · Método asignado: **EBGM (Elastic Bunch Graph
Matching)** · Famoso propio: **El Cigala** · Fases cubiertas: **1 (creación)** y
**2 (evaluación propia)**.

Pipeline: `foto → [YuNet] → cara 112×112 alineada → [EBGM: CLAHE + Gabor 7×8 + grafo fiducial 31 nodos + bunch graph] → template (31×58) → [matcher EBGM] → BD .csv`

Todas las cifras están **medidas** con los scripts del repositorio y se pueden
reproducir (sección 6).

---

## 1. Requisitos del enunciado

| Requisito | Estado |
|---|---|
| Python | ✅ |
| Repositorio fácil de ejecutar + **Docker** | ✅ `docker build -t ebgm . && docker run --rm ebgm` arranca con la BD lista |
| BD en **.csv** | ✅ `datos/bd_sistema.csv`: `id,template,mac` (+ `.mac`); templates cifrados |
| Personas inscritas | ✅ **50** = 49 famosos del `Dataset.tar.gz` + **El Cigala**. (El enunciado dice 50+1; el comprimido trae 49.) |
| 9 métodos (`inscribir/verificar/identificar` × normal, `_sd`, `_t`) | ✅ probados en `scripts/demo.py`, coherentes entre variantes |
| Evaluación con ataques (saltar partes, leer BD, escribir BD) | ✅ `scripts/evaluar_ataques.py`: 23 ataques ejecutados |

---

## 2. Fase 1 — Creación

### 2.1 Problemas detectados durante el desarrollo
| Problema | Consecuencia | Solución |
|---|---|---|
| `aitana_etxebarria_(twin_melody)_2024.jpg` es un **AVIF** con extensión .jpg (y `falete_2018.jpg` es WebP) | OpenCV no los lee: solo se inscribirían 49 personas | Lector de respaldo con Pillow (AVIF/HEIC/WebP + orientación EXIF) |
| Un umbral guardado en un fichero aparte podía faltar o editarse | Si faltaba, se usaba uno por defecto que aceptaba a casi cualquier impostor | Umbrales **dentro del modelo**, protegidos por el HMAC de la BD |
| Calibrar el umbral con las mismas fotos con las que se mide | Métricas optimistas (fuga de datos) | Calibración solo con datos de ajuste; `datos/fotos_cigala_test/` reservado para la evaluación |
| Templates sin validar | Se aceptaban templates con NaN o negativos | Validación estricta de forma, rango y caricidad |
| Templates en claro y clave junto a la BD | Leer el `.csv` daba templates reutilizables | **Cifrado AES-256-GCM** de los templates; la clave no se versiona y se genera en cada despliegue |

### 2.2 Diseño de EBGM
`reconocimiento/ebgm.py` implementa EBGM completo y configurable (`ConfigEBGM`).
La configuración se eligió con `scripts/ajuste_ebgm.py` en 6 rondas de
experimentos, **sin usar el test**:

| Componente | Versión inicial | Versión final |
|---|---|---|
| Preprocesado | ninguno | **CLAHE** (Tan-Triggs probado: peor) |
| Banco de Gabor | 5 escalas (λ 4–16) | **7 escalas (λ 5–40)**, kernels de Wiskott con compensación DC |
| Grafo | rejilla 9×8 fija | **31 nodos fiduciales** (ojos, cejas, nariz, boca, mentón…) |
| Ajuste elástico | ninguno | **Face Bunch Graph**: cada nodo busca (±3 px) el jet más parecido al manojo, con regularización |
| Normalización | resta de cara media | **√ + z-score por nodo y filtro** |
| Matcher | coseno medio de todos los nodos | coseno **invariante a espejo**, con **agregación robusta**: la mejor de las dos medias caras, descartando sus 3 nodos peores |
| Template | magnitudes (72×40) | magnitudes + posición de nodos (31×58), con cabecera de versión |

Resultados del ajuste (protocolo de **1 foto inscrita**: 156 pares del Cigala;
impostores, famosos entre sí y Cigala contra famosos):

| Variante | EER Cigala | d' Cigala | Rank-1 Cigala | TAR@FAR1 % | EER degradaciones | Rank-1 espejo |
|---|---|---|---|---|---|---|
| EBGM inicial | 34,6 % | 0,65 | 19 % | 2,6 % | 4,6 % | — |
| + 7 escalas + CLAHE + fiducial + σ=0,75λ | 16,7 % | 1,87 | 33 % | 32,1 % | 2,6 % | 71 % |
| + invariancia a espejo | 14,1 % | 2,01 | 46 % | 34,6 % | 0,3 % | 98 % |
| **+ agregación robusta de nodos (final)** | **12,8 %** | **2,14** | **51 %** | **41,7 %** | **0,3 %** | **98 %** |

La agregación robusta tolera giros de cabeza (un lado de la cara se ve peor) y
oclusiones parciales (gafas, pelo, barba) sin aprender nada de ninguna persona.

**Descartado** por no aportar (implementado y activable en `ConfigEBGM`, o
medido en el ajuste):
- Tan-Triggs, simetrizar el template, rejilla densa o mixta, término de
  geometría, compresión logarítmica, 8 escalas, 12 orientaciones y
  micro-variaciones al inscribir.
- Normalización de puntuaciones con la cohorte (T-norm): empeora.
- Pesos por nodo aprendidos: sin efecto si se aprenden de las degradaciones. Si
  se aprenden del Cigala, mejoran, pero bajan el peso de los ojos (sus gafas de
  sol) y suben el de boca y mentón (su barba): son pesos de una persona y no
  generalizan.
- Blanqueado de la variación intra-persona (WCCN): +1-3 puntos, dentro del ruido.

Con **una sola foto por persona** y un método de características diseñadas a
mano, el margen de mejora restante sin más datos es pequeño.

---

## 3. Fase 2 — Evaluación del rendimiento (`scripts/evaluar.py`)

Sistema **desplegado** (BD real, umbral 0,327 calibrado a FAR 1 %).

### 3.1 Lo que FUNCIONA
| Prueba | Resultado |
|---|---|
| 5 fotos NUEVAS del Cigala vs Cigala inscrito | **5/5 aceptadas**, rank-1 **100 %** (azar 2 %), d' 5,3 |
| Fotos degradadas de los 49 famosos (oscura, clara, baja resolución, JPEG calidad 15, giro 12°, ruido, luz lateral) | **100 %** rank-1 y verificación |
| Fotos en espejo (selfie) | 98 % |
| FAR real en el umbral desplegado | 1,2 % (1470 pares impostores) |
| El template medio del Cigala **no es un «imán»** | atracción top-3 0,31, por debajo del umbral |
| Velocidad | ~100 ms por foto; `_sd` ~50 ms; `_t` < 1 ms |

### 3.2 Lo que FALLA (límites conocidos del reconocimiento)
| # | Fallo | Evidencia |
|---|---|---|
| R1 | Con **una sola foto inscrita** y fotos muy distintas (otra década, perfil, gafas), EBGM rechaza mucho | Test vs 1 foto: EER 12,4 %, d' 2,0, **TAR ≈ 38 % a FAR 1 %** (FRR 60 % en el umbral) |
| R2 | **Gemelos**: Twin Melody, gemelos Weasley y Olsen están en el Dataset; el gemelo no inscrito pasará por el inscrito | Inherente a la biometría facial |
| R3 | Parejas de famosos que **superan el umbral** entre sí (FAR 1 % ⇒ ~1 de cada 100 pares) | ariana_grande ~ mary-kate_olsen 0,49; ariana_grande ~ metrika 0,41; christian_bale ~ keanu_reeves 0,41 |
| R4 | «Lobos/corderos»: ariana_grande, keanu_reeves, mary-kate_olsen y metrika atraen más impostores | Zoo de Doddington en `scripts/evaluar.py` |
| R5 | El detector **no encuentra** caras < 40 px | 2/7 fotos de test (concierto lejano, panorámica 960×310) |
| R6 | Las métricas de test tienen **pocas muestras** (5 fotos de 2 eventos) | El EER de test (12,4 %) y el de ajuste (12,8 %) concuerdan, pero el intervalo es amplio |

---

## 4. Fase 2 — Evaluación de seguridad (`scripts/evaluar_ataques.py`)

23 ataques **ejecutados** sobre una copia de la BD. **16 bloqueados, 7
vulnerabilidades conocidas.**

Defensas: templates cifrados en reposo (AES-256-GCM, con id y huella del modelo
como datos asociados); HMAC por fila y MAC global (detectan filas alteradas,
añadidas, intercambiadas, borradas o restauradas); BD ligada al hash del modelo,
que lleva dentro los umbrales; modo estricto que falla en cerrado; validación de
ids y templates; control de «caricidad» en `_sd`/`_t`; respuesta sin
puntuaciones por defecto; misma respuesta para un id inexistente que para un
rechazo; no sobrescritura; aviso de duplicados; auditoría; sha256 fijado del
detector; escritura atómica.

### 4.1 (A) El atacante se salta partes del sistema
| Ataque | Resultado | Mitigación / comentario |
|---|---|---|
| A1 *Replay* de un template por `verificar_t` | ⚠️ VULNERABLE (sim 1,0) | Inherente a `_t`. No exponer `_t` fuera; plantillas cancelables; canal autenticado extractor→matcher |
| A2 No-caras por `_sd` | ⚠️ PARCIAL: plano, ruido, gradiente y patrones periódicos **bloqueados**; caras invertidas o giradas y fondos con textura **pasan**. Con las 232 no-caras de calibración: 125 pasan la caricidad y **111 salen identificadas como alguien** | La caricidad EBGM (bunch graph de 49 caras) no separa caras no vistas de esas texturas. En el pipeline completo las para el detector |
| A3 Templates basura por `_t` (NaN, ceros, negativos, enormes, forma errónea, aleatorios, tipos raros) | ✅ 8/8 bloqueados | Validación + caricidad |
| A4 *Hill-climbing* con la configuración entregada | ✅ bloqueado: por defecto solo se devuelve la decisión, sin similitudes | — |
| A4b *Hill-climbing* si se activa `exponer_puntuaciones=True` | ⚠️ VULNERABLE: forja un template aceptado en ~300 consultas | Ese modo es solo para evaluación; añadir un límite de intentos |
| A5 *Morphing* (template medio atacante+víctima) | ⚠️ VULNERABLE: verifica como ambos | `posible_duplicado` lo avisa si uno ya está inscrito; control humano en las altas |
| A6 Ataque de presentación (foto de una foto o de una pantalla) | ⚠️ VULNERABLE | Sin *liveness*/PAD: límite conocido |
| A7 Varias caras en la foto | ✅ al inscribir se rechaza; al identificar se usa la **más grande** (por diseño) | Un cómplice detrás no cuenta; una cara más grande delante, sí |

### 4.2 (B) El atacante lee la BD
| Ataque | Resultado | Mitigación |
|---|---|---|
| B1 Exfiltración de templates | ✅ la columna template va **cifrada** (AES-256-GCM, AAD = id + huella del modelo) | Los ids siguen en claro. Con la clave robada (B3), sí se descifran |
| B2 *Replay* de la celda del .csv por `verificar_t` | ✅ rechazado: la celda cifrada no es un template | El replay de un template obtenido por otra vía es A1 |
| B3 Clave en `models/clave_bd.key` | ⚠️ VULNERABLE si se lee el disco | `EBGM_CLAVE_BD` por entorno, secreto del orquestador o HSM |
| B4 Reconstruir la cara desde el template | ✅ desde el .csv no hay template (cifrado). **No se da por imposible**: con plantillas de magnitudes de Gabor se puede llegar a reconstruir una cara aceptable | Riesgo residual si se roba la clave o se captura un template; *template protection* / plantillas cancelables |
| B5 Enumerar ids con `verificar` | ✅ id inexistente y rechazo dan la misma respuesta | El motivo solo queda en la auditoría |

### 4.3 (C) El atacante escribe en la BD (sin la clave)
| Ataque | Resultado |
|---|---|
| C1 Sustituir el template de la víctima | ✅ HMAC lo rechaza y el **modo estricto** detiene el sistema (falla en cerrado: DoS, nunca suplantación) |
| C2 Añadir una fila «admin» | ✅ rechazada |
| C3 Intercambiar template+MAC entre dos ids | ✅ el MAC liga id+template |
| C4 Borrar una fila (o el .csv entero) | ✅ MAC global incorrecto ⇒ detectado y bloqueado |
| C5 Restaurar una fila antigua válida | ✅ detectado por el MAC global. ⚠️ Restaurar **BD + .mac completos** de una copia antigua **sí pasa** (no hay contador de versión externo) |
| C6 Escribir **con** la clave robada (B3) | ⚠️ VULNERABLE: control total |
| C7 Bajar el umbral dentro del modelo | ✅ el hash del modelo va en el HMAC ⇒ 0 filas válidas (falla en cerrado) |
| C8 Ids maliciosos (`=HYPERLINK…`, `+cmd`, saltos de línea, vacíos, 200 caracteres, no-texto) | ✅ 8/8 rechazados |
| C9 Re-inscribir un id existente por la API | ✅ exige `sobrescribir=True` |
| C10 Misma cara con otro nombre | ✅ aceptada pero marcada como `posible_duplicado` |

### 4.4 Otros defectos conocidos (por diseño o fuera de alcance)
- **Ids en claro en el .csv**: quien lea la BD sabe QUIÉN está inscrito (no su template). Por la API no se puede enumerar (B5).
- **Oráculo vía `inscribir`**: aunque no se exponen similitudes, `posible_duplicado` devuelve el id de la persona inscrita a la que se parece la cara. Quien tenga permiso para inscribir puede usarlo para identificar a alguien (es un aviso pensado para el operador).
- **Homoglifos**: «аdele» con «а» cirílica es un id distinto que se ve igual.
- **Sin límite de intentos** entre procesos: quien llama a la API en Python puede reintentar sin fin.
- **Concurrencia**: dos procesos que escriben a la vez ⇒ gana el último (la escritura es atómica, pero no hay bloqueo).
- La **clave se hornea en la imagen Docker** si no se pasa `EBGM_CLAVE_BD`.
- `politica_multiples="mayor"` en verificar/identificar (ver A7).
- Umbral elegido a **FAR 1 %**, que prioriza la seguridad sobre la comodidad (FRR alta con 1 foto). Otro punto de operación: `python scripts/inscribir_dataset.py --far-objetivo 0.05`.

---

## 5. Pendiente
- **Fase 3**: evaluar el sistema del grupo rival (`scripts/evaluar_ataques.py` sirve de plantilla).
- **Fase 4**: exposición (este informe y `salida/det.png` son la base).

---

## 6. Cómo reproducir
```bash
python scripts/inscribir_dataset.py   # modelo EBGM + 50 personas + umbrales
python scripts/demo.py                # los 9 métodos
python scripts/ajuste_ebgm.py         # fase 1: comparación de variantes (lento)
python scripts/evaluar.py             # fase 2: rendimiento (+ salida/det.png)
python scripts/evaluar_ataques.py     # fase 2: seguridad
```
