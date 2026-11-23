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
| 9 métodos (`inscribir/verificar/identificar` × normal, `_sd`, `_t`) | ✅ probados en `demo.py`, coherentes entre variantes |
| Evaluación con ataques (saltar partes, leer BD, escribir BD) | ✅ `scripts/evaluar_ataques.py`: 23 ataques ejecutados |

---

## 2. Fase 1 — Creación: qué se mejoró y por qué

### 2.1 Fallos del sistema anterior (encontrados y corregidos)
| Fallo | Consecuencia | Corrección |
|---|---|---|
| `aitana_etxebarria_(twin_melody)_2024.jpg` es un **AVIF** con extensión .jpg (y `falete_2018.jpg` es WebP) | OpenCV no lo leía: **solo se inscribían 49 personas** | Lector de respaldo con Pillow (AVIF/HEIC/WebP + orientación EXIF) |
| Sin `umbral.txt`, el sistema usaba **0.047** en vez del umbral calibrado (0.277) | Casi cualquier impostor era aceptado | Umbrales dentro del modelo, protegidos por el HMAC de la BD |
| El umbral se calibraba con las **mismas 5 fotos de test** con las que se medía | Métricas optimistas (fuga de datos) | Calibración solo con datos de ajuste; `fotos_cigala_test/` reservado |
| La demo probaba con `fotos_cigala/4.jpg` diciendo «no inscrita» | Era una foto de inscripción | La demo usa una foto de test nunca vista |
| El fondo se entrenaba con la cohorte que se inscribía, sin validar | Templates NaN/negativos aceptados | Validación estricta de templates |

### 2.2 EBGM mejorado
Se reescribió `ebgm.py` como EBGM completo y configurable (`ConfigEBGM`), y se
eligió la configuración con `ajuste_ebgm.py` en 5 rondas de experimentos,
**sin usar el test**:

| Componente | Antes | Ahora |
|---|---|---|
| Preprocesado | ninguno | **CLAHE** (Tan-Triggs probado: peor) |
| Banco de Gabor | 5 escalas (λ 4–16) | **7 escalas (λ 5–40)**, kernels de Wiskott con compensación DC |
| Grafo | rejilla 9×8 fija | **31 nodos fiduciales** (ojos, cejas, nariz, boca, mentón…) |
| Ajuste elástico | ninguno | **Face Bunch Graph**: cada nodo busca (±3 px) el jet más parecido al manojo, con regularización |
| Normalización | resta de cara media | **√ + z-score por nodo y filtro** |
| Matcher | coseno medio | coseno medio **invariante a espejo** |
| Template | magnitudes (72×40) | magnitudes + posición de nodos (31×58), con cabecera de versión |

Resultados del ajuste (protocolo de **1 foto inscrita**: 156 pares del Cigala;
impostores, famosos entre sí y Cigala contra famosos):

| Variante | EER Cigala | d' Cigala | Rank-1 Cigala | TAR@FAR1 % | EER degradaciones | Rank-1 espejo |
|---|---|---|---|---|---|---|
| EBGM anterior | 34,6 % | 0,65 | 19 % | 2,6 % | 4,6 % | — |
| + 7 escalas + CLAHE + fiducial + σ=0,75λ | 16,7 % | 1,87 | 33 % | 32,1 % | 2,6 % | 71 % |
| + invariancia a espejo (ronda 5) | 14,1 % | 2,01 | 46 % | 34,6 % | 0,3 % | 98 % |
| **+ agregación robusta: mejor media cara sin sus 3 nodos peores (final, ronda 6)** | **12,8 %** | **2,14** | **51 %** | **41,7 %** | **0,3 %** | **98 %** |

Descartado por no aportar (implementado y activable): Tan-Triggs, simetrizar el
template, rejilla densa o mixta, término de geometría, compresión logarítmica,
8 escalas, 12 orientaciones y micro-variaciones al inscribir.

---

## 3. Fase 2 — Evaluación del rendimiento (`evaluar.py`)

Sistema **desplegado** (BD real, umbral 0,327 calibrado a FAR 1 %).

### 3.1 Lo que FUNCIONA
| Prueba | Resultado |
|---|---|
| 5 fotos NUEVAS del Cigala vs Cigala inscrito | **5/5 aceptadas**, rank-1 **100 %** (azar 2 %), d' 5,4 |
| Fotos degradadas de los 49 famosos (oscura, clara, baja resolución, JPEG calidad 15, giro 12°, ruido, luz lateral) | **100 %** rank-1 y verificación |
| Fotos en espejo (selfie) | 98 % |
| FAR real en el umbral desplegado | 1,2 % (1470 pares impostores) |
| El template medio del Cigala **no es un «imán»** | atracción top-3 0,19, por debajo del umbral |
| Velocidad | ~50 ms por foto; `_t` < 1 ms |

### 3.2 Lo que FALLA (límites conocidos del reconocimiento)
| # | Fallo | Evidencia |
|---|---|---|
| R1 | Con **una sola foto inscrita** y fotos muy distintas (otra década, perfil, gafas), EBGM rechaza mucho | Test vs 1 foto: EER 12,4 %, **TAR ≈ 38 % a FAR 1 %** (FRR 60 % en el umbral). Antes de la ronda 6: EER 15 %, TAR 25 %, FRR 63 % |
| R2 | **Gemelos**: Twin Melody, gemelos Weasley y Olsen están en el Dataset; el gemelo no inscrito pasará por el inscrito | Inherente a la biometría facial |
| R3 | Parejas de famosos que **superan el umbral** entre sí (FAR 1 % ⇒ ~1 de cada 100 pares) | ariana_grande ~ mary-kate_olsen 0,36; christian_bale ~ keanu_reeves 0,34; idris_elba ~ willem_dafoe 0,30 |
| R4 | «Lobos/corderos»: keanu_reeves, mary-kate_olsen, ariana_grande y metrika atraen más impostores | Zoo de Doddington en `evaluar.py` |
| R5 | El detector **no encuentra** caras < 40 px | 2/7 fotos de test (concierto lejano, panorámica 960×310) |
| R6 | Las métricas de test tienen **pocas muestras** (5 fotos de 2 eventos) | El EER de test (15 %) y el de ajuste (14 %) concuerdan, pero el intervalo es amplio |

---

## 4. Fase 2 — Evaluación de seguridad (`evaluar_ataques.py`)

23 ataques **ejecutados** sobre una copia de la BD. **16 bloqueados, 7
vulnerabilidades conocidas.**

### 4.1 (A) El atacante se salta partes del sistema
| Ataque | Resultado | Mitigación / comentario |
|---|---|---|
| A1 *Replay* de un template por `verificar_t` | ⚠️ VULNERABLE (sim 1,0) | Inherente a `_t`. No exponer `_t` fuera; plantillas cancelables; canal autenticado extractor→matcher |
| A2 No-caras por `_sd` | ⚠️ PARCIAL: plano, ruido, gradiente y patrones periódicos **bloqueados**; caras invertidas o giradas y fondos con textura **pasan**. Medido con las 232 no-caras de calibración: 125 pasan la caricidad y **111 salen identificadas como alguien** (108 con la agregación anterior: no es efecto de la ronda 6) | La caricidad EBGM (bunch graph de 49 caras) no separa caras no vistas de esas texturas. En el pipeline completo las para el detector |
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
| B2 *Replay* de la celda del .csv por `verificar_t` | ✅ rechazado: la celda cifrada no es un template | El replay de un template obtenido por otra vía sigue siendo A1 |
| B3 Clave HMAC en `models/clave_bd.key` | ⚠️ VULNERABLE si se lee el disco | `EBGM_CLAVE_BD` por entorno, secreto del orquestador o HSM |
| B4 Reconstruir la cara desde el template | ✅ desde el .csv no hay template (cifrado). **No lo damos por imposible**: la rama de Pablo Anguita lo logró con magnitudes de Gabor | Riesgo residual si se roba la clave o se captura un template; *template protection* / plantillas cancelables |
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
- **Ids en claro en el .csv**: quien lea la BD sabe QUIÉN está inscrito (no su template). Por la API ya no se puede enumerar (B5).
- **Oráculo vía `inscribir`**: aunque no se exponen similitudes, `posible_duplicado` devuelve el id de la persona inscrita a la que se parece la cara. Quien tenga permiso para inscribir puede usarlo para identificar a alguien (es un aviso pensado para el operador).
- **Homoglifos**: «аdele» con «а» cirílica es un id distinto que se ve igual.
- **Sin límite de intentos** entre procesos: quien llama a la API en Python puede reintentar sin fin.
- **Concurrencia**: dos procesos que escriben a la vez ⇒ gana el último (la escritura es atómica, pero no hay bloqueo).
- La **clave se hornea en la imagen Docker** si no se pasa `EBGM_CLAVE_BD`.
- `politica_multiples="mayor"` en verificar/identificar (ver A7).
- Umbral elegido a **FAR 1 %**, que prioriza la seguridad sobre la comodidad (FRR alta con 1 foto). Otro punto de operación: `inscribir_dataset.py --far-objetivo 0.05`.

---

## 4.5 Cambios de la rama `fernando`
Ver [`COMPARATIVA.md`](COMPARATIVA.md): templates cifrados en reposo, clave fuera
del repositorio (estaba versionada y se ha rotado), sin enumeración de ids y B4
revisado a partir del hallazgo de la rama de Pablo Anguita.

## 5. Pendiente
- **Fase 3**: evaluar el sistema del grupo rival (`evaluar_ataques.py` sirve de plantilla).
- **Fase 4**: exposición (este informe y `salida/det.png` son la base).
- Decisión del grupo: el punto de operación (FAR 1 % frente a uno más permisivo).
  Ya decidido: por defecto **no** se exponen similitudes (`exponer_puntuaciones=False`).

---

## 6. Cómo reproducir
```bash
python scripts/inscribir_dataset.py   # modelo EBGM + 50 personas + umbrales
python scripts/demo.py                # los 9 métodos
python scripts/ajuste_ebgm.py         # fase 1: comparación de variantes (lento)
python scripts/evaluar.py             # fase 2: rendimiento (+ salida/det.png)
python scripts/evaluar_ataques.py     # fase 2: seguridad
```
