# Comparativa de ramas y origen de la rama `fernando`

Esta rama parte de **`eric-ebgm-mejorado`** (pablordrz, que a su vez sale de
`eric`) y la compara con **`reconocimiento-ebgm`** (Pablo Anguita). Las dos
implementan EBGM con los 9 métodos, BD `.csv` con HMAC y Docker. Lo que cambia
es el diseño y, sobre todo, la evaluación.

## 1. Qué tiene cada rama

| Aspecto | `eric-ebgm-mejorado` (pablordrz), **base** | `reconocimiento-ebgm` (Pablo Anguita) |
|---|---|---|
| Detector | YuNet, alineado 112×112 con 5 landmarks | YuNet (+ Haar de respaldo), alineado por ojos 144×144 |
| Gabor | 7 escalas × 8 orientaciones, CLAHE | 5 frecuencias × 8 orientaciones, z-score |
| Grafo | 31 nodos fiduciales, ajuste elástico ±3 px | 33 nodos / 87 aristas, FBG etiquetado offline con LBF 68 puntos |
| Template | magnitudes + posición de nodos (31×58) | solo magnitudes (33×40). Quitó posiciones y fases para que costara más invertirlo |
| Matcher | coseno con √ + z-score por nodo, **invariante a espejo** | coseno con resta del jet medio |
| Umbral | 0,231 calibrado a FAR 1 % **sin usar el test** | 0,62 fijo («FAR=0 en evaluación») |
| Respuesta por defecto | solo la decisión (sin similitudes) | `(aceptado, similitud)`: **expone la puntuación** |
| Integridad BD | HMAC por fila + **MAC global** (detecta borrados) + huella del modelo | HMAC por fila **opcional** (solo con `SB_CLAVE_HMAC`) |
| Modo estricto | sí, falla en cerrado | no: las filas malas se descartan y sigue |
| Ataques | 22, ejecutados (`evaluar_ataques.py`) | 7 (`scripts/ataques.py`) + memoria LaTeX con figuras |
| Clave | `models/clave_bd.key` **subida al repo** ⚠️ | por entorno |

## 2. Lo que la rama de Pablo Anguita encontró y la de pablordrz no

| Hallazgo de Anguita | Situación en la base | Qué hace esta rama |
|---|---|---|
| **P5**: con un template de magnitudes se puede reconstruir una cara que pasa `verificar_sd` (0,672 frente a un umbral de 0,62) | B4 decía «no factible» **sin haberlo probado**. Si el rival lo prueba, sería un defecto que no conocíamos | B4 se reescribe: **no se da por imposible**. Se mitiga quitando el template del alcance de quien solo lee el `.csv` (cifrado) y se documenta el riesgo residual |
| **P2**: replay del template leído de la BD (50/50) | B1/B2 vulnerables: templates en claro | **Templates cifrados con AES-256-GCM**: la celda del `.csv` ya no es un template utilizable |
| **P4**: hill-climbing con puntuación (8/8, ~1385 consultas) | Ya bloqueado: por defecto no se exponen similitudes | Sin cambios |
| **P1**: no-caras por `_sd` | Ya medido (A2, caricidad) | Sin cambios |

## 3. Lo que la base tenía mejor (y se conserva)

- MAC global y modo estricto: detectan borrados y restauraciones de filas. En
  Anguita, borrar una fila o quitar el HMAC pasa inadvertido.
- Umbral calibrado sin fuga de datos y test separado (`fotos_cigala_test/`).
- No se exponen puntuaciones por defecto.
- Validación de ids contra la inyección de fórmulas CSV, no sobrescritura y
  aviso de duplicados.

## 4. Cambios de la rama `fernando`

1. **Cifrado en reposo de los templates** (`sistema.py`). AES-256-GCM con una
   clave derivada de la maestra (distinta de la del HMAC). El id y la huella
   del modelo van como AAD, así que una celda cifrada no se puede mover a otra
   fila. B1, B2 y B4 pasan a estar **bloqueados** cuando el atacante solo lee el `.csv`.
2. **La clave sale del repositorio.** `models/clave_bd.key` estaba versionada:
   cualquiera que clonara el repo podía firmar filas (C6 sin necesidad de B3).
   Ahora la ignora `.gitignore`, igual que la BD generada, su `.mac` y la
   auditoría. Cada despliegue genera su propia clave con `inscribir_dataset.py`
   (Docker lo hace en el build). **La clave antigua sigue en el historial de la
   rama de pablordrz y hay que darla por comprometida**: se ha rotado.
3. **Sin enumeración de ids.** `verificar` con un id inexistente devuelve lo
   mismo que un rechazo (`{'verificado': False}`) y hace el mismo trabajo de
   comparación. El motivo solo va a la auditoría. Nuevo ataque **B5** que lo comprueba.
4. **B4 honesto** y nueva dependencia `cryptography` en `requirements.txt`.

Resultado de `python evaluar_ataques.py`: **16 de 23 ataques bloqueados**
(antes 13 de 22), y los 7 restantes son vulnerabilidades conocidas y documentadas.
## 4b. Mejora del modelo (ronda 6, sin datos nuevos)

Sin añadir fotos (las de los famosos son comunes a todos los equipos), solo
cambiando el matcher. Ajustado con `ajuste_ebgm.py`, **sin tocar el test**:

| Probado | Resultado en ajuste | Decisión |
|---|---|---|
| T-norm con la cohorte | EER 14,1 → 14,7 % | descartado |
| Pesos de Fisher por nodo (de los aumentos) | sin efecto | descartado |
| Pesos de Fisher por nodo (del Cigala, validación cruzada) | TAR 34,6 → 40–44 % | descartado: bajan los ojos (gafas de sol) y suben boca/mentón (barba). Son pesos de UNA persona |
| WCCN por nodo (de los aumentos) | TAR +1–3 puntos, ruidoso | descartado |
| WCCN con aumentos de pose | peor | descartado |
| **Mejor media cara, sin sus 3 nodos peores** | **EER 12,8 %, d' 2,14, rank-1 51 %, TAR 41,7 %** | **adoptado** (`ConfigEBGM.agregacion`) |

Test reservado (fotos nuevas del Cigala contra 1 sola foto inscrita), medido
una vez al final: **TAR a FAR 1 % 24,6 → 38,5 %**, EER 15,1 → 12,4 %, d' 1,86 →
1,99. Con el template medio, sigue en 5/5 y rank-1 100 %. FAR real 1,2 %. Nuevo
umbral 0,327. Ataques: siguen 16/23 bloqueados.

## 5. Ideas de Anguita que no se han incorporado

- **Rechazar templates idénticos bit a bit** al almacenado (P2): se evita
  añadiendo un ruido mínimo, y con el cifrado el atacante ya no obtiene el
  template de la BD. Se deja como mejora opcional.
- **Etiquetado del FBG con LBF de 68 puntos**: cambia el modelo y obligaría a
  repetir todo el ajuste de la fase 1.
- **Memoria en LaTeX con figuras**: útil para la fase 4 (exposición). Se puede
  reutilizar su estructura.
