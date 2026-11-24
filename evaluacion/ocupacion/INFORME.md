# Evaluación de la ocupación por puesto · parqueadero Archivos (video del 18-nov-2026)

Reproducir: `python evaluacion/evaluar_ocupacion.py correr` (≈ 14 min en CPU) y `python evaluacion/evaluar_ocupacion.py medir`.
Probar otras reglas de ocupación sin correr YOLO: `python evaluacion/evaluar_ocupacion.py repasar --subir 0.08 --liberar-cerca 8`.

**Resultado actual** (yolo11n, zona lejana ampliada, punto de apoyo subido, celdas cercanas liberadas a los 8 s):
precisión **0,979**, recall **0,940**, F1 **0,959**. Carros 0,992 / 0,992; motos 0,871 / 0,628. Ver "Mejora 2".

## Resumen: antes y después de la mejora para motos (24-nov)
Las mismas 16 muestras, las mismas etiquetas y el suavizado de 30 s.
- **Antes:** el detector como estaba.
- **Después:** zona lejana ampliada más el filtro de duplicados ajustado (ver "Mejora 1").

| | Antes P | Antes R | Después P | Después R |
|---|---|---|---|---|
| **Todas las celdas** | 0,963 | 0,874 | 0,953 | **0,940** |
| Puestos de carro | 0,969 | 0,969 | 0,962 | **0,992** |
| Puestos de moto | 0,867 | 0,302 | 0,871 | **0,628** |
| Zona del fondo | 0,966 | 0,816 | 0,957 | **0,943** |

- P = precisión: de lo que el sistema marca ocupado, cuánto lo está de verdad.
- R = recall: de lo que está ocupado de verdad, cuánto ve el sistema.
- F1 total: de 0,916 a 0,946. Falsos negativos: de 38 a 18. Falsos positivos: de 10 a 14.
- Matriz actual en `matriz_confusion.png`; la de antes, en `antes/`.

### Mejora 1: motos
1. **Segunda pasada ampliada sobre las celdas lejanas** (`far_zone`, `ZOOM_FAR_CELLS=true`).
   - Cuáles son: las celdas de menos del 40 % del área de la más grande. Sale solo de la geometría de las celdas, así
     que no depende de esta cámara; con una cámara cenital no hay zona.
   - Ese recuadro (aquí 642×256 px) se vuelve a pasar por YOLO, que lo amplía unas 2 veces.
   - Se descartan las cajas que tocan su borde, que son vehículos cortados que ya ve el cuadro completo.
   - Costo: una inferencia más por análisis.
2. **Filtro de cajas duplicadas ajustado.**
   - Antes se quitaba toda caja que quedara un 70 % dentro de otra. Eso también quitaba carros de verdad: en una fila
     vista en perspectiva, el de atrás queda casi dentro de la caja del de adelante.
   - Ahora el 70 % aplica solo a las cajas de recortes. A las del cuadro completo se les exige un 90 %, para seguir
     quitando la caja extra que YOLO a veces da de la parte de arriba del mismo carro.
   - Sin ningún filtro, el recall de carros llega a 0,996, pero los duplicados suben los falsos positivos a 19
     (precisión total 0,938).

Probado sin mejora (banco de cuadros sueltos): contar las bicicletas como motos, bajar la confianza de las motos a
0,05, ampliar también la zona de motos del frente, y partir todo el lote en 2×2. Ninguna ayudó, y la última empeoró.

**Lo que falta en motos:** casi todo es PM-03, una moto oscura junto al poste, contra la arcada en sombra; tampoco se
ve ampliada. En algunas muestras PM-03 es en realidad el carro plateado montado entre AR-01 y PM-03. El siguiente paso
sería un modelo más grande (`yolo11s`) o entrenar con recortes de motos de esta cámara.

**Los falsos positivos que quedan** son sobre todo los que sostiene el suavizado de 30 s (AR-07, AR-10, AR-35: un
carro que pasa o gente) y el punto de apoyo que se pasa a la celda vecina (PM-02 por la camioneta de AR-01). Es la
mejora 2 pendiente: ocupar solo si el vehículo está quieto y liberar antes.

## Mejora 2: falsos positivos (24-nov, tarde)
Mismas 16 muestras y etiquetas. Las reglas se ajustaron con `evaluar_ocupacion.py repasar`, que vuelve a decidir la
ocupación con las detecciones guardadas por `correr` (`detecciones.json`), sin correr YOLO.

| | FP | FN | Precisión | Recall | F1 |
|---|---|---|---|---|---|
| Después de la mejora 1 | 14 | 18 | 0,953 | 0,940 | 0,946 |
| **+ punto de apoyo subido y liberar antes cerca** | **6** | **18** | **0,979** | **0,940** | **0,959** |
| Puestos de carro | 2 (antes 10) | 2 | 0,992 | 0,992 | |
| Puestos de moto | 4 | 16 | 0,871 | 0,628 | |

Reglas probadas, cada una por separado:
- **Subir el punto de apoyo un 8 % de la altura de la caja** (`FOOTPOINT_LIFT=0.08`). La caja incluye sombra y
  defensa; subido, el punto ya no cae en el puesto de adelante. FP de 14 a 8 sin perder ningún puesto. Con
  0,04 a 0,16 el resultado es parecido.
- **Liberar las celdas cercanas tras 8 s** (`OCCUPANCY_FREE_SECONDS_NEAR=8`). Las lejanas siguen con 30 s, porque
  ahí el detector pierde a ratos las motos. Con 5 s hay un FP menos; se dejó 8 s como margen, porque se ajustó sobre
  estas mismas 16 muestras.
- **Un vehículo, un puesto:** si el punto cae en el borde entre dos celdas, cuenta solo en la de centro más cercano.
- **Ocupar solo si el vehículo está quieto** (`OCCUPANCY_STILL_SECONDS`): **desactivado**. Con 3 s baja los FP a 6,
  pero pierde 10 puestos de verdad, sobre todo motos del fondo, cuya caja salta demasiado de un cuadro a otro para
  parecer quieta. Con más tolerancia los FP vuelven a subir; ninguna combinación superó al F1 sin esta regla.

La ganancia se mantiene en las dos mitades del día: mañana FP de 3 a 0, tarde de 11 a 6, con los mismos FN.
Quedan: AR-35 en la muestra 8 (un carro que pasa), PM-02 en dos muestras (sigue siendo la camioneta de AR-01) y
motos sueltas.

## Modelo ajustado con imágenes de esta cámara (24-nov)
`entrenamiento/motos/`: yolo11n ajustado 30 épocas en CPU con 600 imágenes de la mañana (lejos de las muestras de
esta evaluación), etiquetadas por un modelo grande (`yolo11x`) en el cuadro completo y en la zona lejana ampliada.
Resultados en `modelo-entrenado/`, con las mismas reglas de la mejora 2.

| | FP | FN | Precisión | Recall | F1 |
|---|---|---|---|---|---|
| yolo11n original | 6 | 18 | 0,979 | 0,940 | **0,959** |
| yolo11n ajustado | 9 | 16 | 0,969 | 0,947 | 0,958 |
| Motos: original | 4 | 16 | 0,871 | 0,628 | |
| Motos: ajustado | 8 | 14 | 0,784 | 0,674 | |

Por período:
- **Mañana** (el período de entrenamiento): FN de 12 a 10, pero FP de 0 a 3.
- **Tarde** (que el modelo no vio): **idéntico**, 6 FP y 6 FN.

**Conclusión: no mejora y se sigue usando el original.** Las 2 motos más que ve son de la mañana, el período que
conoce, y en la tarde no gana nada: aprendió la escena, no a ver mejor las motos. Además marca motos donde no hay
(PM-02, PM-04). PM-03, la moto oscura junto al poste, la sigue perdiendo siempre: si el modelo grande tampoco la ve,
entrenar con sus etiquetas no puede enseñársela. Para mejorar ahí haría falta etiquetar motos a mano, más días de
video o una mejor posición o iluminación de la cámara.

*(Lo que sigue es la evaluación original, antes de las mejoras.)*

## Cómo se midió
- **Video:** `parqueadero-18nov-dia-completo.mp4` (cámara fija de Archivos, 1280×720, 7:40 a 18:00).
- **Muestras:** 16 cuadros repartidos en todo el día, cada ~41 min.
- **Detector:** el mismo que corre en el servicio: `yolo11n`, confianza 0,10, imgsz 1280, sin recortes, alineación con
  la referencia. Para cada muestra se analizan los 45 s anteriores, un cuadro por segundo, para que el suavizado actúe
  como en vivo (ocupa con 1 s viendo el vehículo y libera tras 30 s sin verlo).
- **Celdas:** las 46 que había en la plataforma el 24-nov (`celdas.json`), trazadas por filas.
- **Verdad:** etiquetada a mano celda por celda (`etiquetas.json`) sobre imágenes sin la predicción.
  - Un puesto está ocupado si un vehículo estacionado apoya en él sus ruedas del lado de la cámara: la carrocería tapa
    visualmente el puesto de atrás. Se comprobó con el lote casi vacío (muestra 7) que las celdas coinciden con las
    rayas pintadas (±5 px).
  - Una moto en un puesto de carro cuenta como ocupado.
  - Los casos que no se pueden decidir a ojo van como *dudosas* y no cuentan. Son 100 de 736 celdas-muestra
    (carro tapado por otro, motos en sombra, vehículo justo en el borde entre dos celdas).
  - Quedan **636 celdas-muestra evaluadas**.

## Resultados (con suavizado, como en vivo)
| | TP | FP | FN | TN | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **Todas** | 263 | 10 | 38 | 325 | **0,963** | **0,874** | 0,916 |
| Puestos de carro | 250 | 8 | 8 | 277 | 0,969 | 0,969 | 0,969 |
| Puestos de moto (PM-*, AR-38, AR-39) | 13 | 2 | 30 | 48 | 0,867 | **0,302** | 0,448 |
| Zona del fondo | 115 | 4 | 26 | 126 | 0,966 | 0,816 | |
| Zona media | 86 | 1 | 2 | 99 | 0,989 | 0,977 | |
| Zona del frente | 62 | 5 | 10 | 100 | 0,925 | 0,861 | |

Exactitud total: 0,925. Matriz de confusión en `matriz_confusion.png`.

Sin suavizado (solo el cuadro de la muestra): precisión 0,978 y recall 0,734. El suavizado recupera 42 puestos que se
perdían en un cuadro suelto, a cambio de 5 falsos positivos más.

## Lectura
1. **El problema real del fondo son las motos que no se ven, no los falsos positivos.**
   - 30 de los 38 falsos negativos son motos: PM-03 junto a la columna del edificio, PM-05, PM-06 y las motos de AR-38
     y AR-39.
   - En la fila del fondo una moto mide unos 15×20 px y el modelo nano casi no la detecta: en el cuadro solo, el
     recall de motos es 0,14.
   - En carros el sistema acierta el 97 % en ambos sentidos.
2. **Los falsos positivos son pocos (10 de 273 marcados) y tienen dos causas:**
   - **El suavizado los sostiene.** Algo que se vio al menos 1 s (un carro que pasa por el pasillo, gente caminando en
     el fondo) deja la celda "ocupada" 30 s más. En AR-35 (muestra 8), AR-07 y AR-10 no había ningún vehículo en el
     cuadro y la celda seguía ocupada. Así se explica que "a veces" aparezca un ocupado en el fondo y luego se libere:
     en la tarde (muestras 7 a 9) pasa mucha gente por la fila del fondo.
   - **El punto de apoyo se pasa a la celda vecina.** La caja de un carro grande incluye sombra y defensa, y su borde
     inferior cae en la celda de adelante. Ejemplo: la camioneta de AR-01 marca PM-02.
3. En la zona media casi no hay errores (precisión 0,99, recall 0,98).

## Mejoras propuestas (por orden de impacto)
1. **Motos:**
   - Cambiar a un modelo más grande (`yolo11s`/`m`), o hacer *fine-tuning* con recortes de motos de esta cámara.
   - Confianza y área mínima propias para la clase moto.
   - Para la fila del fondo, recortes ampliados de esa franja (*tiles* solo sobre esa zona).
2. **Falsos positivos sostenidos:**
   - Ocupar solo si el vehículo está **quieto** varios segundos (el carro que pasa por el pasillo no se detiene).
   - Bajar `OCCUPANCY_FREE_SECONDS` de 30 a ~10 s en las celdas cercanas, donde el detector no pierde los carros.
   - Medir ambos cambios con este mismo script.
3. **Punto de apoyo:** usar el borde inferior de la silueta (segmentación) en vez de la caja, o subir el punto unos
   píxeles según el tamaño de la caja.

## Límites
- Un solo día y una sola cámara; 16 muestras (636 celdas-muestra). Hay pocos casos de moto (43 ocupados), así que su
  porcentaje tiene mucha incertidumbre.
- Las *dudosas* (14 %) salen de la evaluación. Son justo los casos difíciles (carros tapados por otros), así que las
  cifras reales en un lote lleno pueden ser algo peores.
- La verdad la etiquetó una sola persona: no hay medida de acuerdo entre etiquetadores.
