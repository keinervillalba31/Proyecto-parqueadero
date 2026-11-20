# EP14-HU099 · Evaluación de precisión del modelo de detección de ocupación

**Responsables:** Edinson Hernandez y Keiner Villalba · **Modelo:** YOLO11n (`yolo11n.pt`), confianza 0.15, imgsz 960, recortes 2×2
(la configuración con la que corre el servicio). Reproducir: `python evaluacion/evaluar_modelo.py`.

## 1. Conjunto de imágenes de prueba etiquetadas (criterio 1)
- `conjunto_prueba/`: 21 imágenes (fotogramas de `parking_simulacion.mp4` y `video-largo-parqueadero.mp4`, más fotos del parqueadero de Archivos y de otro con motos).
- `etiquetas.json` (congeladas, no dependen del modelo): **4 imágenes con vehículos** (45 vehículos etiquetados: carros y motos) y
  **6 imágenes del parqueadero vacío** (verdad: 0 vehículos).
- Las zonas con carros tan pegados que no se distinguen (columna central, fila izquierda en diagonal, copas de árboles) se marcan *ignorar*:
  no cuentan a favor ni en contra. Acierto = IoU ≥ 0.5 con una caja etiquetada.

## 2. Métricas (criterios 2 y 3)
| Condición (simulada sobre las mismas imágenes) | TP | FP | FN | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|
| Original | 45 | 10 | 0 | 0.818 | 1.000 | 0.900 |
| Sobreexpuesto (sol fuerte) | 43 | 12 | 2 | 0.782 | 0.956 | 0.860 |
| Atardecer (−50 % brillo) | 44 | 13 | 1 | 0.772 | 0.978 | 0.863 |
| Poca luz (−75 % + ruido) | 40 | 7 | 5 | 0.851 | 0.889 | 0.870 |
| Noche (−88 % + ruido) | 36 | 3 | 9 | 0.923 | 0.800 | 0.857 |
| Niebla (bajo contraste) | 42 | 10 | 3 | 0.808 | 0.933 | 0.866 |
| Compresión / ruido | 42 | 15 | 3 | 0.737 | 0.933 | 0.833 |

Lectura: con poca luz el modelo **deja de ver** vehículos (recall 1.00 → 0.80, sobre todo motos y carros oscuros); con luz fuerte, niebla o
compresión **inventa** detecciones (precisión baja a 0.74–0.78).

## 3. Casos en que falla (criterio 4) — imágenes en `resultados/fallos/` (rojo = no detectado, naranja = falso positivo)
1. **Arcada del edificio del fondo** detectada como vehículo (3 de los 4 falsos positivos de imagen3 y los de los parqueaderos vacíos). Está fuera de las
   celdas, así que **no cambia el conteo de puestos**: el sistema solo cuenta vehículos dentro de una celda.
2. **Motos** en la noche/poca luz (imagen2: 5 de 10 vehículos perdidos) y en grupo (cajas que se pisan).
3. **Carros oscuros / lejanos** bajo sombra de los árboles (largo_00900: se pierden 3 en poca luz).
4. Falsos positivos puntuales en árboles/peatones (largo_00900) y más con compresión fuerte o sol directo.
5. **Carros pegados** en la columna central: no se pueden separar ni a ojo; es el caso más difícil y queda sin medir (ver límites).

## 4. Límites de esta evaluación (decirlo en la tesis)
- Las condiciones de iluminación son **simuladas** (gamma, ruido, niebla, compresión), no hay fotos reales de noche ni lluvia.
- Las cajas verdad salen de revisar a mano las detecciones del modelo y añadir las que faltaban; por eso el recall del caso *original* (1.00) es optimista y
  sirve sobre todo para comparar condiciones. Lo que sí es independiente son los **falsos positivos** y los 6 parqueaderos vacíos.
- Solo 45 vehículos: los porcentajes tienen mucha incertidumbre. Falta medir la **ocupación por puesto** (celda) con un conjunto etiquetado por el equipo.

## 5. Mejoras propuestas
1. Contar solo vehículos con el punto de apoyo dentro de una celda (ya se hace) y mostrar la zona fuera de celdas como ignorada en el recorte.
2. Subir a `yolo11s`/`yolo11m` si hay GPU (el nano es el que pierde motos y carros oscuros).
3. Tomar fotos reales de atardecer, noche y lluvia con la cámara definitiva y volver a correr este script (basta añadirlas a `conjunto_prueba/` y a `etiquetas.json`).
4. Ajustar la confianza por condición o por tamaño (más alta para cajas grandes, más baja para motos) y medirlo con este mismo script.
5. Hacer *fine-tuning* con imágenes propias del parqueadero (carros apilados, motos) y en modo nocturno; en la noche, cámara con IR.
6. El suavizado temporal (2 s para ocupar, 8 s para liberar) ya absorbe parte de los falsos negativos aislados; evaluarlo con secuencias etiquetadas.
