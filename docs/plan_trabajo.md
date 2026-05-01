# Plan de trabajo

## 1. Prototipo sin video real

- Definir formato de acordes/notas.
- Probar conversion de trastes a notas con afinacion estandar.
- Generar tablatura a partir de eventos sinteticos.

## 2. Deteccion por frame

- Usar MediaPipe Hands para obtener 21 landmarks por mano.
- Usar YOLO/Ultralytics para detectar mastil, diapason, cuerdas o trastes.
- Calibrar la relacion pixel -> cuerda/traste.

## 3. Tracking temporal

- Suavizar detecciones entre frames.
- Evitar cambios bruscos de acorde por un solo frame malo.
- Consolidar intervalos estables para generar tablatura.

## 4. Evaluacion

- Crear videos en condiciones distintas: buena luz, baja luz, angulo lateral, movimiento rapido y oclusion parcial.
- Etiquetar segmentos con acorde/nota real.
- Medir accuracy por acorde, precision por condicion y errores de tablatura.
