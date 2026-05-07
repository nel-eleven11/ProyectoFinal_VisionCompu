# Datos

Carpeta para videos, anotaciones y datasets de acordes/notas.

Estructura sugerida:

- `raw/`: videos originales.
- `annotations/`: etiquetas manuales por frame o por intervalo de tiempo.
- `processed/`: frames extraidos, resultados intermedios y splits.
- `hf_guitar_chords/`: dataset descargado desde Hugging Face en formato `train/val/test` para Ultralytics.
- `hf_guitar_chords_yolo/`: split estratificado usado para entrenar YOLO clasificacion sin clases faltantes en validacion/test.
- `video_manifest.csv`: inventario de videos descargados o grabados.
- `video_frames/`: frames extraidos de videos para pruebas.
- `chord_shapes.example.json`: formato inicial para acordes/notas en afinacion estandar.

Los videos reales no se suben al repo si son pesados. Dejar aqui solo muestras pequenas o instrucciones para conseguirlos.
