# ProyectoFinal_VisionCompu

Proyecto final de Vision por Computadora: reconocimiento de acordes/notas en guitarra o bajo usando deteccion de manos, deteccion del instrumento y seguimiento temporal en video.

## Idea

El sistema propuesto busca:

- detectar y rastrear manos/dedos del interprete en video;
- identificar la posicion de los dedos sobre el diapason;
- asociar esas posiciones con acordes/notas predefinidas;
- generar una tablatura basica;
- evaluar precision bajo distintas condiciones de grabacion.

## Tecnologias base

- **MediaPipe Hands**: deteccion de 21 landmarks por mano.
- **YOLO/Ultralytics**: deteccion por frame de mastil, diapason, cuerdas o trastes.
- **Tracking temporal**: suavizado de predicciones entre frames para evitar cambios falsos por ruido.
- **OpenCV**: lectura de video y visualizacion de resultados.

## Estructura

```text
configs/        Configuracion del pipeline
data/           Videos, anotaciones y dataset de acordes/notas
docs/           Plan de trabajo y documentacion
models/         Pesos YOLO entrenados o descargados
notebooks/      Notebook principal del prototipo
outputs/        Resultados generados
src/chordvision Modulos reutilizables del proyecto
tests/          Pruebas del nucleo sin video real
```

## Notebook inicial

Abrir:

```text
notebooks/01_skeleton_reconocimiento_acordes.ipynb
```

Ese notebook documenta el enfoque, conecta el proyecto con lo visto en clase sobre video/deteccion/tracking y deja celdas listas para cuando existan videos y pesos YOLO.

## Datos

La estrategia de datos esta documentada en `docs/data_strategy.md`.

Descargar el dataset de acordes de Hugging Face:

```bash
python scripts/download_hf_guitar_chords.py
python scripts/prepare_yolo_classification_split.py
```

Crear un manifest de los videos descargados en `data/`:

```bash
python scripts/build_video_manifest.py --video-dir data --out data/video_manifest.csv
```

Extraer frames para probar el modelo sobre videos:

```bash
python scripts/extract_video_frames.py --video-dir data --out data/video_frames
```

Entrenar un baseline YOLO de clasificacion:

```bash
yolo classify train model=yolo11n-cls.pt data=data/hf_guitar_chords_yolo epochs=20 imgsz=224
```

En Windows, si PowerShell dice que `yolo` no existe, usar la ruta completa:

```powershell
& "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\Scripts\yolo.exe" classify train model=yolo11n-cls.pt data=data/hf_guitar_chords_yolo epochs=20 imgsz=224
```

Generar videos anotados con el acorde predicho:

```bash
python scripts/annotate_videos_yolo.py --model runs/classify/chord_cls_yolo/weights/best.pt --video-dir data --out-dir outputs/annotated_videos
```

Ver una demo en pantalla mientras YOLO predice:

```powershell
python scripts\annotate_videos_yolo.py --model runs\classify\chord_cls_yolo\weights\best.pt --video-path data\139-135737102_medium.mp4 --out-dir outputs\annotated_videos --csv-dir outputs\video_predictions --predict-every 5 --show --display-scale 0.8
```

## Instalacion sugerida

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Para validar el nucleo actual:

```bash
pytest
```
