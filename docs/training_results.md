# Resultados de entrenamiento

## Baseline YOLO de clasificacion

Modelo base:

```text
yolo11n-cls.pt
```

Dataset:

```text
data/hf_guitar_chords_yolo
```

Comando usado:

```powershell
$yolo = "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\Scripts\yolo.exe"
& $yolo classify train model=yolo11n-cls.pt data=data/hf_guitar_chords_yolo epochs=20 imgsz=224 batch=32 workers=0 device=0 name=chord_cls_yolo exist_ok=True
```

Hardware validado:

```text
NVIDIA GeForce RTX 4060 Laptop GPU
torch 2.10.0+cu128
CUDA:0
```

Resultado final en validacion:

```text
top1_acc = 0.955
top5_acc = 1.000
```

Resultado en test:

```text
top1_acc = 0.923
top5_acc = 0.994
```

Modelo generado:

```text
runs/classify/chord_cls_yolo/weights/best.pt
```

## Prediccion sobre videos descargados

Se extrajeron frames cada 2 segundos:

```powershell
python scripts\extract_video_frames.py --video-dir data --out data\video_frames --manifest data\video_frames_manifest.csv --every-seconds 2
```

Total:

```text
54 frames
```

Prediccion:

```powershell
python scripts\predict_frames_yolo.py --model runs\classify\chord_cls_yolo\weights\best.pt --source data\video_frames --out outputs\frame_predictions.csv
```

Distribucion de predicciones sobre videos descargados:

```text
A  = 18
C  = 10
Gm = 7
G  = 7
E  = 4
Cm = 3
D  = 2
B  = 2
Dm = 1
```

Estas predicciones no deben reportarse como accuracy de videos porque los videos descargados no tienen ground truth confirmado. Sirven como prueba visual de transferencia a frames externos.

## Visualizacion del modelo en videos

Para ver el clasificador funcionando sobre los videos, se generan copias anotadas con el acorde predicho y la confianza:

```powershell
python scripts\annotate_videos_yolo.py --model runs\classify\chord_cls_yolo\weights\best.pt --video-dir data --out-dir outputs\annotated_videos
```

Para verlo mientras se procesa, se puede abrir una ventana de OpenCV:

```powershell
python scripts\annotate_videos_yolo.py --model runs\classify\chord_cls_yolo\weights\best.pt --video-path data\139-135737102_medium.mp4 --out-dir outputs\annotated_videos --csv-dir outputs\video_predictions --predict-every 5 --show --display-scale 0.8
```

En esa ventana se ve el video, el acorde predicho, la confianza y el tiempo. Con `q` o `Esc` se detiene la demo.

Salidas esperadas:

```text
outputs/annotated_videos/
outputs/video_predictions/
```

Esta visualizacion muestra el baseline de clasificacion por frame. Todavia no equivale a deteccion de cuerda/traste ni a tablatura exacta; esa parte requiere combinar MediaPipe Hands con la geometria del diapason.
