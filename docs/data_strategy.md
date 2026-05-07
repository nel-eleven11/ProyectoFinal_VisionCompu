# Estrategia de datos

## Hugging Face

Dataset: `dduka/guitar-chords`

Este dataset trae imagenes de guitarra con una etiqueta de acorde. Las clases visibles son:

`A`, `Am`, `B`, `Bm`, `C`, `Cm`, `D`, `Dm`, `E`, `Em`, `F`, `Fm`, `G`, `Gm`.

La API de Hugging Face muestra que los splits son:

- `train`: 2719 imagenes
- `validation`: 290 imagenes
- `test`: 142 imagenes

Este dataset sirve para un primer baseline de clasificacion visual de acordes. No trae cajas de `diapason`, `cuerda` o `traste`, por lo que no entrena directamente un detector YOLO de partes del instrumento. Si se usa YOLO con este dataset, el enfoque correcto es YOLO en modo clasificacion.

El script local descarga los splits en formato compatible con Ultralytics:

- `data/hf_guitar_chords/train/<clase>/*.jpg`
- `data/hf_guitar_chords/val/<clase>/*.jpg`
- `data/hf_guitar_chords/test/<clase>/*.jpg`

Como el split original de Hugging Face no trae todas las clases en validacion/test, tambien se genera un split estratificado para YOLO en:

- `data/hf_guitar_chords_yolo/train/<clase>/*.jpg`
- `data/hf_guitar_chords_yolo/val/<clase>/*.jpg`
- `data/hf_guitar_chords_yolo/test/<clase>/*.jpg`

## Videos descargados

Los videos descargados de internet sirven como prueba de robustez visual:

- deteccion de mano;
- visibilidad del diapason;
- cambios de angulo;
- movimiento;
- oclusiones parciales.

No deben usarse como verdad absoluta de precision de acordes si no se anotan manualmente. Muchos videos no tienen audio o no dejan claro que acorde se toca, pero eso no invalida el proyecto porque el metodo principal es visual.

## Videos propios

Para metricas finales se recomiendan videos propios cortos, con acordes conocidos y anotados:

- 3-5 clips por acorde principal;
- buena luz y baja luz;
- camara fija y camara con angulo;
- cambios lentos y cambios rapidos.

Estos videos son los que permiten reportar accuracy real por acorde o nota.

## Guitarras distintas

No afecta que haya guitarras distintas si el objetivo es reconocer patrones visuales generales de mano y diapason. De hecho, ayuda a que el modelo no dependa de una sola guitarra.

La precaucion es mantener claro el alcance:

- primer alcance: guitarra en afinacion estandar;
- videos donde se vea mano + diapason con suficiente claridad;
- evitar mezclar bajo con guitarra dentro del mismo modelo de acordes si no se separa por instrumento;
- no prometer precision de traste/cuerda si el dataset usado solo trae etiqueta de acorde.

## Comandos

Descargar el dataset completo:

```bash
python scripts/download_hf_guitar_chords.py
python scripts/prepare_yolo_classification_split.py
```

Descargar una muestra rapida por split:

```bash
python scripts/download_hf_guitar_chords.py --max-per-split 25
```

Crear manifest de videos locales:

```bash
python scripts/build_video_manifest.py --video-dir data --out data/video_manifest.csv
```

Entrenar un baseline de clasificacion con Ultralytics:

```bash
yolo classify train model=yolo11n-cls.pt data=data/hf_guitar_chords_yolo epochs=20 imgsz=224
```

En Windows puede pasar que `yolo.exe` se instale pero no quede en el `PATH`. En ese caso:

```powershell
& "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\Scripts\yolo.exe" classify train model=yolo11n-cls.pt data=data/hf_guitar_chords_yolo epochs=20 imgsz=224
```

Probar el modelo sobre frames extraidos de los videos descargados:

```bash
python scripts/extract_video_frames.py --video-dir data --out data/video_frames
python scripts/predict_frames_yolo.py --model runs/classify/train/weights/best.pt --source data/video_frames
```
