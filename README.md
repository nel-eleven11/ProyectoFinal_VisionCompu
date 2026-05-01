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
