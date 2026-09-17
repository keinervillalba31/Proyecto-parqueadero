# Sistema de parqueadero

## Responsabilidades

- `scripts/admin/configure_spaces.py`: administrador que marca las cuatro coordenadas de cada espacio y guarda `estacionamientos.json`.
- `scripts/app/parking_detector.py`: carga las coordenadas y usa YOLO para detectar vehículos y ocupar/liberar espacios en vivo.
- `scripts/app/plate_reader.py`: envía capturas seleccionadas a Roboflow y confirma la lectura de la placa.
- `scripts/app/video_service.py`: coordina el video, el detector de espacios y el lector de placas.
- `scripts/app/main.py`: expone FastAPI.
- `scripts/api_video.py`: punto de entrada compatible con el comando anterior.

## Ejecución

Desde `Proyecto-parqueadero`:

```powershell
..\.venv312\Scripts\python.exe -m uvicorn api_video:app --app-dir scripts --host 0.0.0.0 --port 8000
```

La configuración se carga desde `.env`.

## Endpoints

- `/video`: video anotado en vivo.
- `/placas/estado`: última lectura y estado de la placa.
- `/espacios/estado`: ocupación de espacios.
- `/estado`: estado combinado.
- `/vision/license-plates`: prueba puntual con una imagen.

## Flujo

1. El administrador configura las coordenadas de los espacios.
2. YOLO analiza cada cuadro del video y actualiza la ocupación.
3. Cuando YOLO detecta un vehículo, el lector de placas toma capturas según `PLATE_INTERVAL_FRAMES`.
4. Roboflow devuelve el texto; el servicio lo confirma con varias lecturas y lo publica en FastAPI.
