# Sistema de parqueadero

Detecta la ocupación de las celdas de un parqueadero con YOLO a partir de un video,
lee las placas con Roboflow y asigna una celda libre a los vehículos registrados.

## Responsabilidades

- `scripts/admin/configure_spaces.py`: el administrador marca las cuatro esquinas de cada celda y se guarda `estacionamientos.json`.
- `scripts/admin/probar_placas.py`: prueba solo el lector de placas con un video o una imagen.
- `scripts/app/parking_detector.py`: usa YOLO para detectar vehículos y marcar las celdas ocupadas o libres, con suavizado entre cuadros.
- `scripts/app/plate_reader.py`: envía capturas a Roboflow y confirma la placa con varias lecturas.
- `scripts/app/video_service.py`: coordina el video, el detector, el lector de placas y las reservas de celdas.
- `scripts/app/main.py`: expone la API con FastAPI.
- `scripts/api_video.py`: punto de entrada compatible con el comando anterior.

## Instalación

Requiere **Python 3.12** (`inference-sdk` todavía no es compatible con versiones más nuevas).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # y completar ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW
```

Todas las rutas relativas del `.env` se toman desde la raíz del proyecto.

## Uso

Todos los comandos se ejecutan desde `Proyecto-parqueadero`.

### 1. Configurar las celdas

```powershell
python scripts/admin/configure_spaces.py
```

Se abre `CALIBRATION_IMAGE`, o un cuadro de `VIDEO_SOURCE` si esa imagen no existe.
Puedes usar una imagen más nítida y más grande que el video, siempre que tenga
**el mismo encuadre**. Al guardar, las coordenadas se re-escalan al tamaño del
video, y si la proporción no coincide se muestra una advertencia.

| Acción | Tecla |
|---|---|
| Marcar esquina | Clic izquierdo (4 clics = 1 celda) |
| Seleccionar celda | Clic derecho |
| Borrar celda seleccionada | Supr o Backspace |
| Deshacer última celda | Z |
| Borrar puntos sin terminar | R |
| Guardar y salir | S |
| Salir sin guardar | ESC |

### 2. Ejecutar el servicio

```powershell
python -m uvicorn api_video:app --app-dir scripts --host 0.0.0.0 --port 8000
```

Luego abre http://localhost:8000/video para ver el video anotado o http://localhost:8000/docs para probar la API.
Si `VIDEO_SOURCE` es un archivo, el video se repite en bucle.

### 3. Probar el lector de placas

```powershell
python scripts/admin/probar_placas.py videos/video-placa-prueba.mp4 --cada 5 --max-lecturas 40 --guardar salida_placas
python scripts/admin/probar_placas.py imagenes/imagen1.jpg
```

Por defecto se recortan los vehículos con YOLO y cada recorte se envía por separado.
Cada recorte enviado es una llamada a la API de Roboflow.

| Opción | Uso |
|---|---|
| `--guardar carpeta` | Guarda las imágenes enviadas, con la placa leída en el nombre |
| `--cuadro-completo` | Envía el cuadro entero, sin recortar (para comparar) |
| `--patron "..."` | Cambia el formato de placa válido; `--patron ""` acepta todo |

La salida muestra, por cada vehículo, las placas válidas y los textos descartados
(letreros, señales, etc.).

### 4. Pruebas automatizadas

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

Las pruebas simulan YOLO, así que no necesitan GPU ni video.

## Endpoints

- `GET /video`: video anotado en vivo (verde = libre, rojo = ocupado, amarillo = reservado).
- `GET /espacios/estado`: ocupación de las celdas y reservas.
- `GET /placas/estado`: última lectura y estado de la placa.
- `GET /reservas`: celdas asignadas a placas registradas.
- `GET /estado`: estado combinado. También se publica cada 0.5 s por WebSocket en `/ws/estado`.
- `GET /health`: indica si el hilo de video funciona.
- `POST /vision/license-plates`: prueba puntual con una imagen (máximo 10 MB).

## Flujo

1. El administrador configura las coordenadas de las celdas.
2. YOLO analiza cada cuadro. Una celda está ocupada si el punto inferior central de la
   caja de un vehículo cae dentro de ella. El cambio de estado solo se acepta después de
   `OCCUPANCY_FRAMES` cuadros seguidos, para que no parpadee.
3. Cuando hay un vehículo, cada `PLATE_INTERVAL_FRAMES` cuadros se recortan los vehículos
   más cercanos que no estén estacionados y se envían a Roboflow.
   - Se descarta el texto que no tenga formato de placa (`PLATE_PATTERN`) y se corrigen
     confusiones típicas del OCR (O/0, I/1, B/8, S/5...).
   - La placa se confirma con `PLATE_CONFIRMATION_READS` lecturas parecidas: las que
     difieren en un carácter cuentan como la misma placa.
   - Una placa confirmada no se vuelve a confirmar durante `PLATE_COOLDOWN_SECONDS`,
     pero otros vehículos sí se siguen leyendo.
4. Si la placa está en `vehiculos_registrados.json`, se le reserva la primera celda libre.
   La reserva se libera cuando el vehículo se estaciona y luego sale, o después de
   `RESERVATION_TIMEOUT_SECONDS` si nunca llegó a ocuparla.

### Formato de `vehiculos_registrados.json`

Las llaves son las placas en mayúsculas y sin guiones; el valor puede tener cualquier dato adicional:

```json
{
    "ABC123": {"propietario": "Nombre", "tipo": "carro"},
    "XYZ98F": {"propietario": "Nombre", "tipo": "moto"}
}
```
