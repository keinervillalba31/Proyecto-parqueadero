# Sistema de parqueadero

Detecta la ocupación de las celdas de un parqueadero con YOLO a partir de un video,
lee las placas con Roboflow y asigna una celda libre a los vehículos registrados.

## Responsabilidades

- `scripts/admin/configure_spaces.py`: el administrador marca las cuatro esquinas de cada celda y se guarda `estacionamientos.json`.
- `scripts/admin/probar_placas.py`: prueba solo el lector de placas con un video o una imagen.
- `scripts/app/parking_detector.py`: usa YOLO para detectar vehículos y marcar las celdas ocupadas o libres, con suavizado entre cuadros.
- `scripts/app/plate_reader.py`: envía capturas a Roboflow y confirma la placa con varias lecturas.
- `scripts/app/video_service.py`: coordina el video, el detector, el lector de placas y las reservas de celdas.
- `scripts/app/backend_client.py`: cuando una placa registrada se confirma, le pide a `educore-backend` que asigne y guarde el puesto (ver "Sincronización con el backend").
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

### 0. Celdas trazadas desde la plataforma (recomendado)

El administrador traza las celdas **desde la plataforma web**, sin tocar este servicio:

1. En **Monitoreo**, registra la cámara con la URL de transmisión `http://<este-servidor>:8000/video`.
2. En su tarjeta pulsa **Trazar celdas**: se abre una imagen limpia de la cámara (`GET /snapshot`).
   Con **+ Trazar celdas** marca las 4 esquinas de cada puesto, y a cada celda le asigna el puesto real
   del parqueadero (A-01, N-03...). Las esquinas se pueden arrastrar para ajustarlas. **Guardar celdas**.
3. En el `.env` de este servicio pon `BACKEND_SYNC_ENABLED=true` y `BACKEND_CAMERA_ID=<id de la cámara>`.

El servicio trae esas celdas del backend al arrancar y **revisa cada `BACKEND_CELLS_POLL_SECONDS` si
cambiaron**, así que ajustar o agregar celdas no requiere reiniciar nada. Las coordenadas se guardan
relativas al tamaño de la imagen, por eso siguen valiendo si cambia la resolución. `GET /estado` indica en
`fuente.celdas_origen` si se usan celdas de la `plataforma` o las `local`, y cada celda trae el
`puesto` al que está vinculada. Si la plataforma aún no tiene celdas para la cámara, se usa el archivo
local del paso siguiente.

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

Un solo comando; al encender ya queda leyendo video, detectando y sincronizando:

```powershell
python run_service.py
```

(El comando anterior, `python -m uvicorn api_video:app --app-dir scripts ...`, sigue funcionando.)

**Cámara con video de respaldo.** Pon la cámara en `VIDEO_SOURCE` (por ejemplo
`rtsp://usuario:clave@192.168.1.20/stream`) y un video corto en `VIDEO_FALLBACK`.
Si la cámara no responde, el servicio reproduce ese video en bucle sin parar y
reintenta la cámara cada `VIDEO_CAMERA_RETRY_SECONDS`; cuando responde, cambia a
ella solo. Si la cámara se cae después, vuelve al respaldo. La fuente en uso
aparece en `GET /estado` → `fuente.modo` (`principal` o `respaldo`) y en `GET /health`.

**Arranque automático en Windows** (sin abrir consola ni ejecutar nada), con el
Programador de tareas; se reinicia solo si falla:

```powershell
.\scripts\windows\servicio.ps1 instalar        # arranca al iniciar sesión
.\scripts\windows\servicio.ps1 instalar -AlEncender   # como administrador: arranca al encender el equipo
.\scripts\windows\servicio.ps1 iniciar | detener | estado | desinstalar
```

**Tiempo real.** El video y la detección corren en hilos separados: `/video` entrega
el video a su velocidad real (~24 cuadros/s) y YOLO analiza siempre el cuadro más
reciente, descartando los intermedios si no da abasto, y dibuja su último resultado
sobre cada cuadro nuevo. Así el video nunca se atrasa, aunque el análisis sea más
lento. `GET /estado` → `rendimiento` muestra `fps_video` y `fps_analisis`.

Medido en un equipo sin GPU (video de 1280x720):

| `YOLO_TILE_GRID` | Video | Análisis de YOLO |
|---|---|---|
| 2 (máxima precisión) | ~24 cuadros/s | ~1 por segundo |
| 1 | ~24 cuadros/s | ~4-5 por segundo |

Con una GPU el análisis sube a decenas por segundo y se puede usar `YOLO_TILE_GRID=2`.

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

- `GET /snapshot`: cuadro actual sin dibujos (la plataforma lo usa para trazar celdas).
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
5. Si `BACKEND_SYNC_ENABLED=true`, la misma placa confirmada se envía además a
   `educore-backend` para que quede guardada de verdad (no solo en memoria local).

## Sincronización con el backend

Además de la reserva local (en memoria, para la vista en vivo), el servicio puede
pedirle a `educore-backend` que haga y guarde la asignación real:

1. Se autentica como una cuenta de servicio (un usuario con rol **Operador**, que
   ya tiene los permisos `ASSIGNMENTS_MANAGE` y `PLATES_VIEW`).
2. Con la placa confirmada, busca el vehículo (`GET /api/vehicles/plate/{placa}`).
3. Si el vehículo está activo, busca el id del estudiante dueño por su código
   (`GET /api/assignments/students`).
4. Le pide al backend el espacio y lo guarda (`POST /api/assignments/auto`).

Esto corre en un hilo aparte (`backend-sync`): si el backend no responde o la
placa no está registrada allá, el video y la lectura de placas siguen igual —
nunca se bloquean por esto. El resultado (sincronizado, error, o la placa no
existe en el backend) queda en `GET /placas/estado` → `backend_estado`.

Para activarlo, en `.env`:

```
BACKEND_SYNC_ENABLED=true
BACKEND_BASE_URL=http://localhost:8080
BACKEND_SERVICE_USER_CODE=OP001
BACKEND_SERVICE_IDENTITY_DOCUMENT=<documento del operador>
BACKEND_SERVICE_PASSWORD=<contraseña del operador>
BACKEND_PARKING_ID=   # vacío = cualquier parqueadero con espacio libre
```

> Nota: el backend invalida la cookie CSRF en cuanto se hace cualquier petición
> GET, así que `backend_client.py` vuelve a pedirla justo antes de cada POST
> (el mismo motivo por el que el frontend también la reintenta).

### Formato de `vehiculos_registrados.json`

Las llaves son las placas en mayúsculas y sin guiones; el valor puede tener cualquier dato adicional:

```json
{
    "ABC123": {"propietario": "Nombre", "tipo": "carro"},
    "XYZ98F": {"propietario": "Nombre", "tipo": "moto"}
}
```
