# Demo: parqueadero Archivos con la grabación del 18-nov-2026

Montaje de la demo final: el servicio de IA analiza la grabación real de la cámara del parqueadero Archivos (18 de
noviembre de 2026, de 7:40 a 18:00) como si fuera la cámara en vivo, y la plataforma muestra la ocupación de los
46 puestos (45 de carro y ARD-01, accesible) en Monitoreo, el tablero, la disponibilidad pública y el plano 3D.

## Qué hay en esta carpeta

| Archivo | Para qué |
|---|---|
| `env.demo` | Los valores del `.env` para la demo (video, ajustes del detector, conexión con el backend). Sin claves. |
| `celdas_archivos.json` | Las 46 celdas de la cámara, alineadas con el video, cada una con su puesto (AR-01…AR-45, ARD-01). |
| `referencia_celdas.jpg` | Cuadro de referencia del lote (7:41) para que la alineación no dé falsas alarmas de "la cámara se movió". |
| `cargar_demo.py` | Crea la cámara en el backend y le carga las celdas. |

## Lo que NO está en el repositorio: el video

El video pesa 15,6 GB y GitHub no acepta archivos de más de 100 MB: **se comparte por Google Drive**. Descárgalo y
guárdalo exactamente aquí (la carpeta `videos/` no se sube a git):

```
videos/parqueadero-18nov/parqueadero-18nov-dia-completo.mp4   (10,3 h a velocidad real, el principal)
videos/parqueadero-18nov/dia-completo-acelerado.mp4           (20,7 min, 1,8 GB, respaldo opcional)
```

## Pasos

1. **Backend y frontend** en las ramas que traen el plano 3D real de Archivos (`feature/correcciones-pruebas-e2e`):
   con la base nueva (`DDL_AUTO=create`) el parqueadero "Parqueadero Archivos" se carga con sus 46 puestos. Si ya tenías
   la base, sube el plano 3D de Archivos desde la plataforma marcando "crear puestos".
2. **Cuenta de servicio:** en Usuarios crea un usuario con rol Operador para la IA (p. ej. código `servicio.ia`).
3. **Video:** descárgalo de Drive a la ruta de arriba.
4. **Cámara y celdas:** con el backend encendido, desde la raíz del proyecto:
   ```
   python demo/archivos-18nov/cargar_demo.py
   ```
   Pide los datos de un administrador e imprime el id de la cámara. También copia `referencia_celdas.jpg` a la raíz.
5. **`.env`:** copia los valores de `env.demo` en tu `.env`, con la contraseña de la cuenta de servicio y el
   `BACKEND_CAMERA_ID` que imprimió el paso 4.
6. **Servicio de IA:** `python run_service.py`. En Monitoreo la cámara "Cámara Archivos (video 18-nov)" queda en línea
   y los puestos de Archivos cambian solos según el video.

## Notas

- Las celdas quedan ~15-20 px por debajo de las rayas pintadas a propósito: siguen el punto de apoyo de cada carro
  que da el detector (borde inferior del recuadro). Sin eso la mayoría de carros caía en la celda vecina.
- AR-10 está donde en el video hay un montón de arena; falta confirmarlo en el sitio.
- Solo se usa `yolo11n.pt`; un modelo más grande detectaría mejor los carros del fondo.
