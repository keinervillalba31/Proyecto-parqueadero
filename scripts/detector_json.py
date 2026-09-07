from pathlib import Path
import json
import cv2
import numpy as np

# Rutas del proyecto
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
imagen_path = project_dir / "imagenes" / "imagen2.jpg"

imagen = cv2.imread(str(imagen_path))

if imagen is None:
    raise FileNotFoundError(f"No se pudo abrir la imagen en: {imagen_path}")

imagen_dibujo = imagen.copy()
puntos_actuales = []
parqueaderos = {}
contador_espacios = 1


def dibujar_poligono(event, x, y, flags, param):
    global puntos_actuales, imagen_dibujo, contador_espacios, parqueaderos

    if event == cv2.EVENT_LBUTTONDOWN:
        puntos_actuales.append([int(x), int(y)])
        print(f"Punto registrado: ({x}, {y})")

        # Dibujar punto guía
        cv2.circle(imagen_dibujo, (x, y), 4, (0, 0, 255), -1)

        # Cuando se completan 4 clics
        if len(puntos_actuales) == 4:
            nombre_espacio = f"Celda_{contador_espacios}"
            parqueaderos[nombre_espacio] = puntos_actuales.copy()

            # Dibujar polígono en pantalla
            pts = np.array(puntos_actuales, np.int32)
            cv2.polylines(
                imagen_dibujo,
                [pts],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )
            cv2.putText(
                imagen_dibujo,
                nombre_espacio,
                (puntos_actuales[0][0], puntos_actuales[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            print(f"--> {nombre_espacio} guardado con éxito.")
            contador_espacios += 1
            puntos_actuales = []


cv2.namedWindow("Calibrador de Parqueaderos")
cv2.setMouseCallback("Calibrador de Parqueaderos", dibujar_poligono)

print("=" * 60)
print("INSTRUCCIONES DE CALIBRACIÓN:")
print("1. Haz clic en la ventana de la imagen para darle el foco.")
print("2. Marca los 4 puntos en orden horario (Arriba-Izq -> Arriba-Der -> Abajo-Der -> Abajo-Izq).")
print("3. Presiona 'S' para GUARDAR el archivo 'estacionamientos.json'.")
print("4. Presiona 'ESC' para salir.")
print("=" * 60)

while True:
    cv2.imshow("Calibrador de Parqueaderos", imagen_dibujo)
    key = cv2.waitKey(20) & 0xFF

    if key in [ord("s"), ord("S")]:
        if len(parqueaderos) == 0:
            print("[ADVERTENCIA] No has definido ninguna celda todavía.")
        else:
            json_output = script_dir / "estacionamientos.json"
            with open(json_output, "w", encoding="utf-8") as f:
                json.dump(parqueaderos, f, indent=4)

            print(f"\n¡ÉXITO! Archivo guardado en: {json_output.resolve()}")
            break

    elif key == 27:  # Tecla ESC
        print("\nSaliendo sin guardar...")
        break

cv2.destroyAllWindows()