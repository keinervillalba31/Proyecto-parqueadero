from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = SCRIPT_DIR.parent
IMAGE_PATH = PROJECT_DIR / "imagenes" / "imagen10.jpg"
OUTPUT_PATH = SCRIPT_DIR / "estacionamientos.json"
load_dotenv(PROJECT_DIR / ".env")
CALIBRATION_IMAGE = os.getenv("CALIBRATION_IMAGE", str(IMAGE_PATH))
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "")

points: list[list[int]] = []
spaces: dict[str, list[list[int]]] = {}
canvas: np.ndarray
base_image: np.ndarray
selected_space: str | None = None
next_space = 1


def redraw() -> None:
    global canvas

    canvas = base_image.copy()
    for name, space_points in spaces.items():
        polygon = np.array(space_points, dtype=np.int32)
        color = (0, 165, 255) if name == selected_space else (0, 255, 0)
        cv2.polylines(canvas, [polygon], True, color, 2)
        cv2.putText(
            canvas,
            name,
            (space_points[0][0], max(space_points[0][1] - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    for point in points:
        cv2.circle(canvas, tuple(point), 4, (0, 0, 255), -1)


def find_space_at(x: int, y: int) -> str | None:
    for name, space_points in reversed(list(spaces.items())):
        polygon = np.array(space_points, dtype=np.int32)
        if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
            return name
    return None


def video_dimensions() -> tuple[int, int] | None:
    if not VIDEO_SOURCE:
        return None

    source: int | str = int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else VIDEO_SOURCE
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        return None

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return (width, height) if width > 0 and height > 0 else None


def scaled_spaces(target_size: tuple[int, int] | None) -> dict[str, list[list[int]]]:
    if target_size is None:
        return spaces

    source_height, source_width = base_image.shape[:2]
    target_width, target_height = target_size
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return {
        name: [[round(x * scale_x), round(y * scale_y)] for x, y in space_points]
        for name, space_points in spaces.items()
    }


def draw_space(event: int, x: int, y: int, flags: int, param: object) -> None:
    global points, selected_space, next_space

    if event == cv2.EVENT_RBUTTONDOWN:
        selected_space = find_space_at(x, y)
        if selected_space:
            print(f"{selected_space} seleccionada. Presiona Delete para borrarla.")
        else:
            print("No hay una celda en ese punto.")
        redraw()
        return

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    points.append([int(x), int(y)])
    selected_space = None
    if len(points) != 4:
        redraw()
        return

    name = f"Celda_{next_space}"
    spaces[name] = points.copy()
    print(f"{name} guardada")
    next_space += 1
    points = []
    redraw()


def main() -> None:
    global canvas, base_image, selected_space, points, spaces, next_space
    image_path = Path(CALIBRATION_IMAGE)
    image = cv2.imread(str(image_path)) if image_path.exists() else None

    if image is None and VIDEO_SOURCE:
        source: int | str = (
            int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else VIDEO_SOURCE
        )
        capture = cv2.VideoCapture(source)
        if capture.isOpened():
            ok, image = capture.read()
        else:
            ok = False
        capture.release()
        if not ok:
            raise RuntimeError(
                f"No se pudo capturar un cuadro de VIDEO_SOURCE: {VIDEO_SOURCE}"
            )
    else:
        image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"No se pudo abrir la imagen de calibración: {image_path}"
        )

    base_image = image.copy()
    canvas = base_image.copy()
    window = "Administrador de espacios"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, draw_space)
    print(
        "Clic izquierdo: marcar puntos | clic derecho: seleccionar celda | "
        "Delete: borrar | Z: deshacer | R: limpiar puntos | S: guardar | ESC: salir"
    )

    while True:
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("s"), ord("S")):
            if not spaces:
                print("No hay espacios para guardar.")
                continue
            output_spaces = scaled_spaces(video_dimensions())
            with OUTPUT_PATH.open("w", encoding="utf-8") as file:
                json.dump(output_spaces, file, indent=4)
            print(f"Configuración guardada en {OUTPUT_PATH}")
            break
        if key in (ord("z"), ord("Z")):
            if spaces:
                removed_name, _ = spaces.popitem()
                selected_space = None
                next_space = max(
                    (int(name.split("_")[1]) for name in spaces if name.startswith("Celda_")),
                    default=0,
                ) + 1
                print(f"{removed_name} eliminada")
                redraw()
            else:
                print("No hay celdas para deshacer.")
        if key in (8, 127):
            if selected_space and selected_space in spaces:
                del spaces[selected_space]
                print(f"{selected_space} eliminada")
                selected_space = None
                redraw()
            else:
                print("Selecciona una celda con clic derecho antes de borrarla.")
        if key in (ord("r"), ord("R")):
            points = []
            selected_space = None
            redraw()
            print("Puntos sin completar eliminados.")
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
