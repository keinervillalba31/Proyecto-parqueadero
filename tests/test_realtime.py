import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

from app.state import RuntimeState
from app.video_service import VideoService

CAR = 2
CAR_IN_CELL_1 = (CAR, 0.9, (30, 50, 70, 90))


def blank_frame() -> np.ndarray:
    return np.zeros((120, 220, 3), dtype=np.uint8)


def make_service(detector, detections=()) -> VideoService:
    detector.vehicles._detect_with_tiles = lambda frame: list(detections)
    service = VideoService.__new__(VideoService)
    service.settings = SimpleNamespace(
        reservation_timeout_seconds=600,
        plate_require_vehicle=True,
        plate_interval_frames=10_000,
        plate_video_source="",
        plate_crop_vehicles=True,
        plate_max_vehicles=3,
        plate_min_crop_width=1,
        plate_skip_parked=False,
    )
    service.parking = detector
    service.state = RuntimeState(total_spaces=2)
    service.reservations = {}
    service.reservations_lock = threading.Lock()
    service.frame_lock = threading.Lock()
    service.plate_frame_lock = threading.Lock()
    service.latest_plate_frame = None
    service.latest_plate_frame_id = 0
    service.overlay = None
    service.latest_frame = None
    service.latest_frame_id = 0
    service.plate_thread = None
    service.vehicle_was_detected = True  # sin "recién llegó": no se lanza lectura de placas
    service.frame_number = 0
    service.detector_failed = False
    service.analysis_seconds = 0.0
    service.next_frame_at = 0.0
    service.video_frames = 0
    service.fps_window_start = time.monotonic()
    return service


def test_analyze_does_not_touch_the_frame(make_detector):
    detector = make_detector()
    detector.vehicles._detect_with_tiles = lambda frame: [CAR_IN_CELL_1]
    frame = blank_frame()

    state, vehicles = detector.analyze(frame)

    assert not frame.any(), "analyze no debe dibujar sobre el cuadro"
    assert state["celdas"]["Celda_1"]["ocupado"] is True
    assert len(vehicles) == 1


def test_draw_paints_the_last_result_on_a_new_frame(make_detector):
    detector = make_detector()
    detector.vehicles._detect_with_tiles = lambda frame: [CAR_IN_CELL_1]
    state, vehicles = detector.analyze(blank_frame())

    fresh_frame = blank_frame()
    detector.draw(fresh_frame, vehicles, state)

    assert fresh_frame.any(), "el resultado de YOLO se dibuja sobre cualquier cuadro nuevo"


def test_process_detection_publishes_state_and_overlay(make_detector):
    service = make_service(make_detector(), detections=[CAR_IN_CELL_1])

    service._process_detection(blank_frame())

    snapshot = service.state.snapshot()
    assert snapshot["celdas"]["Celda_1"]["ocupado"] is True
    assert service.overlay is not None
    assert snapshot["rendimiento"]["fps_analisis"] > 0


def test_publish_frame_streams_original_without_cell_overlay(make_detector):
    service = make_service(make_detector(), detections=[CAR_IN_CELL_1])
    service.overlay = ([CAR_IN_CELL_1], {"celdas": {}}, {})
    frame = blank_frame()

    service._publish_frame(frame)

    _, encoded = service.state.get_frame()
    published = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(published, frame)


def test_publish_plate_frame_is_independent_from_parking_stream(make_detector):
    service = make_service(make_detector())
    plate_frame = np.full((120, 220, 3), 255, dtype=np.uint8)

    service._publish_plate_frame(plate_frame)

    plate_id, plate_encoded = service.get_plate_frame()
    parking_id, parking_encoded = service.state.get_frame()
    assert plate_id == 1
    assert plate_encoded is not None
    assert parking_id == 0
    assert parking_encoded is None


def test_process_detection_clears_a_previous_detector_error(make_detector):
    service = make_service(make_detector())
    service.state.worker_error = "falló antes"
    service.detector_failed = True

    service._process_detection(blank_frame())

    assert service.state.worker_error is None


def test_pace_holds_a_file_to_its_real_speed(make_detector):
    service = make_service(make_detector())
    interval = 0.05

    started = time.monotonic()
    for _ in range(5):
        service._pace(interval)
    elapsed = time.monotonic() - started

    # El primer cuadro sale de inmediato y los otros cuatro esperan su turno.
    assert 0.15 <= elapsed < 0.4


def test_pace_does_not_slow_down_a_live_camera(make_detector):
    service = make_service(make_detector())

    started = time.monotonic()
    for _ in range(50):
        service._pace(0.0)

    assert time.monotonic() - started < 0.1
