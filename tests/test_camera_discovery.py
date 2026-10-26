from app import camera_discovery, video_service
from app.camera_discovery import _parse_reply, mask_credentials, uses_auto_host, with_host
from test_video_source import make_service

AUTO = "rtsp://admin:CLAVE@auto:554/Streaming/Channels/101"

REPLY = (
    "<d:ProbeMatch><w:EndpointReference><w:Address>urn:uuid:7d49925b-4fc7-406b-a0ec-f47018c853e7"
    "</w:Address></w:EndpointReference>"
    "<d:XAddrs>http://192.168.1.13/onvif/device_service https://192.168.1.13/onvif/device_service </d:XAddrs>"
    "</d:ProbeMatch>"
)


def test_auto_host_is_detected_only_in_urls():
    assert uses_auto_host(AUTO)
    assert not uses_auto_host("rtsp://admin:CLAVE@192.168.1.13:554/x")
    assert not uses_auto_host("videos/auto.mp4")


def test_with_host_keeps_credentials_port_and_path():
    assert with_host(AUTO, "10.0.0.7") == "rtsp://admin:CLAVE@10.0.0.7:554/Streaming/Channels/101"
    assert with_host("rtsp://auto/stream", "10.0.0.7") == "rtsp://10.0.0.7/stream"


def test_mask_credentials_hides_only_the_password():
    assert mask_credentials(AUTO) == "rtsp://admin:***@auto:554/Streaming/Channels/101"
    assert mask_credentials("videos/respaldo.mp4") == "videos/respaldo.mp4"
    assert mask_credentials("rtsp://10.0.0.7/stream") == "rtsp://10.0.0.7/stream"


def test_parse_reply_reads_the_camera_id_and_its_ip():
    assert _parse_reply(REPLY) == ("urn:uuid:7d49925b-4fc7-406b-a0ec-f47018c853e7", ["192.168.1.13"])


def auto_service(monkeypatch, available, found):
    service = make_service(monkeypatch, available=available)
    service.settings.video_source = AUTO
    monkeypatch.setattr(video_service, "discover_cameras", lambda device_id: list(found))
    return service


def test_auto_source_opens_the_camera_found_on_the_network(monkeypatch):
    camera = with_host(AUTO, "10.0.0.7")
    service = auto_service(monkeypatch, available={camera, "respaldo.mp4"}, found=["10.0.0.9", "10.0.0.7"])

    capture = service._open_source()

    assert capture.name == camera
    assert service.camera_host == "10.0.0.7"
    assert service.state.snapshot()["fuente"]["origen"] == "rtsp://admin:***@10.0.0.7:554/Streaming/Channels/101"


def test_auto_source_follows_the_camera_when_its_ip_changes(monkeypatch):
    moved = with_host(AUTO, "192.168.0.40")
    service = auto_service(monkeypatch, available={moved, "respaldo.mp4"}, found=["192.168.0.40"])
    service.camera_host = "10.0.0.7"

    assert service._open_source().name == moved
    assert service.camera_host == "192.168.0.40"


def test_auto_source_falls_back_when_no_camera_answers(monkeypatch):
    service = auto_service(monkeypatch, available={"respaldo.mp4"}, found=[])

    assert service._open_source().name == "respaldo.mp4"
    assert service.using_fallback is True


def test_discovery_survives_a_machine_without_network(monkeypatch):
    monkeypatch.setattr(camera_discovery, "local_ipv4_addresses", lambda: [])

    assert camera_discovery.discover_cameras(timeout=0.1) == []
