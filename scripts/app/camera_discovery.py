"""Encuentra la cámara en la red local, para no depender de su IP.

Con `VIDEO_SOURCE=rtsp://admin:CLAVE@auto:554/...` el servicio pregunta por ONVIF
(WS-Discovery) qué cámaras hay en la red y usa la que responda. Si en la red hay
más de una, `CAMERA_DISCOVERY_ID` (un trozo del identificador de la cámara, que no
cambia de red a red) indica cuál es.
"""

from __future__ import annotations

import logging
import re
import socket
import time
import uuid
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger(__name__)

AUTO_HOST = "auto"
WS_DISCOVERY_ADDRESS = ("239.255.255.250", 3702)

PROBE_TEMPLATE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"'
    ' xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"'
    ' xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"'
    ' xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
    "<e:Header><w:MessageID>uuid:{message_id}</w:MessageID>"
    '<w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
    '<w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>'
    "</e:Header><e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>"
    "</e:Envelope>"
)


def uses_auto_host(source: str) -> bool:
    """True si la URL de la cámara pide buscarla en la red (`...@auto:554/...`)."""
    return "://" in source and (urlsplit(source).hostname or "").lower() == AUTO_HOST


def with_host(source: str, host: str) -> str:
    """La misma URL (usuario, clave, puerto y ruta) apuntando a otro equipo."""
    parts = urlsplit(source)
    credentials = parts.netloc.rpartition("@")[0]
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{credentials}@{host}{port}" if credentials else f"{host}{port}"
    return urlunsplit(parts._replace(netloc=netloc))


def mask_credentials(source: str) -> str:
    """Oculta la clave de una URL para los registros y el estado público."""
    if "://" not in source:
        return source
    parts = urlsplit(source)
    if parts.password is None:
        return source
    credentials, _, address = parts.netloc.rpartition("@")
    user = credentials.split(":", 1)[0]
    return urlunsplit(parts._replace(netloc=f"{user}:***@{address}"))


def local_ipv4_addresses() -> list[str]:
    """IPs de este equipo en cada tarjeta de red (sin la de loopback)."""
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        infos = []
    addresses = {info[4][0] for info in infos}
    addresses.add(lan_address())
    return sorted(address for address in addresses if not address.startswith("127."))


def lan_address(target: str = "8.8.8.8") -> str:
    """IP con la que este equipo sale a la red (no envía nada: solo elige la ruta)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((target, 80))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def _parse_reply(text: str) -> tuple[str, list[str]]:
    """(identificador de la cámara, IPs de sus direcciones ONVIF)."""
    endpoint = re.search(r"Address>\s*([^<]+?)\s*<", text)
    xaddrs = re.search(r"XAddrs>\s*([^<]+?)\s*<", text)
    hosts = []
    for url in (xaddrs.group(1).split() if xaddrs else []):
        host = urlsplit(url).hostname
        if host and host not in hosts:
            hosts.append(host)
    return (endpoint.group(1) if endpoint else ""), hosts


def discover_cameras(device_id: str = "", timeout: float = 2.0) -> list[str]:
    """IPs de las cámaras ONVIF que responden en la red local.

    La pregunta se envía por cada tarjeta de red: en Windows, con varias
    (Wi-Fi, Ethernet, adaptadores virtuales), el multicast sale solo por una.
    """
    sockets = []
    for address in local_ipv4_addresses():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((address, 0))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(address))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.sendto(PROBE_TEMPLATE.format(message_id=uuid.uuid4()).encode(), WS_DISCOVERY_ADDRESS)
            sock.setblocking(False)
            sockets.append(sock)
        except OSError:
            sock.close()

    found: list[str] = []
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline and sockets:
            for sock in sockets:
                try:
                    data, _ = sock.recvfrom(65535)
                except (BlockingIOError, OSError):
                    continue
                endpoint, hosts = _parse_reply(data.decode("utf-8", errors="replace"))
                if device_id and device_id.lower() not in endpoint.lower():
                    continue
                for host in hosts:
                    if host not in found:
                        found.append(host)
            time.sleep(0.05)
    finally:
        for sock in sockets:
            sock.close()
    if found:
        log.info("Cámaras encontradas en la red: %s", ", ".join(found))
    else:
        log.warning("No se encontró ninguna cámara ONVIF en la red")
    return found
