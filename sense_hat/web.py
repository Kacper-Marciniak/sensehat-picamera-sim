"""Web UI for the Sense HAT simulator (standard library only).

``SenseHat()`` starts this automatically; you normally never import it.

    GET  /              the page (sense_hat_ui.html, next to this file)
    GET  /events        Server-Sent Events: ``frame`` (LED image), ``state``
                        (simulation values + what your code would read) and
                        ``stick`` (joystick events)
    GET  /api/state     the same state as JSON
    POST /api/sim       partial update of ``hat.sim`` (JSON)
    POST /api/light     colour sensor light level: {red, green, blue, clear}
    POST /api/stick     {"direction": "up", "action": "pressed"}
    POST /api/reset     restore simulated inputs to their defaults

The server only talks to the hat through its public simulator hooks, so it
can be removed without touching the rest of the simulator. By default it
listens on 127.0.0.1 only; POSTs from other origins are rejected.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

__all__ = ["WebUI", "build_state"]

_HTML_PATH = Path(__file__).with_name("sense_hat_ui.html")
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}
_MAX_BODY = 64 * 1024
_STATE_INTERVAL = 0.15  # seconds between ``state`` events


def _round(value: Any, digits: int = 3) -> Any:
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def build_state(hat) -> dict[str, Any]:
    """Everything the page needs: the simulated world and what code would read."""
    colour = hat.colour
    compass, gyro, accel = hat._imu_config
    return {
        "sim": _round(hat.sim.to_dict(), 6),
        "light": _round(colour.light, 3),
        "hat": {
            "rotation": hat.rotation,
            "low_light": hat.low_light,
            "imu": {"compass": compass, "gyro": gyro, "accel": accel},
        },
        "readouts": _round(
            {
                "temperature": hat.get_temperature(),
                "humidity": hat.get_humidity(),
                "pressure": hat.get_pressure(),
                "orientation": hat.get_orientation_degrees(),
                "accelerometer": hat.get_accelerometer(),
                "gyroscope": hat.get_gyroscope(),
                "accelerometer_raw": hat.get_accelerometer_raw(),
                "gyroscope_raw": hat.get_gyroscope_raw(),
                "compass": hat.get_compass(),
                "compass_raw": hat.get_compass_raw(),
                "colour": {
                    "colour": list(colour.colour),
                    "colour_raw": list(colour.colour_raw),
                    "gain": colour.gain,
                    "integration_cycles": colour.integration_cycles,
                    "integration_time": colour.integration_time,
                },
            }
        ),
    }


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR would let a second process bind the same port.
    allow_reuse_address = os.name != "nt"
    ui: "WebUI"


class _Server6(_Server):
    address_family = socket.AF_INET6


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    server_version = "SenseHatSim"

    def log_message(self, format: str, *args: Any) -> None:  # keep the console clean
        pass

    @property
    def hat(self):
        return self.server.ui.hat

    # -- plumbing -------------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, _dumps(payload).encode(), "application/json")

    def _host_ok(self) -> bool:
        """Reject DNS-rebinding attempts when bound to loopback."""
        if not self.server.ui.loopback_only:
            return True
        hostname = urlsplit("//" + (self.headers.get("Host") or "")).hostname
        return hostname in _LOOPBACK

    # -- GET ------------------------------------------------------------------

    def do_GET(self) -> None:
        if not self._host_ok():
            self._json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "bad Host header"})
            return
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            try:
                page = _HTML_PATH.read_bytes()
            except OSError:
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, b"sense_hat_ui.html not found next to sense_hat_web.py", "text/plain")
                return
            self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(HTTPStatus.OK, build_state(self.hat))
        elif path == "/events":
            self._stream()
        else:
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def _event(self, name: str, data: str) -> None:
        self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode())
        self.wfile.flush()

    def _stream(self) -> None:
        hat = self.hat
        ui = self.server.ui
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        last_version = -1
        last_state = 0.0
        stick_seq, backlog = hat.stick._log_since(0)
        try:
            for seq, event in backlog[-15:]:
                self._event("stick", _dumps({"seq": seq, **event._asdict()}))
            while not ui._stopping.is_set():
                with hat._changed:
                    hat._changed.wait_for(
                        lambda: hat._frame_version != last_version or hat.stick._log_seq != stick_seq,
                        timeout=0.05,
                    )
                version = hat._frame_version
                if version != last_version:
                    last_version = version
                    frame = base64.b64encode(hat.led_matrix.tobytes()).decode()
                    self._event("frame", frame)
                stick_seq, fresh = hat.stick._log_since(stick_seq)
                for seq, event in fresh:
                    self._event("stick", _dumps({"seq": seq, **event._asdict()}))
                now = time.monotonic()
                if now - last_state >= _STATE_INTERVAL:
                    last_state = now
                    self._event("state", _dumps(build_state(hat)))
        except (BrokenPipeError, ConnectionError, OSError):
            return  # browser went away

    # -- POST -----------------------------------------------------------------

    def do_POST(self) -> None:
        if not self._host_ok():
            self._json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "bad Host header"})
            return
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self._json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "cross-origin request refused"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"ok": False, "error": "send application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 <= length <= _MAX_BODY:
                raise ValueError("body too large")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            self._route(urlsplit(self.path).path, body)
        except LookupError as error:  # KeyError / IndexError: unknown name
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"unknown field: {error}"})
        except (ValueError, TypeError) as error:  # includes JSON decode errors
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})
        else:
            self._json(HTTPStatus.OK, {"ok": True})

    def _route(self, path: str, body: dict[str, Any]) -> None:
        hat = self.hat
        if path == "/api/sim":
            hat.sim.update(body)
        elif path == "/api/light":
            allowed = {"red", "green", "blue", "clear"}
            unknown = set(body) - allowed
            if unknown:
                raise KeyError(sorted(unknown)[0])
            hat.colour.simulate(**body)
        elif path == "/api/stick":
            hat.stick.inject_event(body.get("direction"), body.get("action", "pressed"))
        elif path == "/api/reset":
            hat.reset_simulation()
        else:
            raise KeyError(path)


class WebUI:
    """Serves the simulator page for one ``SenseHat``."""

    def __init__(
        self,
        hat,
        host: str = "127.0.0.1",
        port: int = 8000,
        open_browser: bool = False,
        port_tries: int = 20,
    ) -> None:
        self.hat = hat
        self.host = host
        self.port = port
        self.loopback_only = host in _LOOPBACK
        self._open_browser = open_browser
        self._port_tries = port_tries
        self._server: _Server | None = None
        self._stopping = threading.Event()

    @property
    def url(self) -> str:
        shown = "localhost" if self.host in ("", "0.0.0.0", "127.0.0.1") else self.host
        if ":" in shown:
            shown = f"[{shown}]"
        return f"http://{shown}:{self.port}/"

    def start(self) -> None:
        server_class = _Server6 if ":" in self.host else _Server
        last_error: OSError | None = None
        for candidate in range(self.port, self.port + self._port_tries):
            try:
                server = server_class((self.host, candidate), _Handler)
            except OSError as error:
                last_error = error
                continue
            break
        else:
            raise OSError(f"no free port in {self.port}-{self.port + self._port_tries - 1}: {last_error}")
        server.ui = self
        self._server = server
        self.port = server.server_address[1]
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()
        if self._open_browser:
            webbrowser.open(self.url)

    def stop(self) -> None:
        self._stopping.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def wait(self) -> None:
        """Block until Ctrl+C - handy to keep the UI up after your script finishes."""
        try:
            while not self._stopping.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass