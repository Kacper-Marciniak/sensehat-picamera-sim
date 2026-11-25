"""Sense HAT simulator.

Drop-in stand-in for the Raspberry Pi ``sense_hat`` package::

    from sense_hat import SenseHat, ACTION_PRESSED, DIRECTION_UP
    hat = SenseHat()
    hat.set_pixel(0, 0, 255, 0, 0)
    print(hat.colour.colour, hat.temperature)

Creating a ``SenseHat`` also starts a small web UI (see ``sense_hat_web.py``)
that previews the LEDs and lets you change the simulated sensors and press
the joystick in real time. The address is printed to stderr, usually
http://localhost:8000/ .

Simulator-only extras (not part of the real API):
    SenseHat(web=False)          don't start the web UI
    SenseHat(web_port=9000)      choose the port (next free one is used if busy)
    hat.web                      the WebUI object (``hat.web.url``, ``hat.web.wait()``)
    hat.sim                      values returned by the sensors (see SensorSimulation)
    hat.colour.simulate(...)     set the light seen by the colour sensor
    hat.stick.inject_event(...)  fake a joystick event from code
    hat.led_matrix               rendered 8x8x3 uint8 image of the LEDs
    hat.reset_simulation()       restore all simulated inputs to their defaults
Set SENSE_HAT_SIM_HEADLESS=1 to disable the web UI everywhere, and
SENSE_HAT_SIM_OPEN=1 to open it in your browser automatically.
"""

from __future__ import annotations

import math
import os
import queue
import random
import sys
import threading
import time
import traceback
import warnings
from collections import deque, namedtuple
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__version__ = "0.3.0"

__all__ = [
    "SenseHat",
    "SenseStick",
    "ColourSensor",
    "InputEvent",
    "SensorSimulation",
    "ACTION_PRESSED",
    "ACTION_RELEASED",
    "ACTION_HELD",
    "DIRECTION_UP",
    "DIRECTION_DOWN",
    "DIRECTION_LEFT",
    "DIRECTION_RIGHT",
    "DIRECTION_MIDDLE",
]

Colour = Sequence[int]

# --------------------------------------------------------------------------
# Joystick
# --------------------------------------------------------------------------

ACTION_PRESSED = "pressed"
ACTION_RELEASED = "released"
ACTION_HELD = "held"

DIRECTION_UP = "up"
DIRECTION_DOWN = "down"
DIRECTION_LEFT = "left"
DIRECTION_RIGHT = "right"
DIRECTION_MIDDLE = "middle"

_ACTIONS = (ACTION_PRESSED, ACTION_RELEASED, ACTION_HELD)
_DIRECTIONS = (DIRECTION_UP, DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_MIDDLE)

# Same field order as the real library.
InputEvent = namedtuple("InputEvent", ("timestamp", "direction", "action"))


def _callback_property(key: str) -> property:
    def getter(self: "SenseStick"):
        return self._callbacks.get(key)

    def setter(self: "SenseStick", value) -> None:
        self._set_callback(key, value)

    return property(getter, setter, doc=f"Callback called with an InputEvent for '{key}' events.")


class SenseStick:
    """Simulated joystick."""

    def __init__(self) -> None:
        self._events: queue.Queue = queue.Queue()
        self._callbacks: dict[str, Any] = {}
        self._callback_thread: threading.Thread | None = None
        self._callback_lock = threading.Lock()
        # Simulator-only: history of every event, for the web UI.
        self._log: deque = deque(maxlen=100)
        self._log_seq = 0
        self._log_lock = threading.Lock()
        self._listener = None

    # -- event sources ------------------------------------------------------

    def _push(self, event: InputEvent) -> None:
        with self._log_lock:
            self._log_seq += 1
            self._log.append((self._log_seq, event))
        self._events.put(event)
        if self._listener is not None:
            self._listener()

    def _log_since(self, seq: int) -> tuple[int, list[tuple[int, InputEvent]]]:
        with self._log_lock:
            return self._log_seq, [(s, e) for s, e in self._log if s > seq]

    def inject_event(
        self,
        direction: str,
        action: str = ACTION_PRESSED,
        timestamp: float | None = None,
    ) -> InputEvent:
        """Simulator-only: queue a joystick event from code."""
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}")
        if action not in _ACTIONS:
            raise ValueError(f"action must be one of {_ACTIONS}")
        event = InputEvent(time.time() if timestamp is None else float(timestamp), direction, action)
        self._push(event)
        return event

    # -- real API -----------------------------------------------------------

    def get_events(self) -> list[InputEvent]:
        """Return (and clear) all events received since the last call."""
        events: list[InputEvent] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def wait_for_event(self, emptybuffer: bool = False) -> InputEvent:
        """Block until an event arrives. ``emptybuffer`` discards queued events first."""
        if emptybuffer:
            self.get_events()
        while True:
            try:
                return self._events.get(timeout=0.25)
            except queue.Empty:
                continue

    direction_up = _callback_property(DIRECTION_UP)
    direction_down = _callback_property(DIRECTION_DOWN)
    direction_left = _callback_property(DIRECTION_LEFT)
    direction_right = _callback_property(DIRECTION_RIGHT)
    direction_middle = _callback_property(DIRECTION_MIDDLE)
    direction_any = _callback_property("any")

    # -- callbacks ----------------------------------------------------------

    def _set_callback(self, key: str, value) -> None:
        if value is None:
            self._callbacks.pop(key, None)
            return
        if not callable(value):
            raise TypeError("callback must be callable or None")
        self._callbacks[key] = value
        with self._callback_lock:
            if self._callback_thread is None or not self._callback_thread.is_alive():
                self._callback_thread = threading.Thread(target=self._run_callbacks, daemon=True)
                self._callback_thread.start()

    def _run_callbacks(self) -> None:
        # Like the real library, the callback thread consumes events.
        while True:
            with self._callback_lock:
                if not self._callbacks:
                    self._callback_thread = None
                    return
            try:
                event = self._events.get(timeout=0.1)
            except queue.Empty:
                continue
            for key in (event.direction, "any"):
                callback = self._callbacks.get(key)
                if callback is not None:
                    try:
                        callback(event)
                    except Exception:  # keep the thread alive
                        traceback.print_exc()


# --------------------------------------------------------------------------
# Colour sensor
# --------------------------------------------------------------------------


class ColourSensor:
    """Simulated TCS34725 colour sensor (Sense HAT v2)."""

    _DEFAULT_LIGHT = {"red": 128.0, "green": 128.0, "blue": 128.0, "clear": 384.0}

    def __init__(self, gain: int = 1, integration_cycles: int = 1) -> None:
        self._gain = 1
        self._integration_cycles = 1
        self.gain = gain
        self.integration_cycles = integration_cycles
        # Light level seen by the sensor, in raw counts at gain=1, 1 cycle.
        self._light = dict(self._DEFAULT_LIGHT)
        self._sim: "SensorSimulation | None" = None  # set by SenseHat, supplies noise

    def simulate(
        self,
        red: float | None = None,
        green: float | None = None,
        blue: float | None = None,
        clear: float | None = None,
    ) -> None:
        """Simulator-only: set the light level (raw counts at gain 1, 1 cycle)."""
        for name, value in (("red", red), ("green", green), ("blue", blue), ("clear", clear)):
            if value is not None:
                if value < 0:
                    raise ValueError("light levels must be non-negative")
                self._light[name] = float(value)

    def reset_light(self) -> None:
        """Simulator-only: restore the default light level."""
        self._light = dict(self._DEFAULT_LIGHT)

    @property
    def light(self) -> dict[str, float]:
        """Simulator-only: the light level set with ``simulate``."""
        return dict(self._light)

    # -- helpers ------------------------------------------------------------

    @property
    def _max_raw(self) -> int:
        return min(65535, 1024 * self._integration_cycles)

    def _raw(self, channel: str) -> int:
        level = self._light[channel]
        if self._sim is not None and self._sim.noise > 0:
            level = max(0.0, level * (1.0 + random.gauss(0.0, 0.01 * self._sim.noise)))
        value = level * self._gain * self._integration_cycles
        return int(min(value, self._max_raw))

    def _scaled(self, channel: str) -> int:
        return min(int(self._raw(channel) / self._max_raw * 256), 255)

    # -- raw values ---------------------------------------------------------

    @property
    def red_raw(self) -> int:
        return self._raw("red")

    @property
    def green_raw(self) -> int:
        return self._raw("green")

    @property
    def blue_raw(self) -> int:
        return self._raw("blue")

    @property
    def clear_raw(self) -> int:
        return self._raw("clear")

    @property
    def colour_raw(self) -> tuple[int, int, int, int]:
        return (self.red_raw, self.green_raw, self.blue_raw, self.clear_raw)

    # -- scaled (0-255) values ----------------------------------------------

    @property
    def red(self) -> int:
        return self._scaled("red")

    @property
    def green(self) -> int:
        return self._scaled("green")

    @property
    def blue(self) -> int:
        return self._scaled("blue")

    @property
    def clear(self) -> int:
        return self._scaled("clear")

    @property
    def colour(self) -> tuple[int, int, int, int]:
        return (self.red, self.green, self.blue, self.clear)

    @property
    def rgb(self) -> tuple[int, int, int]:
        return (self.red, self.green, self.blue)

    @property
    def brightness(self) -> int:
        return self.clear_raw

    # -- settings -----------------------------------------------------------

    @property
    def gain(self) -> int:
        return self._gain

    @gain.setter
    def gain(self, value: int) -> None:
        if value not in (1, 4, 16, 60):
            raise ValueError("Gain must be one of 1, 4, 16 or 60")
        self._gain = int(value)

    @property
    def integration_cycles(self) -> int:
        return self._integration_cycles

    @integration_cycles.setter
    def integration_cycles(self, value: int) -> None:
        if not isinstance(value, (int, np.integer)) or not 1 <= value <= 256:
            raise ValueError("Integration cycles must be an integer between 1 and 256")
        self._integration_cycles = int(value)

    @property
    def integration_time(self) -> float:
        """Integration time in milliseconds (2.4 ms per cycle)."""
        return self._integration_cycles * 2.4


# --------------------------------------------------------------------------
# Font (classic 5x7 column font, ASCII 0x20-0x7E; bit 0 = top row)
# --------------------------------------------------------------------------

_FONT_DATA = [
    (0x00, 0x00, 0x00, 0x00, 0x00), (0x00, 0x00, 0x5F, 0x00, 0x00), (0x00, 0x07, 0x00, 0x07, 0x00),
    (0x14, 0x7F, 0x14, 0x7F, 0x14), (0x24, 0x2A, 0x7F, 0x2A, 0x12), (0x23, 0x13, 0x08, 0x64, 0x62),
    (0x36, 0x49, 0x56, 0x20, 0x50), (0x00, 0x08, 0x07, 0x03, 0x00), (0x00, 0x1C, 0x22, 0x41, 0x00),
    (0x00, 0x41, 0x22, 0x1C, 0x00), (0x2A, 0x1C, 0x7F, 0x1C, 0x2A), (0x08, 0x08, 0x3E, 0x08, 0x08),
    (0x00, 0x80, 0x70, 0x30, 0x00), (0x08, 0x08, 0x08, 0x08, 0x08), (0x00, 0x00, 0x60, 0x60, 0x00),
    (0x20, 0x10, 0x08, 0x04, 0x02),
    (0x3E, 0x51, 0x49, 0x45, 0x3E), (0x00, 0x42, 0x7F, 0x40, 0x00), (0x72, 0x49, 0x49, 0x49, 0x46),
    (0x21, 0x41, 0x49, 0x4D, 0x33), (0x18, 0x14, 0x12, 0x7F, 0x10), (0x27, 0x45, 0x45, 0x45, 0x39),
    (0x3C, 0x4A, 0x49, 0x49, 0x31), (0x41, 0x21, 0x11, 0x09, 0x07), (0x36, 0x49, 0x49, 0x49, 0x36),
    (0x46, 0x49, 0x49, 0x29, 0x1E),
    (0x00, 0x00, 0x14, 0x00, 0x00), (0x00, 0x40, 0x34, 0x00, 0x00), (0x00, 0x08, 0x14, 0x22, 0x41),
    (0x14, 0x14, 0x14, 0x14, 0x14), (0x00, 0x41, 0x22, 0x14, 0x08), (0x02, 0x01, 0x59, 0x09, 0x06),
    (0x3E, 0x41, 0x5D, 0x59, 0x4E),
    (0x7C, 0x12, 0x11, 0x12, 0x7C), (0x7F, 0x49, 0x49, 0x49, 0x36), (0x3E, 0x41, 0x41, 0x41, 0x22),
    (0x7F, 0x41, 0x41, 0x41, 0x3E), (0x7F, 0x49, 0x49, 0x49, 0x41), (0x7F, 0x09, 0x09, 0x09, 0x01),
    (0x3E, 0x41, 0x41, 0x51, 0x73), (0x7F, 0x08, 0x08, 0x08, 0x7F), (0x00, 0x41, 0x7F, 0x41, 0x00),
    (0x20, 0x40, 0x41, 0x3F, 0x01), (0x7F, 0x08, 0x14, 0x22, 0x41), (0x7F, 0x40, 0x40, 0x40, 0x40),
    (0x7F, 0x02, 0x1C, 0x02, 0x7F), (0x7F, 0x04, 0x08, 0x10, 0x7F), (0x3E, 0x41, 0x41, 0x41, 0x3E),
    (0x7F, 0x09, 0x09, 0x09, 0x06), (0x3E, 0x41, 0x51, 0x21, 0x5E), (0x7F, 0x09, 0x19, 0x29, 0x46),
    (0x26, 0x49, 0x49, 0x49, 0x32), (0x03, 0x01, 0x7F, 0x01, 0x03), (0x3F, 0x40, 0x40, 0x40, 0x3F),
    (0x1F, 0x20, 0x40, 0x20, 0x1F), (0x3F, 0x40, 0x38, 0x40, 0x3F), (0x63, 0x14, 0x08, 0x14, 0x63),
    (0x03, 0x04, 0x78, 0x04, 0x03), (0x61, 0x59, 0x49, 0x4D, 0x43),
    (0x00, 0x7F, 0x41, 0x41, 0x41), (0x02, 0x04, 0x08, 0x10, 0x20), (0x00, 0x41, 0x41, 0x41, 0x7F),
    (0x04, 0x02, 0x01, 0x02, 0x04), (0x40, 0x40, 0x40, 0x40, 0x40), (0x00, 0x03, 0x07, 0x08, 0x00),
    (0x20, 0x54, 0x54, 0x78, 0x40), (0x7F, 0x28, 0x44, 0x44, 0x38), (0x38, 0x44, 0x44, 0x44, 0x28),
    (0x38, 0x44, 0x44, 0x28, 0x7F), (0x38, 0x54, 0x54, 0x54, 0x18), (0x00, 0x08, 0x7E, 0x09, 0x02),
    (0x18, 0xA4, 0xA4, 0x9C, 0x78), (0x7F, 0x08, 0x04, 0x04, 0x78), (0x00, 0x44, 0x7D, 0x40, 0x00),
    (0x20, 0x40, 0x40, 0x3D, 0x00), (0x7F, 0x10, 0x28, 0x44, 0x00), (0x00, 0x41, 0x7F, 0x40, 0x00),
    (0x7C, 0x04, 0x78, 0x04, 0x78), (0x7C, 0x08, 0x04, 0x04, 0x78), (0x38, 0x44, 0x44, 0x44, 0x38),
    (0xFC, 0x18, 0x24, 0x24, 0x18), (0x18, 0x24, 0x24, 0x18, 0xFC), (0x7C, 0x08, 0x04, 0x04, 0x08),
    (0x48, 0x54, 0x54, 0x54, 0x24), (0x04, 0x04, 0x3F, 0x44, 0x24), (0x3C, 0x40, 0x40, 0x20, 0x7C),
    (0x1C, 0x20, 0x40, 0x20, 0x1C), (0x3C, 0x40, 0x30, 0x40, 0x3C), (0x44, 0x28, 0x10, 0x28, 0x44),
    (0x4C, 0x90, 0x90, 0x90, 0x7C), (0x44, 0x64, 0x54, 0x4C, 0x44),
    (0x00, 0x08, 0x36, 0x41, 0x00), (0x00, 0x00, 0x77, 0x00, 0x00), (0x00, 0x41, 0x36, 0x08, 0x00),
    (0x02, 0x01, 0x02, 0x04, 0x02),
]
assert len(_FONT_DATA) == 95

_FONT: dict[str, list[np.ndarray]] = {
    chr(0x20 + i): [np.array([(col >> row) & 1 for row in range(8)], dtype=bool) for col in cols]
    for i, cols in enumerate(_FONT_DATA)
}


def _glyph(char: str) -> list[np.ndarray]:
    return _FONT.get(char, _FONT["?"])


def _trim_whitespace(columns: list[np.ndarray]) -> list[np.ndarray]:
    """Strip empty columns from both ends (a blank glyph such as ' ' is kept as is)."""
    if any(col.any() for col in columns):
        while not columns[0].any():
            columns = columns[1:]
        while not columns[-1].any():
            columns = columns[:-1]
    return columns


# --------------------------------------------------------------------------
# Simulated sensor values
# --------------------------------------------------------------------------

_EARTH_FIELD_HORIZONTAL = 20.0  # microteslas
_EARTH_FIELD_VERTICAL = -40.0


class SensorSimulation:
    """The world as seen by the simulated sensors.

    Assign attributes directly (``hat.sim.temperature = 30``) or call
    ``hat.sim.update(...)``. The web UI edits the same object.

    With ``link_motion_sensors`` (the default) the accelerometer and compass
    are derived from ``orientation`` - tilt the board and gravity follows. The
    axis/sign conventions are an approximation of the real hardware. Set it to
    False to control ``accelerometer_raw`` and ``compass_raw`` by hand.
    ``noise`` scales Gaussian read noise (0 = none, 1 = typical sensor noise).
    """

    # One standard deviation of read noise per unit of ``noise``.
    NOISE_SIGMA = {
        "temperature": 0.10,  # degrees C
        "humidity": 0.50,  # % RH
        "pressure": 0.10,  # millibars
        "orientation": 0.20,  # degrees
        "accelerometer": 0.004,  # Gs
        "gyroscope": 0.002,  # rad/s
        "compass": 0.30,  # microteslas
    }

    _SCALARS = {
        "temperature": (-40.0, 120.0),
        "humidity": (0.0, 100.0),
        "pressure": (260.0, 1260.0),
        "noise": (0.0, 10.0),
    }
    _VECTORS = {
        "orientation": (("pitch", "roll", "yaw"), (-720.0, 720.0)),
        "accelerometer_raw": (("x", "y", "z"), (-16.0, 16.0)),
        "gyroscope_raw": (("x", "y", "z"), (-100.0, 100.0)),
        "compass_raw": (("x", "y", "z"), (-1000.0, 1000.0)),
    }

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.temperature = 25.0  # degrees C (humidity- and pressure-sensor readings)
        self.humidity = 45.0  # % relative humidity
        self.pressure = 1013.25  # millibars
        self.orientation = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}  # degrees
        self.accelerometer_raw = {"x": 0.0, "y": 0.0, "z": 1.0}  # Gs
        self.gyroscope_raw = {"x": 0.0, "y": 0.0, "z": 0.0}  # rad/s
        self.compass_raw = {"x": 20.0, "y": 0.0, "z": -40.0}  # microteslas
        self.noise = 0.0
        self.link_motion_sensors = True

    # -- bulk access (used by the web UI) -----------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Current values; accelerometer/compass are the effective (derived) ones."""
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "pressure": self.pressure,
            "orientation": dict(self.orientation),
            "accelerometer_raw": self._accelerometer_base(),
            "gyroscope_raw": dict(self.gyroscope_raw),
            "compass_raw": self._compass_base(),
            "noise": self.noise,
            "link_motion_sensors": bool(self.link_motion_sensors),
        }

    def update(self, values: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Validate and apply several values at once (all or nothing).

        Vector values may be partial: ``update(orientation={"pitch": 10})``.
        Raises KeyError for unknown names, TypeError/ValueError for bad values.
        """
        values = dict(values or {}, **kwargs)
        staged: dict[str, Any] = {}
        for key, value in values.items():
            if key in self._SCALARS:
                staged[key] = self._number(key, value, *self._SCALARS[key])
            elif key in self._VECTORS:
                axes, (low, high) = self._VECTORS[key]
                if not isinstance(value, dict):
                    raise TypeError(f"{key} must be an object with keys {axes}")
                vector = dict(getattr(self, key))
                for axis, component in value.items():
                    if axis not in axes:
                        raise KeyError(f"{key}.{axis}")
                    number = self._number(f"{key}.{axis}", component, low, high)
                    vector[axis] = number % 360.0 if key == "orientation" else number
                staged[key] = vector
            elif key == "link_motion_sensors":
                staged[key] = bool(value)
            else:
                raise KeyError(key)
        for key, value in staged.items():
            setattr(self, key, value)

    @staticmethod
    def _number(name: str, value: Any, low: float, high: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        value = float(value)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low:g} and {high:g}")
        return value

    # -- derived motion sensors ---------------------------------------------

    def _accelerometer_base(self) -> dict[str, float]:
        if not self.link_motion_sensors:
            return dict(self.accelerometer_raw)
        pitch = math.radians(self.orientation["pitch"])
        roll = math.radians(self.orientation["roll"])
        return {
            "x": round(-math.sin(pitch), 6),
            "y": round(math.sin(roll) * math.cos(pitch), 6),
            "z": round(math.cos(roll) * math.cos(pitch), 6),
        }

    def _compass_base(self) -> dict[str, float]:
        if not self.link_motion_sensors:
            return dict(self.compass_raw)
        yaw = math.radians(self.orientation["yaw"])
        return {
            "x": round(_EARTH_FIELD_HORIZONTAL * math.cos(yaw), 6),
            "y": round(-_EARTH_FIELD_HORIZONTAL * math.sin(yaw), 6),
            "z": _EARTH_FIELD_VERTICAL,
        }

    # -- what the sensors report (with optional noise) ----------------------

    def _noisy(self, value: float, kind: str) -> float:
        if self.noise > 0:
            return value + random.gauss(0.0, self.NOISE_SIGMA[kind] * self.noise)
        return value

    def _noisy_vector(self, vector: dict[str, float], kind: str) -> dict[str, float]:
        return {axis: self._noisy(value, kind) for axis, value in vector.items()}

    def read_temperature(self) -> float:
        return self._noisy(self.temperature, "temperature")

    def read_humidity(self) -> float:
        return min(100.0, max(0.0, self._noisy(self.humidity, "humidity")))

    def read_pressure(self) -> float:
        return self._noisy(self.pressure, "pressure")

    def read_orientation(self) -> dict[str, float]:
        return self._noisy_vector(self.orientation, "orientation")

    def read_accelerometer(self) -> dict[str, float]:
        return self._noisy_vector(self._accelerometer_base(), "accelerometer")

    def read_gyroscope(self) -> dict[str, float]:
        return self._noisy_vector(self.gyroscope_raw, "gyroscope")

    def read_compass(self) -> dict[str, float]:
        return self._noisy_vector(self._compass_base(), "compass")


# --------------------------------------------------------------------------
# Sense HAT
# --------------------------------------------------------------------------

_PIXEL_ARG_ERROR = "Pixel arguments must be given as (r, g, b) or r, g, b"
_QUANTISE_MASK = np.array([0xF8, 0xFC, 0xF8], dtype=np.uint8)  # RGB565, as on the real LED matrix

# NOTE: approximations - the real kernel driver's tables differ slightly.
_GAMMA_DEFAULT = list(range(32))
_GAMMA_LOW_LIGHT = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 9, 10, 10]


class SenseHat:
    """
    Simulated Sense HAT.
    """
    def __init__(
        self,
        imu_settings_file: str = "RTIMULib",
        text_assets: str = "sense_hat_text",
        *,
        web: bool | None = None,
        web_host: str = "127.0.0.1",
        web_port: int = 8000,
        web_open: bool = False,
    ) -> None:
        self._imu_settings_file = imu_settings_file
        self._text_assets = text_assets

        # Change notification for the web UI (see sense_hat_web.py).
        self._changed = threading.Condition()
        self._frame_version = 0

        self.sim = SensorSimulation()
        self.colour = ColourSensor()
        self.colour._sim = self.sim
        self.stick = SenseStick()
        self.stick._listener = self._notify

        self._rotation = 0
        self._low_light = False
        self._gamma = list(_GAMMA_DEFAULT)
        self._imu_config = (True, True, True)  # compass, gyro, accel
        self._hw = np.zeros((8, 8, 3), dtype=np.uint8)  # as physically lit: [row, col]

        self.web = None
        headless = bool(os.environ.get("SENSE_HAT_SIM_HEADLESS"))
        if web is None:
            web = not headless
            explicit = False
        else:
            explicit = bool(web)
            web = bool(web) and not headless
        if web:
            self._start_web(explicit, web_host, web_port, web_open or bool(os.environ.get("SENSE_HAT_SIM_OPEN")))

    def _start_web(self, explicit: bool, host: str, port: int, open_browser: bool) -> None:
        try:
            from .web import WebUI

            ui = WebUI(self, host=host, port=port, open_browser=open_browser)
            ui.start()
        except Exception as error:  # missing file, no free port, ...
            if explicit:
                raise
            warnings.warn(f"Sense HAT simulator web UI not started: {error}", RuntimeWarning, stacklevel=3)
            return
        self.web = ui
        print(f"Sense HAT simulator UI: {ui.url}", file=sys.stderr)

    def reset_simulation(self) -> None:
        """Simulator-only: restore every simulated input to its default."""
        self.sim.reset()
        self.colour.reset_light()

    def _notify(self) -> None:
        with self._changed:
            self._changed.notify_all()

    # -- display plumbing -----------------------------------------------------

    @staticmethod
    def _hw_pos(x: int, y: int, rotation: int) -> tuple[int, int]:
        """Map user-frame (x, y) to the (row, col) that lights up."""
        if rotation == 0:
            return y, x
        if rotation == 90:
            return x, 7 - y
        if rotation == 180:
            return 7 - y, 7 - x
        return 7 - x, y

    def _get_logical(self) -> np.ndarray:
        return np.rot90(self._hw, self._rotation // 90).copy()

    def _set_logical(self, frame: np.ndarray) -> None:
        self._hw = np.ascontiguousarray(np.rot90(frame, -(self._rotation // 90))) & _QUANTISE_MASK
        self._refresh()

    def _render_frame(self) -> np.ndarray:
        lut = np.array(self._gamma, dtype=np.uint16)
        return (lut[self._hw >> 3] * 255 // 31).astype(np.uint8)

    def _refresh(self) -> None:
        with self._changed:
            self._frame_version += 1
            self._changed.notify_all()

    @property
    def led_matrix(self) -> np.ndarray:
        """Simulator-only: the 8x8x3 uint8 image currently shown on the LEDs."""
        return self._render_frame()

    @staticmethod
    def _check_pixel(pixel) -> tuple[int, int, int]:
        try:
            r, g, b = pixel
        except (TypeError, ValueError):
            raise ValueError(_PIXEL_ARG_ERROR) from None
        for element in (r, g, b):
            if element > 255 or element < 0:
                raise ValueError("Pixel elements must be between 0 and 255")
        return int(r), int(g), int(b)

    @classmethod
    def _pixel_from_args(cls, args: tuple) -> tuple[int, int, int]:
        if len(args) == 1:
            return cls._check_pixel(args[0])
        if len(args) == 3:
            return cls._check_pixel(args)
        raise ValueError(_PIXEL_ARG_ERROR)

    # -- rotation -------------------------------------------------------------

    def set_rotation(self, r: int = 0, redraw: bool = True) -> None:
        if r not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0, 90, 180 or 270 degrees")
        if redraw:
            logical = self._get_logical()
            self._rotation = r
            self._set_logical(logical)
        else:
            self._rotation = r

    @property
    def rotation(self) -> int:
        return self._rotation

    @rotation.setter
    def rotation(self, r: int) -> None:
        self.set_rotation(r)

    # -- pixels ---------------------------------------------------------------

    def set_pixel(self, x: int, y: int, *args) -> None:
        pixel = self._pixel_from_args(args)
        if x > 7 or x < 0:
            raise ValueError("X position must be between 0 and 7")
        if y > 7 or y < 0:
            raise ValueError("Y position must be between 0 and 7")
        row, col = self._hw_pos(x, y, self._rotation)
        self._hw[row, col] = np.array(pixel, dtype=np.uint8) & _QUANTISE_MASK
        self._refresh()

    def get_pixel(self, x: int, y: int) -> list[int]:
        if x > 7 or x < 0:
            raise ValueError("X position must be between 0 and 7")
        if y > 7 or y < 0:
            raise ValueError("Y position must be between 0 and 7")
        row, col = self._hw_pos(x, y, self._rotation)
        return [int(v) for v in self._hw[row, col]]

    def set_pixels(self, pixel_list: Iterable[Colour]) -> None:
        pixel_list = list(pixel_list)
        if len(pixel_list) != 64:
            raise ValueError("Pixel lists must have 64 elements")
        frame = np.zeros((64, 3), dtype=np.uint8)
        for index, pixel in enumerate(pixel_list):
            if len(pixel) != 3:
                raise ValueError(
                    "Pixel at index %d is invalid. Pixels must contain 3 elements: Red, Green and Blue" % index
                )
            for element in pixel:
                if element > 255 or element < 0:
                    raise ValueError(
                        "Pixel at index %d is invalid. Pixel elements must be between 0 and 255" % index
                    )
            frame[index] = pixel
        self._set_logical(frame.reshape(8, 8, 3))

    def get_pixels(self) -> list[list[int]]:
        return self._get_logical().reshape(64, 3).astype(int).tolist()

    def clear(self, *args) -> None:
        if len(args) == 0:
            pixel = (0, 0, 0)
        else:
            pixel = self._pixel_from_args(args)
        self.set_pixels([pixel] * 64)

    def flip_h(self, redraw: bool = True) -> list[list[int]]:
        flipped = np.flip(self._get_logical(), axis=1)
        if redraw:
            self._set_logical(flipped)
        return flipped.reshape(64, 3).astype(int).tolist()

    def flip_v(self, redraw: bool = True) -> list[list[int]]:
        flipped = np.flip(self._get_logical(), axis=0)
        if redraw:
            self._set_logical(flipped)
        return flipped.reshape(64, 3).astype(int).tolist()

    def load_image(self, file_path: str | Path, redraw: bool = True) -> list[list[int]]:
        file_path = os.fspath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError("%s not found" % file_path)
        from PIL import Image  # same dependency as the real library

        with Image.open(file_path) as img:
            pixel_list = [list(pixel) for pixel in img.convert("RGB").getdata()]
        if redraw:
            self.set_pixels(pixel_list)
        return pixel_list

    # -- text -----------------------------------------------------------------

    def _show_mask(self, mask: np.ndarray, text_colour: Colour, back_colour: Colour) -> None:
        frame = np.empty((8, 8, 3), dtype=np.uint8)
        frame[:] = back_colour
        frame[mask] = text_colour
        self._set_logical(frame)

    def show_letter(
        self,
        s: str,
        text_colour: Colour = (255, 255, 255),
        back_colour: Colour = (0, 0, 0),
    ) -> None:
        if len(s) > 1:
            raise ValueError("Only one character may be passed into this method")
        text = self._check_pixel(text_colour)
        back = self._check_pixel(back_colour)
        mask = np.zeros((8, 8), dtype=bool)
        for offset, column in enumerate(_glyph(s)):
            mask[:, 1 + offset] = column
        self._show_mask(mask, text, back)

    def show_message(
        self,
        text_string: str,
        scroll_speed: float = 0.1,
        text_colour: Colour = (255, 255, 255),
        back_colour: Colour = (0, 0, 0),
    ) -> None:
        text = self._check_pixel(text_colour)
        back = self._check_pixel(back_colour)
        blank = np.zeros(8, dtype=bool)
        columns = [blank] * 8
        for char in str(text_string):
            columns.extend(_trim_whitespace(_glyph(char)))
            columns.append(blank)
        columns.extend([blank] * 8)
        for start in range(len(columns) - 8):
            self._show_mask(np.stack(columns[start : start + 8], axis=1), text, back)
            time.sleep(scroll_speed)

    # -- gamma / low light ----------------------------------------------------

    @property
    def low_light(self) -> bool:
        return self._low_light

    @low_light.setter
    def low_light(self, value: bool) -> None:
        self._low_light = bool(value)
        self._gamma = list(_GAMMA_LOW_LIGHT if self._low_light else _GAMMA_DEFAULT)
        self._refresh()

    @property
    def gamma(self) -> list[int]:
        return list(self._gamma)

    @gamma.setter
    def gamma(self, buffer: Sequence[int]) -> None:
        if len(buffer) != 32:
            raise ValueError("Gamma array must be of length 32")
        if not all(0 <= b <= 31 for b in buffer):
            raise ValueError("Gamma values must be between 0 and 31")
        self._gamma = [int(b) for b in buffer]
        self._refresh()

    def gamma_reset(self) -> None:
        self._gamma = list(_GAMMA_DEFAULT)
        self._refresh()

    # -- environmental sensors ------------------------------------------------

    def get_humidity(self) -> float:
        return float(self.sim.read_humidity())

    humidity = property(get_humidity)

    def get_temperature_from_humidity(self) -> float:
        return float(self.sim.read_temperature())

    def get_temperature_from_pressure(self) -> float:
        return float(self.sim.read_temperature())

    def get_temperature(self) -> float:
        return self.get_temperature_from_humidity()

    temp = property(get_temperature)
    temperature = property(get_temperature)

    def get_pressure(self) -> float:
        return float(self.sim.read_pressure())

    pressure = property(get_pressure)

    # -- IMU ------------------------------------------------------------------

    def set_imu_config(self, compass_enabled: bool, gyro_enabled: bool, accel_enabled: bool) -> None:
        self._imu_config = (bool(compass_enabled), bool(gyro_enabled), bool(accel_enabled))

    def _orientation_radians(self, config: tuple[bool, bool, bool]) -> dict[str, float]:
        compass, gyro, accel = config
        source = self.sim.read_orientation()
        result = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}
        valid = []
        if gyro or accel:
            valid += ["pitch", "roll"]
        if gyro or compass:
            valid.append("yaw")
        for key in valid:
            rad = math.radians(source[key])
            result[key] = math.atan2(math.sin(rad), math.cos(rad))  # wrap to -pi..pi
        return result

    @staticmethod
    def _to_degrees(radians: dict[str, float]) -> dict[str, float]:
        out = {}
        for key, value in radians.items():
            deg = math.degrees(value)
            out[key] = round(deg + 360 if deg < 0 else deg, 9)
        return out

    def get_orientation_radians(self) -> dict[str, float]:
        return self._orientation_radians(self._imu_config)

    orientation_radians = property(get_orientation_radians)

    def get_orientation_degrees(self) -> dict[str, float]:
        return self._to_degrees(self.get_orientation_radians())

    orientation_degrees = property(get_orientation_degrees)

    def get_orientation(self) -> dict[str, float]:
        return self.get_orientation_degrees()

    orientation = property(get_orientation)

    def get_compass(self) -> float:
        return self._to_degrees(self._orientation_radians((True, False, False)))["yaw"]

    compass = property(get_compass)

    def get_compass_raw(self) -> dict[str, float]:
        if not self._imu_config[0]:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        return self.sim.read_compass()

    compass_raw = property(get_compass_raw)

    def get_gyroscope(self) -> dict[str, float]:
        return self._to_degrees(self._orientation_radians((False, True, False)))

    gyroscope = property(get_gyroscope)
    gyro = property(get_gyroscope)

    def get_gyroscope_raw(self) -> dict[str, float]:
        if not self._imu_config[1]:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        return self.sim.read_gyroscope()

    gyroscope_raw = property(get_gyroscope_raw)
    gyro_raw = property(get_gyroscope_raw)

    def get_accelerometer(self) -> dict[str, float]:
        return self._to_degrees(self._orientation_radians((False, False, True)))

    accelerometer = property(get_accelerometer)
    accel = property(get_accelerometer)

    def get_accelerometer_raw(self) -> dict[str, float]:
        if not self._imu_config[2]:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        return self.sim.read_accelerometer()

    accelerometer_raw = property(get_accelerometer_raw)
    accel_raw = property(get_accelerometer_raw)