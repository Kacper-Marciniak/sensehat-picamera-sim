"""Small webcam-backed stand-in for the Raspberry Pi ``picamera2`` package.
 
    from picamera2 import Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
    picam2.start()
    frame = picam2.capture_array()       # numpy array, same channel order as the real camera
 
Supported: configure/start/stop/close, capture_array, capture_file, capture_image,
formats RGB888 / BGR888 / XRGB8888 / XBGR8888, and Transform flips.
Not simulated (accepted and ignored): controls, previews, video recording, other streams.
"""
 
import threading
import time
import warnings
 
import cv2
import numpy as np
 
from .libcamera import Transform
 
CAMERA_ID = 0  # webcam index, or the path of a video file to play instead (it loops)
 
__all__ = ["Picamera2", "Preview"]
 
 
class Preview:  # so that "from picamera2 import Preview" works
    NULL, DRM, QT, QTGL = range(4)
 
 
_FORMATS = ("RGB888", "BGR888", "XRGB8888", "XBGR8888")
 
 
class Picamera2:
    """
    Simulated PiCamera2.
    """
    def __init__(self, camera_num=0):
        source = CAMERA_ID + camera_num if isinstance(CAMERA_ID, int) else CAMERA_ID
        self._cap = cv2.VideoCapture(source)
        ok, frame = self._cap.read() if self._cap.isOpened() else (False, None)
        if not ok:
            self._cap.release()
            raise RuntimeError("Camera __init__ sequence did not complete.") from OSError(
                f"cannot read from camera {source!r}")
        self.sensor_resolution = (frame.shape[1], frame.shape[0])
        self.camera_config = None
        self.started = False
        self._frame, self._running = frame, True
        self._paced = isinstance(source, str)
        self._lock = threading.Lock()
        self._warned = set()
        threading.Thread(target=self._read_forever, daemon=True).start()
 
    def _read_forever(self):
        # Keep reading so capture_array always returns the newest frame, not a buffered old one.
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            elif self._paced:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video files
            if self._paced or not ok:
                time.sleep(1 / 30 if ok or self._paced else 0.05)
 
    # -- configuration --------------------------------------------------------------
 
    def _make_config(self, use_case, main, transform, controls, default_main):
        return {"use_case": use_case, "main": {**default_main, **(main or {})},
                "transform": transform or Transform(), "controls": dict(controls or {})}
 
    # Extra keyword arguments of the real API (lores, raw, buffer_count, queue, ...) are accepted and ignored.
    def create_preview_configuration(self, main=None, transform=None, controls=None, **ignored):
        return self._make_config("preview", main, transform, controls, {"format": "XBGR8888", "size": (640, 480)})
 
    def create_still_configuration(self, main=None, transform=None, controls=None, **ignored):
        return self._make_config("still", main, transform, controls,
                                 {"format": "BGR888", "size": self.sensor_resolution})
 
    def create_video_configuration(self, main=None, transform=None, controls=None, **ignored):
        return self._make_config("video", main, transform, controls, {"format": "XBGR8888", "size": (1280, 720)})
 
    def configure(self, camera_config="preview"):
        if self.started:
            raise RuntimeError("Camera must be stopped before configuring")
        if camera_config is None or isinstance(camera_config, str):
            name = camera_config or "preview"
            if name not in ("preview", "still", "video"):
                raise ValueError(f"unknown configuration {name!r} (use 'preview', 'still' or 'video')")
            camera_config = getattr(self, f"create_{name}_configuration")()
        else:  # a (possibly partial) dict
            use_case = camera_config.get("use_case", "preview")
            camera_config = getattr(self, f"create_{use_case}_configuration")(
                main=camera_config.get("main"), transform=camera_config.get("transform"),
                controls=camera_config.get("controls"))
        main = camera_config["main"]
        if main["format"] not in _FORMATS:
            raise RuntimeError(f"unsupported format {main['format']!r}; the simulator supports {_FORMATS}")
        width, height = (int(v) for v in main["size"])
        main["size"] = (max(2, width // 2 * 2), max(2, height // 2 * 2))  # the hardware wants even sizes
        self.camera_config = camera_config
 
    def set_controls(self, controls):
        self._warn_once("controls", "set_controls() is accepted but has no effect in the simulator")
 
    def _warn_once(self, key, message):
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(message, RuntimeWarning, stacklevel=3)
 
    # -- lifecycle ----------------------------------------------------------------------
 
    def start(self, config=None, show_preview=False):
        if config is not None or self.camera_config is None:
            self.configure(config)
        if show_preview:
            self.start_preview()
        self.started = True
 
    def stop(self):
        self.started = False
 
    def close(self):
        self.stop()
        self._running = False
        time.sleep(0.05)
        self._cap.release()
 
    def __enter__(self):
        return self
 
    def __exit__(self, *exc):
        self.close()
 
    def start_preview(self, *args, **kwargs):
        self._warn_once("preview", "previews are not simulated; show frames with cv2.imshow instead")
 
    def stop_preview(self):
        pass
 
    # -- capturing ----------------------------------------------------------------------
 
    def capture_array(self, name="main"):
        if not self.started:
            raise RuntimeError("Camera must be started before capturing")
        if name != "main":
            raise RuntimeError("only the 'main' stream is simulated")
        with self._lock:
            bgr = self._frame
        transform = self.camera_config["transform"]
        if transform.transpose:
            bgr = bgr.transpose(1, 0, 2)
        if transform.hflip:
            bgr = cv2.flip(bgr, 1)
        if transform.vflip:
            bgr = cv2.flip(bgr, 0)
        main = self.camera_config["main"]
        bgr = cv2.resize(bgr, main["size"])  # stretched to the requested size
        # Memory order follows the real library (the names are "backwards"):
        #   RGB888 -> [B, G, R]   BGR888 -> [R, G, B]   XRGB8888 -> [B, G, R, 255]   XBGR8888 -> [R, G, B, 255]
        fmt = main["format"]
        if fmt == "RGB888":
            return bgr.copy()
        if fmt == "BGR888":
            return bgr[..., ::-1].copy()
        alpha = np.full(bgr.shape[:2] + (1,), 255, np.uint8)
        return np.concatenate([bgr if fmt == "XRGB8888" else bgr[..., ::-1], alpha], axis=2)
 
    def _bgr(self):
        array, fmt = self.capture_array(), self.camera_config["main"]["format"]
        return {"RGB888": lambda a: a, "BGR888": lambda a: a[..., ::-1],
                "XRGB8888": lambda a: a[..., :3], "XBGR8888": lambda a: a[..., 2::-1]}[fmt](array)
 
    def capture_file(self, file_output, name="main", format=None):
        """Save a JPEG/PNG/BMP image to a path or file object. Returns a small metadata dict."""
        bgr = np.ascontiguousarray(self._bgr())
        if format is None and isinstance(file_output, str):
            format = file_output.rsplit(".", 1)[-1] if "." in file_output else None
        ext = {"jpg": ".jpg", "jpeg": ".jpg", "png": ".png", "bmp": ".bmp"}.get((format or "").lower())
        if ext is None:
            raise ValueError("give a file name ending in .jpg/.png/.bmp, or format= for file objects")
        data = cv2.imencode(ext, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90] if ext == ".jpg" else [])[1].tobytes()
        if isinstance(file_output, str):
            with open(file_output, "wb") as handle:
                handle.write(data)
        else:
            file_output.write(data)
        return {"SensorTimestamp": time.monotonic_ns()}
 
    def capture_image(self, name="main"):
        """Return a PIL image (needs Pillow)."""
        from PIL import Image
        return Image.fromarray(np.ascontiguousarray(self._bgr()[..., ::-1]))