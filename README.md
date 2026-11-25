# Raspberry Pi Sense HAT and Camera Simulator

This project lets you try Raspberry Pi Sense HAT and PiCamera2-style Python
code on a regular computer. It is intended for students learning Python and
anyone who wants to experiment without connecting a Raspberry Pi or Sense HAT.

The simulator provides:

- An 8×8 LED matrix shown in a browser.
- Adjustable simulated temperature, humidity, pressure, light, orientation,
  accelerometer, gyroscope, and compass readings.
- An on-screen joystick that sends events to your Python program.
- A `Picamera2` stand-in that reads from your computer's webcam (or a video
  file).

This is a learning tool, not a complete or hardware-accurate replacement for
the Raspberry Pi libraries.

## What you need

- Python 3.10 or later.
- For the Sense HAT simulator: NumPy.
- For the camera simulator: OpenCV (`opencv-python`) and NumPy.
- Optional: Pillow (`pillow`) for image-loading and PIL image methods.

The web interface uses Python's standard library; you do not need to install a
web framework.

## Set up

You can run a program from this repository, or copy the `sense_hat` and
`picamera2` folders beside your own Python file. Keep the folder names as shown:
Python imports these local folders when your program runs.

In PowerShell, open this repository folder and install the dependencies:

```powershell
python -m pip install numpy opencv-python
```

If you only need the Sense HAT simulator, install just NumPy:

```powershell
python -m pip install numpy
```

If you want to use the optional image features:

```powershell
python -m pip install pillow
```

If `python` is not recognized on Windows, try `py -m pip` instead. To keep
these dependencies separate from other projects, consider using a virtual
environment created with Python's built-in `venv` or with Conda.

## Run the included example

Connect a webcam or enable your computer's built-in camera, then run:

```powershell
python test.py
```

The example starts the Sense HAT simulator, displays webcam frames on its LED
matrix, and prints simulated temperature and humidity readings. Press
**Ctrl+C** in the terminal to stop it. The example needs a working webcam; if
you do not have one, try the Sense HAT example below instead.

## Try the Sense HAT

You can follow the example below or run the included `test.py` file as
described above.

Create a Python file next to the `sense_hat` folder, for example `my_program.py`:

```python
import time
from sense_hat import SenseHat

hat = SenseHat()
hat.show_message("Hello!")

hat.set_pixel(0, 0, 255, 0, 0)  # x, y, red, green, blue

while True:
    print("Temperature:", hat.get_temperature(), "°C")
    print("Humidity:", hat.get_humidity(), "%")
    time.sleep(1)  # keep the program running and refresh readings each second
```

Run it from that folder:

```powershell
python my_program.py
```

Creating `SenseHat()` starts the browser interface automatically. Open the
address printed in the terminal (usually
[http://localhost:8000/](http://localhost:8000/)). Keep your Python program
running while using the interface: it displays the LEDs and lets you change
simulated sensor values and press the joystick. If port 8000 is already in
use, the simulator tries the next available port and prints the address it
selected.

You can read joystick events from your program:

```python
event = hat.stick.wait_for_event()
print(event.direction, event.action)
```

The browser joystick can also be used with the arrow keys and Enter when the
page has no slider focused.

### Change simulated readings in code

The browser controls and your Python program use the same simulated values.
You can set them directly or update several together:

```python
hat.sim.update(temperature=21.5, humidity=55)
hat.sim.update(orientation={"pitch": 20, "roll": 0, "yaw": 90})
```

By default, the simulated accelerometer and compass respond to the board's
orientation. These readings are approximations; they are not measurements of
your computer or a physical device.

## Try the camera

The camera simulator uses webcam **index 0** by default. A minimal example:

```python
from picamera2 import Picamera2

camera = Picamera2()
try:
    camera.configure(
        camera.create_preview_configuration(main={"size": (640, 480)})
    )
    camera.start()
    frame = camera.capture_array()
    print("Captured frame shape:", frame.shape)
finally:
    camera.close()
```

If the webcam cannot be opened, `Picamera2()` raises an error. Close other
programs that may be using the camera and check your operating system's camera
permissions. To use a video file instead, set `picamera2.CAMERA_ID` to its
path before creating the camera:

```python
import picamera2

picamera2.CAMERA_ID = r"C:\Videos\sample.mp4"
camera = picamera2.Picamera2()
```

The supported camera features include configuration, start/stop, frame and
image capture, and basic image transforms. Camera controls, live preview,
recording, and additional streams are not simulated. See
[`picamera2/__init__.py`](picamera2/__init__.py) for the supported details.

## What is simulated

The `sense_hat` folder provides common LED, text, sensor, colour sensor, and
joystick operations. The browser interface lets you inspect and change the
simulated environment while your code is running.

The `picamera2` folder uses OpenCV to read webcam or video-file frames. It
provides a subset of the Picamera2 API; it does not operate a Raspberry Pi
camera module.

Not every feature or behavior of the official Raspberry Pi libraries is
implemented. Sensor values are generated by the simulator, and hardware
timing, sensor noise, camera output, and orientation behavior can differ from
real devices. Check your program on the target Raspberry Pi before relying on
it there.

## Project files

- [`sense_hat/`](sense_hat/) — Sense HAT simulator and browser interface.
- [`picamera2/`](picamera2/) — webcam-backed Picamera2 subset.
- [`test.py`](test.py) — example combining webcam frames and the LED matrix.

## Generative-AI usage

Portions of this project were generated with assistance from Claude and GitHub Copilot.