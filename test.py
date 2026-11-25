from sense_hat import SenseHat
from picamera2 import Picamera2
import cv2 as cv
import time

# Initialize Sense HAT
hat = SenseHat()

# Initialize camera
picam = Picamera2()
picam.configure(picam.create_still_configuration(main={"size": (3280, 2464)}))
picam.start()

try:
    while True:
        # Grab frame
        frame = picam.capture_array(name="main")
        # Resize and convert to BGR
        frame = cv.resize(frame, (8, 8))
        # Display the frame on the Sense HAT LED matrix
        hat.set_pixels([tuple(pixel) for row in frame for pixel in row])
        # Get temperature and humidity
        temperature = hat.get_temperature()
        humidity = hat.get_humidity()
        print(f"Temperature: {temperature:.2f}°C, Humidity: {humidity:.2f}%")
        time.sleep(1)  # Delay for a second
except KeyboardInterrupt:
    print("Exiting...")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Stop the camera to free up resources
    picam.stop()