
import time
from sense_hat import SenseHat

hat = SenseHat()
hat.show_message("Hello!")

hat.set_pixel(0, 0, 255, 0, 0)  # x, y, red, green, blue

while True:
    print("Temperature:", hat.get_temperature(), "°C")
    print("Humidity:", hat.get_humidity(), "%")
    time.sleep(1)  # keep the program running and refresh readings each second