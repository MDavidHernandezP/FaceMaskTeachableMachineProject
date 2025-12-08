import dweepy

url = dweepy.get_latest_dweet_for('00001a57a387')

dict = url[0]

tiempo_en_línea = dict['content']['uptime']

print("El tiempo en línea actual es de",tiempo_en_línea,"segundos")

import time
import math
import random

def Sensor():
    time.sleep(0.40)
    return random.random()

def ControlSensor():
    while True:
        r = Sensor()
        if (tiempo_en_línea > 0.8):
            print("===Ah estado en línea + ===")
        else:
            print("===No ah estado en línea - ===")

if __name__ == "__main__":
    ControlSensor()
