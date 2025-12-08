import time
import random

def Sensor():
    time.sleep(0.30)
    return random.random()

def ControlSensor():
    while True:
        r = Sensor()
        if (r>0.5):
            print("===Se detecto movimiento===")
        

if __name__ == "___main__":
    ControlSensor()
