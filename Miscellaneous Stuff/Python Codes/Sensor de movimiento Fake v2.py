import time
import random

def Sensor():
    time.sleep(0.40)
    return random.random()

def ControlSensor():
    while True:
        r = Sensor()
        if (r > 0.8):
            print("===Ha habido movimiento + ===")
        else:
            print("===No hubo movimiento - ===")

if __name__ == "__main__":
    ControlSensor()
