import RPi.GPIO as GPIO
from time import sleep
import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


class MovementTest():

    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        self.sleepPin = 16
        self.stepPin = 20
        self.dirPin = 21
        GPIO.setup(self.stepPin, GPIO.OUT)
        GPIO.setup(self.dirPin, GPIO.OUT)
        GPIO.setup(self.sleepPin, GPIO.OUT)
        GPIO.output(self.stepPin, GPIO.HIGH)
        GPIO.output(self.dirPin, GPIO.HIGH)
        GPIO.output(self.sleepPin, GPIO.LOW)
        self.stephigh = False
        GPIO.output(self.sleepPin, GPIO.HIGH)

        self.rangei = 1000
        self.x = 0  # not in reset to prevent exceeding boundaries

    def run(self):
        direct = 1
        GPIO.output(self.sleepPin, GPIO.HIGH)
        while True:
            # step
            GPIO.output(self.stepPin, GPIO.HIGH if self.stephigh else GPIO.LOW)
            self.stephigh = not self.stephigh
            sleep(0.0005)
            if abs(self.x + direct) <= self.rangei:
                dirVoltage = GPIO.HIGH if direct == 1 else GPIO.LOW
                GPIO.output(self.dirPin, dirVoltage)
                self.x += direct
            else:
                direct *= -1
        GPIO.output(self.sleepPin, GPIO.LOW)


MovementTest().run()
