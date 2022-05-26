import math
import threading

import RPi.GPIO as GPIO
from time import sleep
import gym
import numpy as np
import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

from settings import gv


# interface to a robot by using the OpenAI Gym

class RealPoleBalancing(gym.Env):

    def __init__(self, absolute_observation=False):
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

        # Create the I2C bus
        i2c = busio.I2C(board.SCL, board.SDA)

        # Create the ADC object using the I2C bus
        ads = ADS.ADS1115(i2c)

        # Create single-ended input on channel 0
        self.sensor = AnalogIn(ads, ADS.P0)
        self.rangei = 1500
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.thread = None
        self.reset()
        self.x = self.rangei // 2  # not in reset to prevent exceeding boundaries
        self.initmotorzero()

    def initmotorzero(self):
        GPIO.setup(14, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        direct = -1
        print("Finding zero position.")
        while GPIO.input(14):
            # step
            GPIO.output(self.stepPin, GPIO.HIGH if self.stephigh else GPIO.LOW)
            self.stephigh = not self.stephigh
            sleep(0.003)
            dirVoltage = GPIO.HIGH if direct == 1 else GPIO.LOW
            GPIO.output(self.dirPin, dirVoltage)
        # found the border
        self.x = int(self.rangei * 1.2)  # add safety margin of 10%
        self.moveto(self.rangei // 2)

    def reset(self):
        self.lasttheta = 0
        self.state = (0, 0, 0, 0)
        return np.array(self.state)

    def step(self, action: list[int]):
        direct = 1 if action else -1
        # non-blocking call
        if self.thread is not None:
            self.thread.do_run = False
            self.thread.join()
        self.thread = threading.Thread(target=self.move, args=(direct,))
        self.thread.start()
        theta = (self.sensor.value - 12500) / 440 * 2 * math.pi  # full sensor range 0-25k, to radians 440~2pi
        theta_dot = theta - self.lasttheta if self.lasttheta != 0 else 0
        done = theta > self.theta_threshold_radians or abs(self.x + 1) >= self.rangei
        self.state = (self.x, direct, theta, theta_dot)
        return np.array(self.state), 1, done, {}

    def moveto(self, goalpos):
        print("moving to " + str(goalpos))
        while self.x != goalpos:
            # step
            GPIO.output(self.stepPin, GPIO.HIGH if self.stephigh else GPIO.LOW)
            self.stephigh = not self.stephigh
            sleep(0.001)
            print(f"{self.x}->{goalpos}\r", end="")
            dir = -1 if (self.x - goalpos) else 1
            self.x += dir
            dirVoltage = GPIO.HIGH if dir == -1 else GPIO.LOW
            GPIO.output(self.dirPin, dirVoltage)

    def move(self, direct):
        for i in range(40):
            # stop thread when required
            t = threading.currentThread()
            if abs(self.x + direct) >= self.rangei or not getattr(t, "do_run", True):
                print("limit " + str(self.x))
                return False
            # step
            GPIO.output(self.stepPin, GPIO.HIGH if self.stephigh else GPIO.LOW)
            self.stephigh = not self.stephigh
            sleep(0.0006)
            dirVoltage = GPIO.HIGH if direct == 1 else GPIO.LOW
            GPIO.output(self.dirPin, dirVoltage)
            self.x += direct
            return True

    def render(self, mode='human'):
        print("Currently no rendered implemented")
