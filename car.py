from math import *
import random
import numpy as np
from helper import *
import matplotlib.pyplot as plt
from lasersensor import *
from radarsensor import *
from particle_filter import *

# ------------------------------------------------
# 
# this is the Car class
#

class Car(object):
    def __init__(self, landmarks, length=20.0, localizer = True, pt_number = 10000, gps_error = 10, world_size = 100):
        """
        Creates robot and initializes location/orientation to 0, 0, 0.
        """
        self.x = 0.0
        self.y = 0.0
        self.res = 0.0
        self.orientation = 0.0
        self.landmarks = landmarks
        self.length = length
        self.sense_noise  = 0.0
        self.steering_noise = 0.0
        self.distance_noise = 0.0
        self.steering_drift = 0.0
        self.pt_number = pt_number
        self.gps_error = gps_error
        self.world_size = world_size
        self.lsensor = lasersensor()
        self.rsensor = radarsensor()
        if localizer == True:
            self.localizer = partile_filter(self, pt_number, landmarks, gps_error)
        else:
            print("localizer is set to be False")

    def set(self, x, y, orientation):
        """
        Sets a robot coordinate.
        """
        self.x = x
        self.y = y
        self.orientation = orientation % (2.0 * np.pi)

    def set_steering_drift(self, drift):
        """
        Sets the systematical steering drift parameter
        """
        self.steering_drift = drift

    def move(self, steering, distance, tolerance=0.001, max_steering_angle=np.pi / 4.0):
        """
        steering = front wheel steering angle, limited by max_steering_angle
        distance = total distance driven, most be non-negative
        """
        if steering > max_steering_angle:
            steering = max_steering_angle
        if steering < -max_steering_angle:
            steering = -max_steering_angle
        if distance < 0.0:
            distance = 0.0

        # apply noise
        steering2 = random.gauss(steering, self.steering_noise)
        distance2 = random.gauss(distance, self.distance_noise)

        # apply steering drift
        steering2 += self.steering_drift
        
        # desired motion
        des_turn = np.tan(steering) * distance / self.length
        
        # actual motion
        act_turn = np.tan(steering2) * distance2 / self.length

        if abs(act_turn) < tolerance:
            # approximate by straight line motion
            self.x += distance2 * np.cos(self.orientation)
            self.y += distance2 * np.sin(self.orientation)
            self.orientation = (self.orientation + act_turn) % (2.0 * np.pi)
        else:
            # approximate bicycle model for motion
            radius = distance2 / act_turn
            cx = self.x - (np.sin(self.orientation) * radius)
            cy = self.y + (np.cos(self.orientation) * radius)
            self.orientation = (self.orientation + act_turn) % (2.0 * np.pi)
            self.x = cx + (np.sin(self.orientation) * radius)
            self.y = cy - (np.cos(self.orientation) * radius)
        
        Z = self.sense()
        self.localizer.filter(des_turn, distance,Z)
        
    def sense(self):
        Z2 = []
        for i in range(len(self.landmarks)):
            dist = self.lsensor.measure_landmark(self, self.landmarks[i])
            Z2.append(dist)
    
    
        return Z2
    def __repr__(self):
        return '[x=%.5f y=%.5f orient=%.5f]' % (self.x, self.y, self.orientation)