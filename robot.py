from math import *
import random
import csv
from matplotlib.pyplot import plot
from matplotlib import pyplot as plt1

class robot():
    def __init__(self, world_size):
        self.world_size = world_size
        self.x = random.random() * world_size 
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi
        self.forward_noise = 0.0 #Add Gausian Noise
        self.turn_noise    = 0.0
        self.sense_noise   = 0.0
        self.dist = 0
    
    def set(self, new_x, new_y, new_orientation):
        if new_x < 0 or new_x >= self.world_size:
            raise ValueError('X coordinate out of bound')
        if new_y < 0 or new_y >= self.world_size:
            raise ValueError('Y coordinate out of bound')
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError('Orientation must be in [0..2pi]')
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
    
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.forward_noise = float(new_f_noise)
        self.turn_noise    = float(new_t_noise)
        self.sense_noise   = float(new_s_noise)
    
    def move(self, turn, forward):
        if forward < 0:
            raise ValueError('Robot cant move backwards')         
        
        # turn, and add randomness to the turning command
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * pi
        
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + (cos(orientation) * dist)
        y = self.y + (sin(orientation) * dist)
        x %= self.world_size    # cyclic truncate
        y %= self.world_size
        
        self.x = float(x)
        self.y = float(y)
        self.orientation = float(orientation)
