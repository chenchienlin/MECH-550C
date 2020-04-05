import csv
import random
from math import *
import numpy as np
from robot import *
from particle import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from matplotlib import pyplot as plt1

class partile_filter():
    def __init__(self, target, pt_number, landmarks, gps_error):
        self.target = target
        self.landmarks = landmarks
        self.pt_number = pt_number
        self.gps_error = gps_error
        self.x = 0.0
        self.y = 0.0
        self.pts = []
        self.residual = []
        for i in range(pt_number):
            pt = particle(self.target)
            pt.set_noise(0.05, 0.05, 2.0)
            self.pts.append(pt)
    
    def filter(self, turn, forward, z):
        
        for i in range(self.pt_number):
            self.pts[i] = self.pts[i].move(turn,forward)

        w = []
        for i in range(self.pt_number):
            w.append(self.pts[i].measurement_prob(z, self.landmarks))
        
        p3 = []
        X = []
        Y = []
        index = int(random.random() * self.pt_number)
        beta = 0.0
        mw = max(w)
        
        for i in range(self.pt_number):
            beta += random.random() * 2.0 * mw
            while beta > w[index]:
                beta -= w[index]
                index = (index + 1) % self.pt_number
            p3.append(self.pts[index])
            X.append(self.pts[index].x)
            Y.append(self.pts[index].y)
            
        self.pts = p3
        mu_x = np.mean(X)
        mu_y = np.mean(Y)
        res = self.eval()
        self.x = mu_x
        self.y = mu_y
        self.residual.append(res)

    def eval(self):
        sum = 0.0
        for i in range(len(self.pts)): # calculate mean error
            dx = (self.pts[i].x - self.target.x + (self.target.world_size/2.0)) % self.target.world_size - (self.target.world_size/2.0)
            dy = (self.pts[i].y - self.target.y + (self.target.world_size/2.0)) % self.target.world_size - (self.target.world_size/2.0)
            err = sqrt(dx * dx + dy * dy)
            sum += err
        return sum / float(len(self.pts))
