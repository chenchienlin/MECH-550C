from numpy.random import randn
import random
from math import *
class lasersensor():
    def __init__(self, x_std = 0.15, y_std = 0.15):
        self.x_std = x_std
        self.y_std = y_std
        
    def measure(self, target):
        zx = target.x + random.gauss(0.0, self.x_std)
        zy = target.y + random.gauss(0.0, self.y_std)

        return [[zx],
                [zy]]
        
    def measure＿landmark(self, car, landmark):
        zx = car.x - landmark[0] + random.gauss(0.0, self.x_std)
        zy = car.y - landmark[1] + random.gauss(0.0, self.y_std)
        dist = sqrt((zx) ** 2 + (zy) ** 2)

        return dist
    
    def measure＿target(self, car, target):
        zx = car.x - target.x + random.gauss(0.0, self.x_std)
        zy = car.y - target.y + random.gauss(0.0, self.y_std)

        return zx, zy