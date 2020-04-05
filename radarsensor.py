from numpy.lib.scimath import sqrt
from math import *
from numpy.core.umath import arctan
import random

class radarsensor():

    def __init__(self, rho_std = 0.5, phi_std = 0.05, rhodot_std = 0.5):
        
        self.rho_std = rho_std
        self.phi_std = phi_std
        self.rhodot_std = rhodot_std
        self.zrho_old = 0
        self.zrho_new = 0
        self.zphi = 0
        self.zrhodot = 0
        
        
    def measure(self, target):

        self.zrho_new = sqrt(target.x ** 2 + target.y **2 ) 
        + random.gauss(0.0, self.rho_std)
        
        self.zphi = atan2(target.y, target.x) 
        + random.gauss(0.0, self.phi_std)
        
        self.zrhodot = self.zrho_new - self.zrho_old 
        + random.gauss(0.0, self.rhodot_std)
        
        self.zrho_old = self.zrho_new
        
        return [[self.zrho_new],
                [self.zphi],
                [self.zrhodot]]
    
    def measure_target(self, car, target):

        self.zrho_new = sqrt((car.x - target.x) ** 2 + (car.y - target.y) **2 ) 
        + random.gauss(0.0, self.rho_std)
        
        self.zphi = atan2((car.y - target.y), (car.x - target.x) ) 
        + random.gauss(0.0, self.phi_std)
        
        self.zrhodot = self.zrho_new - self.zrho_old 
        + random.gauss(0.0, self.rhodot_std)
        
        self.zrho_old = self.zrho_new
        
        return self.zrho_new, self.zphi, self.zrhodot