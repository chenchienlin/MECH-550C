import numpy as np
from numpy.random import randn
from math import *
from robot import *
from lasersensor import *
from radarsensor import *
from helper import *
from extended_kalman_filter import *


dt = 1

H = np.matrix([[1, 0, 0, 0],
                [0, 1, 0, 0]], dtype='float')
F = np.matrix([[1, 0, dt, 0],
               [0, 1, 0, dt],
               [0, 0, 1, 0 ],
               [0, 0, 0, 1 ]], dtype='float')

P = np.matrix([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1000, 0], 
               [0, 0, 0, 1000]], dtype='float')

R_radar = 1000 * np.matrix([[0.09, 0, 0], 
                     [0, 0.0009, 0], 
                     [0, 0, 0.09]], dtype='float')

x = [[0.01],[0.01],[0.01], [0.01]] # this is for ekf

true_x = []
true_y = []
true_vx = []
true_vy = []
lzs  = []
rzs_polar = []
rzs_cartesian  = []
target = robot()
target.set(0,0,pi/8)
lsensor = lasersensor()
rsensor = radarsensor()
iter = 250
# before filtering process
for t in range(iter):
    target = target.move(0.005 , 0.5) # target move
    
    lzs.append(lsensor.measure(target))
    
    rz = rsensor.measure(target)
    rzs_polar.append(rz)
    rho = rz[0][0]
    phi = rz[1][0]
    rhodot = rz[2][0]
    rzs_cartesian.append(polar_to_cartesian(rho, phi, rhodot))
    true_x.append(target.x)
    true_y.append(target.y)
    
ekf = extended_kalman_filter(F, P, R_radar, dt)        
ekf.filter(x,P,rzs_polar)
plot_ekf_position(true_x,true_y,rzs_cartesian,ekf)
px, py = calculate_RMSE(ekf.predict_rx, ekf.predict_ry, true_x, true_y)  
print(px)