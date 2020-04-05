import numpy as np
from numpy.random import randn
from math import *
import matplotlib.pyplot as plt
from robot import *
from lasersensor import *
from radarsensor import *
from helper import *
from kalman_filter import *

world_size = 100
dt = 1

H_laser = np.matrix([[1, 0, 0, 0],
                [0, 1, 0, 0]], dtype='float')

F = np.matrix([[1, 0, dt, 0],
               [0, 1, 0, dt],
               [0, 0, 1, 0 ],
               [0, 0, 0, 1 ]], dtype='float')


P = np.matrix([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1000, 0], 
               [0, 0, 0, 1000]], dtype='float')

R = 1 * np.eye(2)

x = np.array([[0],[0],[0],[0]]) # initial condition
u = np.array([[0],[0],[0],[0]]) # input 

true_x = []
true_y = []
true_vx = []
true_vy = []
lzs  = []
rzs_polar = []
rzs_cartesian  = []
target = robot(world_size)
target.set(0,0,pi/8)
lsensor = lasersensor()
rsensor = radarsensor()
iter = 500
    
# before filtering process
for t in range(iter):
    target.move(0.0025, 0.25) # target move
    
    lzs.append(lsensor.measure(target))
    
    rz = rsensor.measure(target)
    rzs_polar.append(rz)
    rho = rz[0][0]
    phi = rz[1][0]
    rhodot = rz[2][0]
    rzs_cartesian.append(polar_to_cartesian(rho, phi, rhodot))
    true_x.append(target.x)
    true_y.append(target.y)
           
# print(lzs)
kf = kalman_filter(F, H_laser, P, R, dt)
kf.filter(x,P,lzs)    
# states, P = kf.filter(x,P,lzs)
# print(states)
# print(kf.filter(x,P,lzs)[1])
plot_kf_position(true_x,true_y,lzs,kf)    

px, py = calculate_RMSE(kf.predict_lx, kf.predict_ly, true_x, true_y)  
print(px)
# print(zip([kf.predict_lx, kf.predict_ly], [true_x, true_y]))
# print(len(kf.predict_lx))