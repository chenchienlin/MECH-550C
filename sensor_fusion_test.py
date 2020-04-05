import numpy as np
from numpy.random import randn
from math import *
import matplotlib.pyplot as plt
from robot import *
from lasersensor import *
from radarsensor import *
from helper import *
from kalman_filter import *
from extended_kalman_filter import *

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

R_laser = 100 * np.eye(2)
R_radar = 100 * np.matrix([[0.09, 0, 0], 
                     [0, 0.0009, 0], 
                     [0, 0, 0.09]], dtype='float')

x = np.array([[0],[0],[0],[0]]) # initial condition
u = np.array([[0],[0],[0],[0]]) # input 

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

update_fusion_x = []
update_fusion_y = []
measurement_x = []
measurement_y = []
kf = kalman_filter(F,H_laser,P,R_laser,dt)
ekf = extended_kalman_filter(F,P,R_radar,dt)
def fusion_process(x, iter, lzs, rzs_polar, P):
    
  for n in range(iter):
    if n % 2 == 0:
      # measurement update
      z = lzs[n]
      measurement_x.append(z[0][0])
      measurement_y.append(z[1][0])

      x, P = kf.fusion(x, P, z)
      
      # store update data
      update_fusion_x.append(x[0, 0])
      update_fusion_y.append(x[1, 0])
      
    elif n % 2 ==1:
      # measurement update
      z = rzs_polar[n]
      measurement_x.append(rzs_cartesian[n][0])
      measurement_y.append(rzs_cartesian[n][1])
      
      x, P = ekf.fusion(x, P, z)
      
      # store update data
      update_fusion_x.append(x[0, 0])
      update_fusion_y.append(x[1, 0])

fusion_process(x, iter, lzs, rzs_polar,P)
# fusion_process(x, P)
plt.scatter(true_x, true_y, color = 'green', s = 2, label = 'true_position') # ground truth
plt.scatter(measurement_x, measurement_y, color = 'black', s = 2, label = 'measurement') # measurement      
# plt.scatter(predict_fusion_x, predict_fusion_y, color = 'blue', s = 2, label = 'predict_states') # predict
plt.scatter(update_fusion_x, update_fusion_y, color = 'red', s = 2, label = 'update_states') # predict
# plt.legend()
plt.title('Sensor Fusion')
plt.show()      
    
px, py = calculate_RMSE(update_fusion_x, update_fusion_y, true_x, true_y)  
print(px, py)