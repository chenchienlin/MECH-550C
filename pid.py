from car import *
from math import *
from helper import *
from particle import *
from robot import *
from lasersensor import *
from radarsensor import *
from particle_filter import *
from kalman_filter import *
from extended_kalman_filter import *
import matplotlib.pyplot as plt
from random import seed
from random import randint

def pid_steer(target, car, params, dt, x, P, n = 100, speed = 0.5):
    car.set_steering_drift(10.0 / 180.0 * pi)
    zx, zy =  car.lsensor.measure_target(car, target)
    target_x = []
    target_y = []
    car_x = []
    car_y = []
    
    localizer_x = []
    localizer_y = []

    prev_cte = zy
    int_cte = 0
    
    seed(1)
    direction = 1

    for i in range(n):
        
        # target move
        target.move(0.005, speed)
        # laser measurement
        zx, zy =  car.lsensor.measure_target(car, target)
        # radar measurement
        zrho, zphi, zrhodot = car.rsensor.measure(target)

        # random number
        rand = randint(1,2)
        
        # fuse sensor signals
        if rand == 1:
            z = [[zx],[zy]]
            x, P = kf.fusion(x, P, z)
        
        elif n % 2 ==1:
            z =  [[zrho],[zphi],[zrhodot]]
            x, P = ekf.fusion(x, P, z)
            
        err_x = x[0, 0]
        err_y = x[1, 0]
        
        # cross track error 
        angle =  car.orientation * 180 / pi

        if err_y <= 0:
            if angle >= 270 or angle < 90:
                cte = -1*sqrt(err_x**2 + err_y**2)
            else:
                cte = 1*sqrt(err_x**2 + err_y**2)
        elif err_y > 0:
            if angle >= 270 or angle < 90:
                cte = 1*sqrt(err_x**2 + err_y**2)
            else:
                cte = -1*sqrt(err_x**2 + err_y**2)
                
        diff_cte = cte - prev_cte
        int_cte += cte
        prev_cte = cte
        steer = -params[0] * cte - params[1] * diff_cte - params[2] * int_cte
        
        
        car.move(steer, speed)
        car_x.append(car.x)
        car_y.append(car.y)
        
        target_x.append(target.x)
        target_y.append(target.y)

        localizer_x.append(car.localizer.x)
        localizer_y.append(car.localizer.y)
        
    return car_x, car_y, localizer_x, localizer_y, target_x, target_y


landmarks  = [[20.0, 20.0], [80.0, 80.0], [20.0, 80.0], [80.0, 20.0]]  
world_size = 100
target = robot(world_size)
target.set(1.0,1.0,0)
# p = [3.0, 5.3, 0.012]
# p = [30.0, 5000, 0.001]
p = [0.006362685441135942, 1.0118527900307346, 0.0]
car = Car(landmarks)
car.set(0.0 ,0.0, 0.0)

# create kalman filter
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

x = np.array([[0.01],[0.01],[0.01],[0.01]]) # initial condition
u = np.array([[0],[0],[0],[0]]) # input 

kf = kalman_filter(F,H_laser,P,R_laser,dt)
ekf = extended_kalman_filter(F,P,R_radar,dt)

car_x, car_y, localizer_x, localizer_y, target_x, target_y= pid_steer(target, car, p, x = x, P = P, dt = 1)

plt.scatter(target_x, target_y, s = 5 , color = 'green', alpha = 0.7, label = 'target_trajectory')
plt.scatter(car_x, car_y, s = 10 , edgecolors = 'blue', alpha = 0.9, label = 'car_trajectory')
plt.scatter(localizer_x,localizer_y,s = 20 , facecolors='none', edgecolors = 'black', alpha = 0.4, label = 'localizer')
    
plt.legend()
plt.show()