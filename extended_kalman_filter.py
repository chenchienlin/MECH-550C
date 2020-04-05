import numpy as np
from helper import *

class extended_kalman_filter(object):
    def __init__(self,F, P, R, dt):
        self.F = F
        self.P = P
        self.Q = self.setQ(dt)
        self.R = R
        self.update_rx = []
        self.update_ry = []
        self.predict_rx = []
        self.predict_ry = [] 
        
    def filter(self,x, P, rzs_polar):
        # for n in range(2):
        for n in range(len(rzs_polar)):
            #predict
            x = self.F * x # + u
            P = self.F * P * self.F.T + self.Q
            self.predict_rx.append(x[0, 0])
            self.predict_ry.append(x[1, 0])
            
            # measurement update
            z = rzs_polar[n]

            px, py, vx, vy = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
            rho, phi, drho = cartesian_to_polar(px, py, vx, vy)
            
            H = calculate_jacobian(px, py, vx, vy)
            Hx = (np.matrix([[rho, phi, drho]])).T
            y = z - Hx
            PHt = P * H.T
            S = H * PHt + self.R    
            K = PHt * (S.I)
            x = x + K * y
            
            self.update_rx.append(x[0, 0])
            self.update_ry.append(x[1, 0])
            
            P = (np.eye(4) - (K * H)) * P
        return x,P
    
    def fusion(self,x, P, z):
        #predict
        x = self.F * x # + u
        P = self.F * P * self.F.T + self.Q
        self.predict_rx.append(x[0, 0])
        self.predict_ry.append(x[1, 0])
        
        # measurement update
        px, py, vx, vy = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
        rho, phi, drho = cartesian_to_polar(px, py, vx, vy)
        
        H = calculate_jacobian(px, py, vx, vy)
        Hx = (np.matrix([[rho, phi, drho]])).T
        y = z - Hx
        PHt = P * H.T
        S = H * PHt + self.R    
        K = PHt * (S.I)
        x = x + K * y
        
        self.update_rx.append(x[0, 0])
        self.update_ry.append(x[1, 0])
        
        P = (np.eye(4) - (K * H)) * P
        return x,P
    
    def setQ(self,dt):
        dt2 = dt * dt
        dt3 = dt * dt2
        dt4 = dt * dt3

        sig_x = 2
        sig_y = 2
        r11 = dt4 * sig_x / 4
        r13 = dt3 * sig_x / 2
        r22 = dt4 * sig_y / 4
        r24 = dt3 * sig_y /  2
        r31 = dt3 * sig_x / 2 
        r33 = dt2 * sig_x
        r42 = dt3 * sig_y / 2
        r44 = dt2 * sig_y

        Q = np.matrix([[r11, 0, r13, 0],
                        [0, r22, 0, r24],
                        [r31, 0, r33, 0], 
                        [0, r42, 0, r44]], dtype='float')
        return Q