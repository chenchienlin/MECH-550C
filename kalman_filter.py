import numpy as np

class kalman_filter(object):
    def __init__(self,F, H, P, R, dt):
        self.F = F
        self.H = H
        self.P = P
        self.R = R 
        self.Q = self.setQ(dt)
        self.update_lx = []
        self.update_ly = []
        self.predict_lx = []
        self.predict_ly = []
        
    def filter(self,x, P, lzs):
        for n in range(len(lzs)):          
            # prediction
            x = self.F * x 
            P = self.F * P * self.F.T + self.Q

            # store predict data
            self.predict_lx.append(x[0, 0])
            self.predict_ly.append(x[1, 0])
        
            # measurement update 
            z = lzs[n]

            # error calculation
            y = z - self.H * x

            # S matrix
            S = self.H * P * self.H.T + self.R

            # kalman gain
            K = P * self.H.T * S.I
            P = (np.eye(4) - (K * self.H)) * P
            x = x + K * y
            
            # store measurement update data
            self.update_lx.append(x[0, 0])
            self.update_ly.append(x[1, 0])


        return x, P
    

    
    def fusion(self,x, P, z):
            # prediction
            x = self.F * x # + u 
            P = self.F * P * self.F.T + self.Q
            
            # store predict data
            self.predict_lx.append(x[0, 0])
            self.predict_ly.append(x[1, 0])
        
            # measurement update
            # error calculation
            y = z - self.H * x

            # S matrix
            S = self.H * P * self.H.T + self.R

            # kalman gain
            K = P * self.H.T * S.I
            P = (np.eye(4) - (K * self.H)) * P
            x = x + K * y
            
            # store measurement update data
            self.update_lx.append(x[0, 0])
            self.update_ly.append(x[1, 0])
            
            return x, P
        
    def setQ(self,dt):
        dt2 = dt * dt
        dt3 = dt * dt2
        dt4 = dt * dt3

        sig_x = 9
        sig_y = 9
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