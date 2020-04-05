from math import *
import numpy as np
import matplotlib.pyplot as plt

def get_col(matrix, i):
    return [row[i] for row in matrix]

def calculate_jacobian(px, py, vx, vy, THRESH = 0.0001, ZERO_REPLACEMENT = 0.0001):
  """
    Calculates the Jacobian given for four state variables

    Args:
      px, py, vx, vy : floats - four state variables in the system 
      THRESH - minimum value of squared distance to return a non-zero matrix
      ZERO_REPLACEMENT - value to replace zero to avoid division by zero error

    Returns:
      H : the jacobian matrix expressed as a 4 x 4 numpy matrix with float values
  """
    
  d_squared = px * px + py * py 
  d = sqrt(d_squared)
  d_cubed = d_squared * d
  
  if d_squared < THRESH:
 
    print("WARNING: in calculate_jacobian(): d_squared < THRESH")
    H = np.matrix(np.zeros([3, 4]))
 
  else:

    r11 = px / d
    r12 = py / d
    r21 = -py / d_squared
    r22 = px / d_squared
    r31 = py * (vx * py - vy * px) / d_cubed
    r32 = px * (vy * px - vx * py) / d_cubed
  
    H = np.matrix([[r11, r12, 0, 0], 
                  [r21, r22, 0, 0], 
                  [r31, r32, r11, r12]], dtype='float')

  return H

def cartesian_to_polar(array, THRESH = 0.0001):
  """   
  Converts 2d cartesian position and velocity coordinates to polar coordinates

  Args:
    x, y, vx, vy : floats - position and velocity components in cartesian respectively 
    THRESH : float - minimum value of rho to return non-zero values
  
  Returns: 
    rho, drho : floats - radius and velocity magnitude respectively
    phi : float - angle in radians
  """  
  x = array[0][0]
  y = array[1][0]  
  vx = array[2][0]  
  vy = array[3][0]  
    
  rho = sqrt(x * x + y * y)
  phi = np.arctan2(y, x)
  
  
  if rho < THRESH:
    print("WARNING: in cartesian_to_polar(): d_squared < THRESH")
    rho, phi, drho = 0, 0, 0
  else:
    drho = (x * vx + y * vy) / rho
    
  return np.array([[rho], [phi], [drho]])

def cartesian_to_polar(x, y, vx, vy, THRESH = 0.0001):
  """   
  Converts 2d cartesian position and velocity coordinates to polar coordinates

  Args:
    x, y, vx, vy : floats - position and velocity components in cartesian respectively 
    THRESH : float - minimum value of rho to return non-zero values
  
  Returns: 
    rho, drho : floats - radius and velocity magnitude respectively
    phi : float - angle in radians
  """  

  rho = sqrt(x * x + y * y)
  phi = np.arctan2(y, x)
  
  
  if rho < THRESH:
    print("WARNING: in cartesian_to_polar(): d_squared < THRESH")
    rho, phi, drho = 0, 0, 0
  else:
    drho = (x * vx + y * vy) / rho
    
  return rho, phi, drho

def polar_to_cartesian(rho, phi, drho):
  """
  Converts 2D polar coordinates into cartesian coordinates

  Args:
    rho. drho : floats - radius magnitude and velocity magnitudes respectively
    phi : float - angle in radians

  Returns:
    x, y, vx, vy : floats - position and velocity components in cartesian respectively
  """
 
  x, y = rho * np.cos(phi), rho * np.sin(phi)
  vx, vy = drho * np.cos(phi) , drho * np.sin(phi)
  return x, y, vx, vy

def plot_ekf_position(true_x,true_y,rzs_cartesian,ekf):
    plt.scatter(true_x, true_y, color = 'green', s = 2, label = 'true_position') # ground truth
    plt.plot(true_x, true_y, color = 'green', linewidth = 1)
    plt.scatter(get_col(rzs_cartesian,0),get_col(rzs_cartesian,1), color = 'black', s = 2, label = 'measurement') # measurement
    plt.scatter(ekf.predict_rx, ekf.predict_ry,s = 2, color = 'blue', label = 'predict_states') # filtered
    plt.plot(ekf.predict_rx, ekf.predict_ry, color = 'b', linewidth = 1)
    plt.scatter(ekf.update_rx, ekf.update_ry,s = 2, color = 'r', label = 'update_states') # filtered
    plt.plot(ekf.update_rx, ekf.update_ry, color = 'b', linewidth = 1)
    plt.legend()
    plt.title('Radar')
    plt.show()

def plot_kf_position(true_x,true_y,lzs,kf):
    plt.scatter(true_x, true_y, color = 'green', s = 2, label = 'true_position') # ground truth
    plt.plot(true_x, true_y, color = 'green', linewidth = 1)
    plt.scatter(get_col(lzs,0),get_col(lzs,1), color = 'black', s = 2, label = 'measurement') # measurement
    plt.scatter(kf.predict_lx, kf.predict_ly,s = 2, color = 'b', label = 'predict_states') # filtered
    plt.plot(kf.predict_lx, kf.predict_ly, color = 'b', linewidth = 1)
    plt.scatter(kf.update_lx, kf.update_ly,s = 2, color = 'r', label = 'update_states') # filtered
    plt.plot(kf.update_lx, kf.update_ly, color = 'r', linewidth = 1)
    plt.legend()
    plt.title('Laser')
    plt.show()
    
def calculate_RMSE(predict_x, predict_y, true_x, true_y):

    pxs, pys, vxs, vys = 0, 0, 0, 0

    for i in range(len(predict_x)):

        pxs += (predict_x[i] - true_x[i]) * (predict_x[i] - true_x[i])
        pys += (predict_y[i] - true_y[i]) * (predict_y[i] - true_y[i])
        # vxs += [(pvx - tvx) * (pvx - tvx)]
        # vys += [(pvy - tvy) * (pvy - tvy)]
        
        px, py = sqrt(pxs/len(predict_x)), sqrt(pys/len(predict_x))
    # return px, py, vx, vy
    return px, py        