import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt

I_b = (10**6)*np.array([[319, 0, 0],
               [0, 420, 0],
               [0, 0, 521]])
omega_0 = np.array([0.5, -0.5, 0.5]).T # 
q_0 = np.array([0,0,0,1]).T
T = 0.1
N=10



def model_linearization(x):
    A = np.array([[0, x[6]/2, -x[5]/2, x[4]/2, x[3]/2, -x[2]/2, x[1]/2],
               [-x[6], 0, x[6]/2, x[5]/2, x[2]/2, x[3]/2, -x[0]/2],
               [x[5]/2, -x[4]/2, 0, x[6]/2, -x[1]/2, x[0]/2, x[3]/2],
               [-x[4]/2, -x[5]/2, x[6]/2, 0, -x[0]/2, -x[1]/2, x[2]/2],
               [0, 0, 0, 0, 0, I_b[0][0]**(-1)*(I_b[2][2]-I_b[1][1])*x[6], I_b[0][0]**(-1)*(I_b[2][2]-I_b[1][1])*x[5]],
               [0, 0, 0, 0, I_b[1][1]**(-1)*(I_b[0][0]-I_b[2][2])*x[6], 0, I_b[1][1]**(-1)*(I_b[0][0]-I_b[2][2])*x[4]],
               [0, 0, 0, 0, I_b[2][2]**(-1)*(I_b[1][1]-I_b[0][0])*x[5], I_b[2][2]**(-1)*(I_b[1][1]-I_b[0][0])*x[4], 0]])
    B = np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [I_b[0][0]**(-1), 0, 0],
                [0, I_b[1][1]**(-1), 0],
                [0, 0, I_b[2][2]**(-1)]])
    C = 0
    
    return A, B, C

def model_sim():
    model = pyo.ConcreteModel()

    model.x1 = pyo.Var(initialize = q_0[0])
    model.x2 = pyo.Var(initialize = q_0[1])
    model.x3 = pyo.Var(initialize = q_0[2])
    model.x4 = pyo.Var(initialize = q_0[3])
    model.x5 = pyo.Var(initialize = omega_0[0])
    model.x6 = pyo.Var(initialize = omega_0[1])
    model.x7 = pyo.Var(initialize = omega_0[2])

    model.constraint1 = pyo.Constraint(expr = (pyo.sqrt(model.x1**2 + model.x2**2 + model.x3**2 + model.x4**2) == 1)) #unit quaternion
    



#    model.N = N
#    model.nx = np.size(A, 0)
#    model.nu = np.size(B, 1)
#
##Kuk og balle ;)
#
#    # length of finite optimization problem:
#    model.tIDX = pyo.Set( initialize= range(model.N+1), ordered=True )
#    model.xIDX = pyo.Set( initialize= range(model.nx), ordered=True )
#    model.uIDX = pyo.Set( initialize= range(model.nu), ordered=True )
#    
    return

model_sim()
