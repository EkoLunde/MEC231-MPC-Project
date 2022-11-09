import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt

I_b = (10**6)*np.array([319, 0, 0],
               [0, 420, 0],
               [0, 0, 521])
omega_0 = np.array([0.5, -0.5, 0.5]).T
q_0 = np.array([0,0,0,1]).T
T = 0.1
N=10

def model_sim():
    model = pyo.ConcreteModel()
    model.N = N
    model.nx = np.size(7, 0)
    model.nu = np.size(3, 1)



    # length of finite optimization problem:
    model.tIDX = pyo.Set( initialize= range(model.N+1), ordered=True )
    model.xIDX = pyo.Set( initialize= range(model.nx), ordered=True )
    model.uIDX = pyo.Set( initialize= range(model.nu), ordered=True )
