import numpy as np
import scipy.signal
import scipy.linalg
import polytope as pt
from utilities import *
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from model import *
from test3D import *

def dlqr(A, B, Q, R):
    # solve Discrete Algebraic Riccatti equation  
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = scipy.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

    # stability check 
    eigVals, eigVecs = scipy.linalg.eig(A - B @ K)
    return K, P, eigVals, eigVecs

def minkowski_sum(X, Y):

    # Minkowski sum between two polytopes based on 
    # vertex enumeration. So, it's not fast for the
    # high dimensional polytopes with lots of vertices.
    V_sum = []
    if isinstance(X, pt.Polytope):
        V1 = pt.extreme(X)
    else:
        # assuming vertices are in (N x d) shape. N # of vertices, d dimension
        V1 = X
        
    if isinstance(Y, pt.Polytope):
        V2 = pt.extreme(Y)
    else:
        V2 = Y

    for i in range(V1.shape[0]):
        for j in range(V2.shape[0]):
            V_sum.append(V1[i,:] + V2[j,:])
    return pt.qhull(np.asarray(V_sum))
    
def precursor(Xset, A, Uset=pt.Polytope(), B=np.array([])):
        if not B.any():
            return pt.Polytope(Xset.A @ A, Xset.b)
        else:
            tmp  = minkowski_sum( Xset, pt.extreme(Uset) @ -B.T )
        return pt.Polytope(tmp.A @ A, tmp.b)

def Oinf(A, Xset):
    
    Omega = Xset
    k = 0
    Omegap = precursor(Omega, A).intersect(Omega)
    while not Omegap == Omega:
        k += 1
        Omega = Omegap
        Omegap = pt.reduce(precursor(Omega, A).intersect(Omega))
    return Omegap

def create_polytope_x_and_u(xU,xL,uU,uL):
    X = pt.Polytope(np.array([[1.0, 0, 0, 0, 0, 0, 0], 
                              [0, 1.0, 0, 0, 0, 0, 0],
                              [0, 0, 1.0, 0, 0, 0, 0],
                              [0, 0, 0, 1.0, 0, 0, 0],
                              [0, 0, 0, 0, 1.0, 0, 0], 
                              [0, 0, 0, 0, 0, 1.0, 0], 
                              [0, 0, 0, 0, 0, 0, 1.0], 
                              [-1.0, 0, 0, 0, 0, 0, 0], 
                              [0, -1.0, 0, 0, 0, 0, 0],
                              [0, 0, -1.0, 0, 0, 0, 0],
                              [0, 0, 0, -1.0, 0, 0, 0],
                              [0, 0, 0, 0, -1.0, 0, 0], 
                              [0, 0, 0, 0, 0, -1.0, 0],  
                              [0, 0, 0, 0, 0, 0, -1.0]]), 
                np.array([[xU[0]], 
                          [xU[1]], 
                          [xU[2]], 
                          [xU[3]],
                          [xU[4]],
                          [xU[5]],
                          [xU[6]], 
                          [xL[0]], 
                          [xL[1]], 
                          [xL[2]], 
                          [xL[3]],
                          [xL[4]],
                          [xL[5]],
                          [xL[6]]])) 
    # input constraint
    U = pt.Polytope(np.array([[1.0, 0, 0], 
                              [0, 1.0, 0],
                              [0, 0, 1.0],
                              [-1.0, 0, 0], 
                              [0, -1.0, 0],
                              [0, 0, -1.0]]), 
                np.array([[uU[0]], 
                          [uU[1]], 
                          [uU[2]], 
                          [uL[0]], 
                          [uL[1]], 
                          [uL[2]]])) 
    return X, U
    
def mpc(Q, R, x0, I_b, N, M, xL, xU, uL, uU, Af, bf, Ts):
    A_c, B_c, C_c = model_linearization(x0, I_b)
    D_c = np.array(np.zeros((1,3)))
    system = (A_c, B_c, C_c, D_c)
    A, B, C, D, dt = scipy.signal.cont2discrete(system, Ts)     


    # Finf, Pinf, eigVals, eigVecs = dlqr(A, B, Q, R)

    # # Create some stability check using eig !!

    # Acl = A - B @ Finf

    # # constraint sets represented as polyhedra
    # # state constraint
    # X, U = create_polytope_x_and_u(xU,xL,uU,uL)

    # # remeber to convet input constraits in state constraints
    # S = X.intersect(pt.Polytope(U.A @ -Finf, U.b))

    # O_inf = Oinf(Acl, S)

    # Xf = O_inf
    # Af = Xf.A
    # bf = Xf.b

    nx = np.size(A, 0)
    nu = np.size(B, 1)
    
    xOpt = np.zeros((nx, M+1))
    uOpt = np.zeros((nu, M))
    xOpt[:, 0] = x0.reshape(nx, )

    xPred = np.zeros((nx, N+1, M))
    predErr = np.zeros((nx, M-N+1))

    feas = np.zeros((M, ), dtype=bool)
    xN = np.zeros((nx,1))

    for t in range(M):
        [model, feas[t], x, u, J] = solve_cftoc(A, B, P, Q, R, N, xOpt[:, t], xL, xU, uL, uU, bf, Af)

        if not feas[t]:
            #xOpt = []
            #uOpt = []
            #predErr = []
            print("infeasable at time ", t)
            #break
        # Save open loop predictions
        xPred[:, :, t] = x

        # Save closed loop trajectory
        # Note that the second column of x represents the optimal closed loop state
        xOpt[:, t+1] = x[:, 1]
        uOpt[:, t] = u[:, 0].reshape(nu, )
        A, B, C = model_linearization(x[:, 1],I_b)
        system = (A_c, B_c, C_c, D_c)
        A, B, C, D, dt = scipy.signal.cont2discrete(system, Ts)   

    return [model, feas, xOpt, uOpt]

I_b = (10**6)*np.array([[319, 0, 0],
               [0, 420, 0],
               [0, 0, 521]])
I_b_inv = np.linalg.inv(I_b)

omega_0 = np.array([0.5, -0.5, 0.5]).T #
q_0 = np.array([1,0,0,0]).T
x0 = np.concatenate((q_0.T, omega_0.T))
Ts = 0.1
N=50
M = 40   # Simulation steps

Q = np.eye(7)
R = np.eye(3) #10*np.array([1]).reshape(1,1)
P = Q
xL = np.array([-1.0, -1.0, -1.0, -1.0, -5.0, -5.0, -5.0]).T
xU = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0]).T
uL = np.array([-0.1, -0.1, -0.1]).T
uU = np.array([0.1, 0.1, 0.1]).T

Af = np.eye(7)
bf = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).T

#[model, feas, x, u] = mpc(Q, R, x0, I_b, N, M, xL, xU, uL, uU, Af, bf, Ts)

A_c, B_c, C_c = model_linearization(x0, I_b)
D_c = np.array(np.zeros((3,1)))
system = (A_c, B_c, C_c, D_c)
A, B, C, D, dt = scipy.signal.cont2discrete(system, Ts)  
[model, feas, x, u, J] = solve_cftoc(A, B, P, Q, R, N, x0, xL, xU, uL, uU, bf, Af)

#plt.plot(x.T)
#plt.ylabel('x')
#plt.legend((r'$\eta$',r'$q_1$',r'$q_2$',r'$q_3$',r'$\omega_1$',r'$\omega_2$',r'$\omega_3$'),)
#plt.grid()
#fig = plt.figure(figsize=(9, 6))
#plt.plot(u.T)
#plt.ylabel('u')
#plt.legend(("u1","u2","u3"))
#plt.grid()
#plt.show()

run_animation(x,True)