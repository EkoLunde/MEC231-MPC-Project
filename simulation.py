import numpy as np
import scipy.signal
import scipy.linalg
import polytope as pt
from utilities import *
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from model import *
import pyIGRF
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
    
def mpc(Q, R, x0, x_ref, I_b, N, M, xL, xU, uL, uU, Af, bf, Ts):
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
    b_mag_skew = np.zeros((nx-4, nx-4, N+1))

    feas = np.zeros((M, ), dtype=bool)
    xN = np.zeros((nx,1))

    for t in range(M):
        #IGRF
        lat = 40
        lon = 116
        alt = 300

        b_mag_skew = np.empty((0,3), float)
        b_mag_vec = np.empty((0,3), float)
        for k in range(N+1):    
            date = 2022+(t+k)*Ts/3.154e+7
            b_mag = pyIGRF.igrf_value(lat, lon, alt, date)[2:5]
            b_mag_vec = np.append(b_mag_vec, [[1e-9*b_mag[0], 1e-9*b_mag[1], 1e-9*b_mag[2]]], axis=0)
        if (t+N < M):
            [model, feas[t], x, u, J] = solve_cftoc(P, Q, R, N, xOpt[:, t], x_ref, xL, xU, uL, uU, bf, Af,b_mag_vec.T, I_b, Ts)
        else:
            [model, feas[t], x, u, J] = solve_cftoc(P, Q, R, (M-t), xOpt[:, t], x_ref, xL, xU, uL, uU, bf, Af,b_mag_vec.T, I_b, Ts)
        if not feas[t]:
            #xOpt = []
            #uOpt = []
            #predErr = []
            print("infeasable at time ", t)
            #break
        # Save open loop predictions
        #xPred[:, :, t] = x

        # Save closed loop trajectory
        # Note that the second column of x represents the optimal closed loop state
        xOpt[:, t+1] = x[:, 1]
        uOpt[:, t] = u[:, 0].reshape(nu, ) 

    return [model, feas, xOpt, uOpt]

I_b = (1/1e6)*np.array([[319, 0, 0],
               [0, 420, 0],
               [0, 0, 521]])
I_b_inv = np.linalg.inv(I_b)

omega_0 = np.array([0.0, 0.0, 0.0]).T #
q_0 = np.array([0.0, 0.0, 1.0, 0.0]).T
x0 = np.concatenate((q_0.T, omega_0.T))
x_ref = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
Ts = 0.1
N=25
M = 500   # Simulation steps
quat_err = 0.1
omega_err = 0.5

Q = np.array([[0, 0, 0, 0, 0, 0, 0],
              [0, 100, 0, 0, 0, 0, 0],
              [0, 0, 100, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0.01, 0, 0],
              [0, 0, 0, 0, 0, 0.01, 0],
              [0, 0, 0, 0, 0, 0, 0]])
R = np.eye(3) #10*np.array([1]).reshape(1,1)
P = Q
xL = np.array([-1.0, -1.0, -1.0, -1.0, -5.0, -5.0, -5.0]).T
xU = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0]).T
uL = np.array([-0.03, -0.03, -0.03]).T
uU = np.array([0.03, 0.03, 0.03]).T

Af = np.array([[1, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 1],
               [-1, 0, 0, 0, 0, 0, 0],
               [0, -1, 0, 0, 0, 0, 0],
               [0, 0, -1, 0, 0, 0, 0],
               [0, 0, 0, -1, 0, 0, 0],
               [0, 0, 0, 0, -1, 0, 0],
               [0, 0, 0, 0, 0, -1, 0],
               [0, 0, 0, 0, 0, 0, -1]])
bf = np.array([x_ref[0] + quat_err, x_ref[1] + quat_err, x_ref[2] + quat_err, x_ref[3] + quat_err, x_ref[4] + omega_err, x_ref[5] + omega_err, x_ref[6] + omega_err,
               -x_ref[0] + quat_err, -x_ref[1] + quat_err, -x_ref[2] + quat_err, -x_ref[3] + quat_err, -x_ref[4] + omega_err, -x_ref[5] + omega_err, -x_ref[6] + omega_err]).T

[model, feas, x, u] = mpc(Q, R, x0, x_ref, I_b, N, M, xL, xU, uL, uU, Af, bf, Ts)

#A_c, B_c, C_c = model_linearization(x0, I_b)
#D_c = np.array(np.zeros((3,1)))
#system = (A_c, B_c, C_c, D_c)
#A, B, C, D, dt = scipy.signal.cont2discrete(system, Ts)  
#[model, feas, x, u, J] = solve_cftoc(A, B, P, Q, R, N, x0, xL, xU, uL, uU, bf, Af)

plt.plot(x[0:4,:].T)
plt.ylabel('unit')
plt.xlabel('time[100ms]')
plt.legend((r'$\eta$',r'$q_1$',r'$q_2$',r'$q_3$'),)
plt.grid()
fig = plt.figure(figsize=(9, 6))
plt.plot(x[4:7,:].T)
plt.ylabel('rad/s')
plt.xlabel('time[100ms]')
plt.legend((r'$\omega_1$',r'$\omega_2$',r'$\omega_3$'))
plt.grid()
fig = plt.figure(figsize=(9, 6))
plt.plot(u.T)
plt.ylabel('Joule/Tesla')
plt.xlabel('time[100ms]')
plt.legend(("u1","u2","u3"))
plt.grid()
plt.show()

run_animation(x,False)