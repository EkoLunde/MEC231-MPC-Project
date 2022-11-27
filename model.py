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

def check_solver_status(model, results):
    from pyomo.opt import SolverStatus, TerminationCondition
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        print('========================================================================================')
        print('================ Problem is feasible and the optimal solution is found ==================')
        print('========================================================================================')
    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        print('========================================================')
        print('================ Problem is infeasible ==================')
        print('========================================================')
        if (results.solver.termination_condition == TerminationCondition.unbounded):
            print('================ Problem is unbounded ==================')
        else:
            print('================ Problem is bounded ==================')

    else:
        if (results.solver.termination_condition == TerminationCondition.unbounded):
            print('================ Problem is unbounded ==================')
        else:
            print('================ Problem is unbounded ==================')

    return

def model_linearization(x):
    A = np.array([[0, x[6]/2, -x[5]/2, x[4]/2, x[3]/2, -x[2]/2, x[1]/2],
               [-x[6], 0, x[6]/2, x[5]/2, x[2]/2, x[3]/2, -x[0]/2],
               [x[5]/2, -x[4]/2, 0, x[6]/2, -x[1]/2, x[0]/2, x[3]/2],
               [-x[4]/2, -x[5]/2, x[6]/2, 0, -x[0]/2, -x[1]/2, x[2]/2],
               [0, 0, 0, 0, 0, (-1)*(I_b[2][2]-I_b[1][1])*x[6]/I_b[0][0], (-1)*(I_b[2][2]-I_b[1][1])*x[5]/I_b[0][0]],
               [0, 0, 0, 0, (-1)*(I_b[0][0]-I_b[2][2])*x[6]/I_b[1][1], 0, (-1)*(I_b[0][0]-I_b[2][2])*x[4]/I_b[1][1]],
               [0, 0, 0, 0, (-1)*(I_b[1][1]-I_b[0][0])*x[5]/I_b[2][2], (-1)*(I_b[1][1]-I_b[0][0])*x[4]/I_b[2][2], 0]])
    B = np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1/I_b[0][0], 0, 0],
                [0, 1/I_b[1][1], 0],
                [0, 0, 1/I_b[2][2]]])
    C = 0
    
    return A, B, C

def solve_cftoc(A, B, P, Q, R, N, x0, xL, xU, uL, uU, bf, Af):

    model = pyo.ConcreteModel()
    model.N = N
    model.nx = np.size(A, 0)
    model.nu = np.size(B, 1)
    model.nf = np.size(Af, 0)

    # length of finite optimization problem:
    model.tIDX = pyo.Set( initialize= range(model.N+1), ordered=True )
    model.xIDX = pyo.Set( initialize= range(model.nx), ordered=True )
    model.uIDX = pyo.Set( initialize= range(model.nu), ordered=True )

    model.nfIDX = pyo.Set( initialize= range(model.nf), ordered=True )

    # these are 2d arrays:
    model.A = A
    model.B = B
    model.Q = Q
    model.P = P
    model.R = R
    model.Af = Af
    model.bf = bf

    # Create state and input variables trajectory:
    model.x = pyo.Var(model.xIDX, model.tIDX)
    model.u = pyo.Var(model.uIDX, model.tIDX, bounds=(uL,uU))

    #Objective:
    def objective_rule(model):
        costX = 0.0
        costU = 0.0
        costTerminal = 0.0
        for t in model.tIDX:
            for i in model.xIDX:
                for j in model.xIDX:
                    if t < model.N:
                        costX += model.x[i, t] * model.Q[i, j] * model.x[j, t]
        for t in model.tIDX:
            for i in model.uIDX:
                for j in model.uIDX:
                    if t < model.N:
                        costU += model.u[i, t] * model.R[i, j] * model.u[j, t]
        for i in model.xIDX:
            for j in model.xIDX:
                costTerminal += model.x[i, model.N] * model.P[i, j] * model.x[j, model.N]
        return costX + costU + costTerminal

    model.cost = pyo.Objective(rule = objective_rule, sense = pyo.minimize)

    # Constraints:
    def equality_const_rule(model, i, t):
        return model.x[i, t+1] - (sum(model.A[i, j] * model.x[j, t] for j in model.xIDX)
                               +  sum(model.B[i, j] * model.u[j, t] for j in model.uIDX) ) == 0.0 if t < model.N else pyo.Constraint.Skip


    model.equality_constraints = pyo.Constraint(model.xIDX, model.tIDX, rule=equality_const_rule)


    model.init_const1 = pyo.Constraint(expr = model.x[0, 0] == x0[0])
    model.init_const2 = pyo.Constraint(expr = model.x[1, 0] == x0[1])
    model.init_const3 = pyo.Constraint(expr = model.x[2, 0] == x0[2])
    model.init_const4 = pyo.Constraint(expr = model.x[3, 0] == x0[3])
    model.init_const5 = pyo.Constraint(expr = model.x[4, 0] == x0[4])
    model.init_const6 = pyo.Constraint(expr = model.x[5, 0] == x0[5])
    model.init_const7 = pyo.Constraint(expr = model.x[6, 0] == x0[6])

    #model.const1 = pyo.Constraint(expr = (pyo.sqrt(model.x[0, 0]*model.x[0, 0] + model.x[1, 0]*model.x[1, 0] + model.x[2, 0]*model.x[2, 0] + model.x[3, 0]*model.x[3, 0]) == 1)) #unit quaternion

    def final_const_rule(model, i):
        return sum(model.Af[i, j] * model.x[j, model.N] for j in model.xIDX) <= model.bf[i]

    model.final_const = pyo.Constraint(model.nfIDX, rule=final_const_rule)

    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)

    check_solver_status(model, results)

    if str(results.solver.termination_condition) == "optimal":
        feas = True
    else:
        feas = False

    xOpt = np.asarray([[model.x[i,t]() for i in model.xIDX] for t in model.tIDX]).T
    uOpt = np.asarray([model.u[:,t]() for t in model.tIDX]).T

    JOpt = model.cost()

    return [model, feas, xOpt, uOpt, JOpt]


def model_sim():


    Xf = O_inf
    Af = Xf.A
    bf = Xf.b

    Q = np.eye(7)
    R = np.eye(3) #10*np.array([1]).reshape(1,1)
    P = Q
    N = 3
    xL = -10.0
    xU = 10.0
    uL = -1.0
    uU = 1.0
    x0 = np.array([0.5, -0.5, 0.5, 0,0,0,1]).T

    A, B, C = model_linearization(x0)

    nx = np.size(A, 0)
    nu = np.size(B, 1)

    M = 25   # Simulation steps
    xOpt = np.zeros((nx, M+1))
    uOpt = np.zeros((nu, M))
    xOpt[:, 0] = x0.reshape(nx, )

    xPred = np.zeros((nx, N+1, M))
    predErr = np.zeros((nx, M-N+1))

    feas = np.zeros((M, ), dtype=bool)
    xN = np.zeros((nx,1))

    fig = plt.figure(figsize=(9, 6))
    for t in range(M):
        [model, feas[t], x, u, J] = solve_cftoc(A, B, P, Q, R, N, xOpt[:, t], xL, xU, uL, uU, bf, Af)

        if not feas[t]:
            xOpt = []
            uOpt = []
            predErr = []
            break
        # Save open loop predictions
        xPred[:, :, t] = x

        # Save closed loop trajectory
        # Note that the second column of x represents the optimal closed loop state
        xOpt[:, t+1] = x[:, 1]
        uOpt[:, t] = u[:, 0].reshape(nu, )
        A, B, C = model_linearization(x[:, 1])

    return


Af = np.eye(7)
bf = np.array([1, 0, 0, 0, 0, 0, 0]).T

Q = np.eye(7)
R = np.eye(3) #10*np.array([1]).reshape(1,1)
P = Q
N = 30
xL = -10.0
xU = 10.0
uL = -1.0
uU = 1.0
x0 = np.array([0,0,0,1, 0.5, -0.5, 0.5]).T

A, B, C = model_linearization(x0)

[model, feas, x, u, J] = solve_cftoc(A, B, P, Q, R, N, x0, xL, xU, uL, uU, bf, Af)
