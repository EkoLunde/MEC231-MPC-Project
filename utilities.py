import numpy as np

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
            print('================ Problem is bounded ==================')
    return

def model_linearization(x, I_b):
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
