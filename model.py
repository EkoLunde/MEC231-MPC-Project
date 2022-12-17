import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
import matplotlib.pyplot as plt
from utilities import *

def solve_cftoc(P, Q, R, N, x0, x_ref, xL, xU, uL, uU, bf, Af, b_mag, I_b, Ts):
    model = pyo.ConcreteModel()
    model.N = N
    model.nx = np.size(Q, 0)
    model.nu = np.size(R, 1)
    model.nf = np.size(Af, 0)

    # length of finite optimization problem:
    model.tIDX = pyo.Set( initialize= range(model.N+1), ordered=True )
    model.xIDX = pyo.Set( initialize= range(model.nx), ordered=True )
    model.uIDX = pyo.Set( initialize= range(model.nu), ordered=True )

    model.nfIDX = pyo.Set( initialize= range(model.nf), ordered=True )

    # these are 2d arrays:
    model.Q = Q
    model.P = P
    model.R = R
    model.Af = Af
    model.bf = bf
    model.x_ref = x_ref
    model.b_mag = b_mag

    # Create state and input variables trajectory:
    model.x = pyo.Var(model.xIDX, model.tIDX)
    model.u = pyo.Var(model.uIDX, model.tIDX)
    #model.epsL = pyo.Var(model.xIDX, model.tIDX)
    #model.epsU = pyo.Var(model.xIDX, model.tIDX, domain=pyo.NonNegativeReals) # Slack variable is introduced 

    #Objective:
    def objective_rule(model):
        costX = 0.0
        costU = 0.0
        costTerminal = 0.0
        #CostSoftPenaltyEpsL = 0.0
        #CostSoftPenaltyEpsU = 0.0
        for t in model.tIDX:
            for i in model.xIDX:
                for j in model.xIDX:
                    if t < model.N:
                        costX += (model.x[i, t] - model.x_ref[i]) * model.Q[i, j] * (model.x[j, t] - model.x_ref[j]) #Tror kanskje dette må endres pga quaternions
                        #CostSoftPenaltyEpsL += model.epsL[i,t] * model.epsL[j,t]
                        #CostSoftPenaltyEpsU += model.epsU[i,t] * model.epsU[j,t] # penalty on the slack variable
        for t in model.tIDX:
            for i in model.uIDX:
                for j in model.uIDX:
                    if t < model.N:
                        costU += model.u[i, t] * model.R[i, j] * model.u[j, t]
        for i in model.xIDX:
            for j in model.xIDX:
                costTerminal += (model.x[i, model.N] - model.x_ref[i]) * model.P[i, j] * (model.x[j, model.N] - model.x_ref[j])
        return costX + costU + costTerminal #+ CostSoftPenaltyEpsU + CostSoftPenaltyEpsL

    model.cost = pyo.Objective(rule = objective_rule, sense = pyo.minimize)
    
    # nonlinear model
    def cubesat_model(model,i,t):
        # quaternions
        if (i == 0):
            return model.x[i, t+1] - (model.x[i, t] + (Ts/2)*(model.x[1, t]*model.x[6, t] - model.x[2, t]*model.x[5, t] + model.x[3, t]*model.x[4, t])) == 0.0 if t < model.N else pyo.Constraint.Skip
        elif (i == 1):
            return model.x[i, t+1] - (model.x[i, t] + (Ts/2)*(-model.x[0, t]*model.x[6, t] + model.x[2, t]*model.x[4, t] + model.x[3, t]*model.x[5, t])) == 0.0 if t < model.N else pyo.Constraint.Skip
        elif (i == 2):
            return model.x[i, t+1] - (model.x[i, t] + (Ts/2)*(model.x[0, t]*model.x[5, t] - model.x[1, t]*model.x[4, t] + model.x[3, t]*model.x[6, t])) == 0.0 if t < model.N else pyo.Constraint.Skip
        elif (i == 3):
            return model.x[i, t+1] - (model.x[i, t] + (Ts/2)*(-model.x[0, t]*model.x[4, t] - model.x[1, t]*model.x[5, t] - model.x[2, t]*model.x[6, t])) == 0.0 if t < model.N else pyo.Constraint.Skip
        # angular velocity
        elif (i == 4):
            return model.x[i, t+1] - (model.x[i, t] + (Ts/I_b[0, 0])*(model.b_mag[2,t]*model.u[1, t] - model.b_mag[1,t]*model.u[2, t] + (I_b[2, 2] - I_b[1, 1])*model.x[5, t]*model.x[6, t])) == 0.0 if t < model.N else pyo.Constraint.Skip
        elif (i == 5):
            return model.x[i, t+1] - (model.x[i, t] + (Ts/I_b[1, 1])*(-model.b_mag[2,t]*model.u[0, t] + model.b_mag[0,t]*model.u[2, t] + (I_b[0, 0] - I_b[2, 2])*model.x[4, t]*model.x[6, t])) == 0.0 if t < model.N else pyo.Constraint.Skip
        elif (i == 6):
            return model.x[i, t+1] - (model.x[i, t] + (Ts/I_b[2, 2])*(model.b_mag[1,t]*model.u[0, t] - model.b_mag[0,t]*model.u[1, t] + (I_b[1, 1] - I_b[0, 0])*model.x[4, t]*model.x[5, t])) == 0.0 if t < model.N else pyo.Constraint.Skip
    model.equality_constraints = pyo.Constraint(model.xIDX, model.tIDX, rule=cubesat_model)

    # Orthogonality constraint
    model.orthogonality_mag_const1 = pyo.Constraint(model.tIDX, rule=lambda model, t: sum(model.u[j, t]*b_mag[j,t] for j in range(3)) == 0 if t <= N else pyo.Constraint.Skip)
    model.orthogonality_mag_const2 = pyo.Constraint(model.tIDX, rule=lambda model, t: (model.b_mag[2,t]*model.u[1, t] - model.b_mag[1,t]*model.u[2, t])*model.u[0,t] 
                                                                                    + (-model.b_mag[2,t]*model.u[0, t] + model.b_mag[0,t]*model.u[2, t])*model.u[1,t] 
                                                                                    + (model.b_mag[1,t]*model.u[0, t] - model.b_mag[0,t]*model.u[1, t])*model.u[2, t] == 0 if t <= N else pyo.Constraint.Skip)

    # Constraints:

    ##Linearized constraints
    #def equality_const_rule(model, i, t):
    #    return model.x[i, t+1] - (sum(model.A[i, j] * model.x[j, t] for j in model.xIDX)
    #                           +  sum(model.B[i, j] * model.u[j, t] for j in model.uIDX) ) == 0.0 if t < model.N else pyo.Constraint.Skip
    #model.equality_constraints = pyo.Constraint(model.xIDX, model.tIDX, rule=equality_const_rule)

    model.init_const1 = pyo.Constraint(expr = model.x[0, 0] == x0[0])
    model.init_const2 = pyo.Constraint(expr = model.x[1, 0] == x0[1])
    model.init_const3 = pyo.Constraint(expr = model.x[2, 0] == x0[2])
    model.init_const4 = pyo.Constraint(expr = model.x[3, 0] == x0[3])
    model.init_const5 = pyo.Constraint(expr = model.x[4, 0] == x0[4])
    model.init_const6 = pyo.Constraint(expr = model.x[5, 0] == x0[5])
    model.init_const7 = pyo.Constraint(expr = model.x[6, 0] == x0[6])

    model.min_const = pyo.Constraint(model.uIDX, model.tIDX, rule=lambda model, i, t: uL[i] <=  model.u[i, t] if t <= N else pyo.Constraint.Skip)
    model.max_const = pyo.Constraint(model.uIDX, model.tIDX, rule=lambda model, i, t: model.u[i, t] <= uU[i] if t <= N else pyo.Constraint.Skip)

    model.unit_quat_const = pyo.Constraint(model.tIDX, rule=lambda model, t: (model.x[0, t]**2 + model.x[1, t]**2 + model.x[2, t]**2 + model.x[3, t]**2) == 1 if t <= N-1 else pyo.Constraint.Skip)

    #model.final_const0 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[0, 0]*model.x[0, t] <= model.bf[0] if t >= N else pyo.Constraint.Skip)
    model.final_const1 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[1, 1]*model.x[1, t] <= model.bf[1] if t >= N else pyo.Constraint.Skip)
    model.final_const2 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[2, 2]*model.x[2, t] <= model.bf[2] if t >= N else pyo.Constraint.Skip)
    #model.final_const3 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[3, 3]*model.x[3, t] <= model.bf[3] if t >= N else pyo.Constraint.Skip)
    #model.final_const4 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[4, 4]*model.x[4, t] <= model.bf[4] if t >= N else pyo.Constraint.Skip)
    #model.final_const5 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[5, 5]*model.x[5, t] <= model.bf[5] if t >= N else pyo.Constraint.Skip)
    #model.final_const6 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[6, 6]*model.x[6, t] == model.bf[6] if t >= N else pyo.Constraint.Skip)
    #model.final_const7 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[7, 0]*model.x[0, t] <= model.bf[7] if t >= N else pyo.Constraint.Skip)
    model.final_const8 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[8, 1]*model.x[1, t] <= model.bf[8] if t >= N else pyo.Constraint.Skip)
    model.final_const9 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[9, 2]*model.x[2, t] <= model.bf[9] if t >= N else pyo.Constraint.Skip)
    #model.final_const10 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[10, 3]*model.x[3, t] <= model.bf[10] if t >= N else pyo.Constraint.Skip)
    #model.final_const11 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[11, 4]*model.x[4, t] <= model.bf[11] if t >= N else pyo.Constraint.Skip)
    #model.final_const12 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[12, 5]*model.x[5, t] <= model.bf[12] if t >= N else pyo.Constraint.Skip)
    #model.final_const13 = pyo.Constraint(model.tIDX, rule=lambda model, t: model.Af[13, 6]*model.x[6, t] <= model.bf[13] if t >= N else pyo.Constraint.Skip)

    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)
    check_solver_status(model, results)
    if str(results.solver.termination_condition) == "optimal":
        feas = True
    else:
        feas = False

    #results = pyo.SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt')#, tee=True)
    #model.display()
    #model.pprint()
    #feas=True

    xOpt = np.asarray([[model.x[i,t]() for i in model.xIDX] for t in model.tIDX]).T
    uOpt = np.asarray([model.u[:,t]() for t in model.tIDX]).T

    JOpt = model.cost()

    return [model, feas, xOpt, uOpt, JOpt]
