import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
import matplotlib.pyplot as plt
from utilities import *

def solve_cftoc(A, B, P, Q, R, N, x0, xL, xU, uL, uU, bf, Af, b_mag_vec, b_mag_skew, Ts, I_b):
    model = pyo.ConcreteModel()
    model.N = N
    model.nx = np.size(A, 0)
    model.nu = np.size(B, 1)

    # length of finite optimization problem:
    model.tIDX = pyo.Set( initialize= range(model.N+1), ordered=True )
    model.xIDX = pyo.Set( initialize= range(model.nx), ordered=True )
    model.vIDX = pyo.Set( initialize= range(model.nx-1), ordered=True )
    model.uIDX = pyo.Set( initialize= range(model.nu), ordered=True )

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
    model.u = pyo.Var(model.uIDX, model.tIDX)
    model.v = pyo.Var(model.vIDX, model.tIDX)
    #model.epsL = pyo.Var(model.xIDX, model.tIDX)
    #model.epsU = pyo.Var(model.xIDX, model.tIDX, domain=pyo.NonNegativeReals) # Slack variable is introduced 
    #Objective rule Euler
    def objective_rule_euler(model):
        costX = 0.0
        costU = 0.0
        costTerminal = 0.0
        #CostSoftPenaltyEpsL = 0.0
        #CostSoftPenaltyEpsU = 0.0
        for t in model.tIDX:
            v = euler_from_quaternion(model.x[0,t], model.x[1,t], model.x[2,t], model.x[3,t]) #translating to euler
            v = v.append(model.x[4,t])
            v = v.append(model.x[5,t])
            v = v.append(model.x[6,t])
            for i in model.vIDX:
                for j in model.vIDX:
                    if t < model.N:
                        model.v[i,t]=v[i]
                        costX +=  (model.v[i, t] - model.bf[i]) * model.Q[i, j] * (model.v[j, t]-model.bf[i]) #Tror kanskje dette må endres pga quaternions
                        #CostSoftPenaltyEpsL += model.epsL[i,t] * model.epsL[j,t]
                        #CostSoftPenaltyEpsU += model.epsU[i,t] * model.epsU[j,t] # penalty on the slack variable
        for t in model.tIDX:
            for i in model.uIDX:
                for j in model.uIDX:
                    if t < model.N:
                        costU += model.u[i, t] * model.R[i, j] * model.u[j, t]
        for i in model.vIDX:
            for j in model.vIDX:
                costTerminal += (model.v[i, model.N]- model.bf[i]) * model.P[i, j] * (model.v[j, model.N]-model.bf[i])
        return costX + costU + costTerminal #+ CostSoftPenaltyEpsU + CostSoftPenaltyEpsL

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
                        costX += (model.x[i, t] - model.bf[i]) * model.Q[i, j] * (model.x[j, t]-model.bf[i]) #Tror kanskje dette må endres pga quaternions
                        #CostSoftPenaltyEpsL += model.epsL[i,t] * model.epsL[j,t]
                        #CostSoftPenaltyEpsU += model.epsU[i,t] * model.epsU[j,t] # penalty on the slack variable
        for t in model.tIDX:
            for i in model.uIDX:
                for j in model.uIDX:
                    if t < model.N:
                        costU += model.u[i, t] * model.R[i, j] * model.u[j, t]
        for i in model.xIDX:
            for j in model.xIDX:
                costTerminal += (model.x[i, model.N]- model.bf[i]) * model.P[i, j] * (model.x[j, model.N]-model.bf[i])
        return costX + costU + costTerminal #+ CostSoftPenaltyEpsU + CostSoftPenaltyEpsL
    model.cost = pyo.Objective(rule = objective_rule_euler, sense = pyo.minimize)
    
    # nonlinear model
    def cubesat_model(model,i,t): #Denne er dobbelsjekket og virker riktig
        # quaternions
        if (i == 0):
            return model.x[i, t+1] - (model.x[i, t] + (Ts/2)*(model.x[1, t]*model.x[6, t] - model.x[2, t]*model.x[5, t] + model.x[3, t]*model.x[4, t])) == 0.0 if t < model.N else pyo.Constraint.Skip
        elif (i == 1):
            return model.x[i, t+1] - (model.x[i, t] + (Ts/2)*(-model.x[0, t]*model.x[6, t] + model.x[2, t]*model.x[4, t] + model.x[3, t]*model.x[5, t])) == 0.0 if t < model.N else pyo.Constraint.Skip
        elif (i == 2):
            return model.x[i, t+1] - (model.x[i, t] + (Ts/2)*(model.x[0, t]*model.x[5, t] - model.x[1, t]*model.x[4, t] + model.x[3, t]*model.x[6, t])) == 0.0 if t < model.N else pyo.Constraint.Skip
        elif (i == 3):
            return model.x[i, t+1] - (model.x[i, t] + (Ts/2)*(-model.x[0, t]*model.x[4, t] - model.x[1, t]*model.x[5, t] - model.x[2, t]*model.x[6, t])) == 0.0 if t < model.N else pyo.Constraint.Skip
        elif (i == 4):
            return model.x[i, t+1] - (model.x[i,t] + Ts*((1/I_b[i-4,i-4])*(model.u[i-4,t]+(I_b[2,2] - I_b[1,1])*model.x[5,t]*model.x[6,t]))) == 0 if t < model.N else pyo.Constraint.Skip
        elif (i == 5):
            return model.x[i, t+1] - (model.x[i,t] + Ts*((1/I_b[i-4,i-4])*(model.u[i-4,t]+(I_b[0,0] - I_b[2,2])*model.x[4,t]*model.x[6,t]))) == 0 if t < model.N else pyo.Constraint.Skip
        elif (i == 6):
            return model.x[i, t+1] - (model.x[i,t] + Ts*((1/I_b[i-4,i-4])*(model.u[i-4,t]+(I_b[1,1] - I_b[0,0])*model.x[4,t]*model.x[5,t]))) == 0 if t < model.N else pyo.Constraint.Skip
    model.equality_constraints = pyo.Constraint(model.xIDX, model.tIDX, rule=cubesat_model)

    #def euler_angle_model(model,i,t): #Denne er dobbelsjekket og virker riktig     
    #    [roll,pitch,yaw] = euler_from_quaternion(model.x[0,t], model.x[1,t], model.x[2,t], model.x[3,t])
    #    # euler angles
    #    if (i == 0):
    #        return model.v[i, t] - roll
    #    elif (i == 1):
    #        return model.v[i, t] - pitch == 0.0 if t < model.N else pyo.Constraint.Skip
    #    elif (i == 2):
    #        return model.v[i, t] - yaw == 0.0 if t < model.N else pyo.Constraint.Skip
    #    else:
    #        return model.v[i, t] - model.x[i+1,t] == 0 if t < model.N else pyo.Constraint.Skip
    #model.euler_angle_const = pyo.Constraint(model.vIDX, model.tIDX, rule=euler_angle_model)
    
    # Orthogonality constraint
    #model.orthogonality_mag_const1 = pyo.Constraint(model.tIDX, rule=lambda model, t: sum(model.u[j, t]*b_mag_vec[t,j] for j in range(3)) == 0 if t <= N-1 else pyo.Constraint.Skip)
    #model.orthogonality_mag_const2 = pyo.Constraint(model.N, model.tIDX, rule=lambda model, i, t: sum(model.u[k,t]*sum(model.u[j,t]*b_mag_skew[3*i, j] for j in range(3)) for k in range(3)) == 0 if t <= N-1 else pyo.Constraint.Skip)

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

    #model.max_const1 = pyo.Constraint(model.xIDX, model.tIDX, rule=lambda model, i, t: model.x[i, t] <= xU[i] if t <= N-1 else pyo.Constraint.Skip)
    #model.min_const1 = pyo.Constraint(model.xIDX, model.tIDX, rule=lambda model, i, t: xL[i] <=  model.x[i, t] if t <= N-1 else pyo.Constraint.Skip)
    model.max_const2 = pyo.Constraint(model.uIDX, model.tIDX, rule=lambda model, i, t: model.u[i, t] <= uU[i] if t <= N-1 else pyo.Constraint.Skip)
    model.min_const2 = pyo.Constraint(model.uIDX, model.tIDX, rule=lambda model, i, t: uL[i] <=  model.u[i, t] if t <= N-1 else pyo.Constraint.Skip)
    #model.epsU = pyo.Constraint(model.xIDX, model.tIDX, rule=lambda model, i, t: model.epsL[i,t] >= 0)

    model.unit_quat_const = pyo.Constraint(model.tIDX, rule=lambda model, t: (model.x[0, t]**2 + model.x[1, t]**2 + model.x[2, t]**2 + model.x[3, t]**2) == 1 if t <= N-1 else pyo.Constraint.Skip)

    #model.final_const1 = pyo.Constraint(expr = sum(model.Af[0, j]*model.x[j, model.N] for j in model.xIDX) == model.bf[0])
    #model.final_const2 = pyo.Constraint(expr = sum(model.Af[1, j]*model.x[j, model.N] for j in model.xIDX) == model.bf[1])
    #model.final_const3 = pyo.Constraint(expr = sum(model.Af[2, j]*model.x[j, model.N] for j in model.xIDX) == model.bf[2])
    #model.final_const4 = pyo.Constraint(expr = sum(model.Af[3, j]*model.x[j, model.N] for j in model.xIDX) == model.bf[3])
    #model.final_const5 = pyo.Constraint(expr = sum(model.Af[4, j]*model.x[j, model.N] for j in model.xIDX) == model.bf[4])
    #model.final_const6 = pyo.Constraint(expr = sum(model.Af[5, j]*model.x[j, model.N] for j in model.xIDX) == model.bf[5])
    #model.final_const7 = pyo.Constraint(expr = sum(model.Af[6, j]*model.x[j, model.N] for j in model.xIDX) == model.bf[6])

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
