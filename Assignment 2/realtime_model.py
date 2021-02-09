# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:55:46 2021

@author: ga47jes
"""

import pyomo.core as pyen
from pyomo.opt import SolverFactory
import numpy as np


def realtime_opt(g_data,Fmax,Bmax,Scen,Dem,res_da):
    
    # Create model
    model = pyen.ConcreteModel()
    
    # Sets
    model.d = pyen.Set(ordered=True, initialize=np.arange(0,len(Dem)))
    model.g = pyen.Set(ordered=True, initialize=np.arange(0,len(g_data)))
    model.n = pyen.Set(ordered=True, initialize=np.arange(0,len(Bmax)))
    model.m = pyen.Set(ordered=True, initialize=np.arange(0,len(Bmax)))
    model.w = pyen.Set(ordered=True, initialize=np.arange(0,1))
    model.s = pyen.Set(ordered=True, initialize=np.arange(0,1)) #Only one scenario
    
    # Parameters
    model.pg_max = pyen.Param(model.g, initialize=g_data['Pmax'].to_dict())
    model.cg = pyen.Param(model.g, initialize=g_data['C'].to_dict())
    model.r_max = pyen.Param(model.g, initialize=g_data['Rmax'].to_dict())
    model.f_max = pyen.Param(model.n, model.n, initialize=Fmax.stack().to_dict())
    model.b_max = pyen.Param(model.n, model.n, initialize=Bmax.stack().to_dict())
    model.w_s = pyen.Param(model.s, initialize=Scen)
    model.w_max = pyen.Param(initialize=70)
    model.dem = pyen.Param(model.d, initialize=Dem['Load'].to_dict())
    model.penalty = pyen.Param(model.d, initialize=Dem['Penalty'].to_dict())
    model.Pg_da = pyen.Param(model.g, initialize=res_da['Pg_da'].to_dict())
    model.Pw_da = pyen.Param(model.w, initialize=res_da['Pw_da'][0])
    
    # Variables
    model.cost = pyen.Var(model.s, within=pyen.Reals)
    model.theta_da = pyen.Var(model.n, within=pyen.Reals)
    model.theta_rt = pyen.Var(model.n, model.s, within=pyen.Reals)
    model.f_DA = pyen.Var(model.n, model.n, within=pyen.Reals, initialize=0)
    model.f_RT = pyen.Var(model.n, model.n, model.s, within=pyen.Reals, initialize=0)
    model.Pg_rt = pyen.Var(model.g, model.s, within=pyen.Reals)
    
    # Positive variables
    model.L_shed = pyen.Var(model.d, model.s, within=pyen.NonNegativeReals)
    model.P_spill = pyen.Var(model.w, model.s, within=pyen.NonNegativeReals)
    
    # DA Constraints
    def nodal_da(model, n, w):
        return sum(model.Pg_da[g] for g in model.g) + sum(model.Pw_da[w] for w in model.w) - \
               sum(model.f_DA[n,m] for n in model.n for m in model.n) - model.dem[0] == 0
    model.nodal_DA_bal = pyen.Constraint(model.n, model.w, rule=nodal_da)    
   
    def flow_DA_calc(model, n, m):
        return model.f_DA[n,m] == model.b_max[n,m]*(model.theta_da[n] - model.theta_da[m])
    model.flow_DA_calc = pyen.Constraint(model.n, model.m, rule=flow_DA_calc)
    
    def flow_DA_bound(model, n, m):
        return (-model.f_max[n,m], model.f_DA[n,m], model.f_max[n,m])
    model.flow_DA_bound = pyen.Constraint(model.n, model.m, rule=flow_DA_bound)
    
    
    # RT Constraints
    def nodel_rt(model, d, g, w, s):    
        return sum(model.Pg_rt[g,s] for g in model.g) +   \
               sum(model.w_s[s] - model.Pw_da[w] - model.P_spill[w,s] for w in model.w) + \
               sum(model.L_shed[d,s] for d in model.d) - \
               sum(model.f_RT[n,m,s] - model.f_DA[n,m] for n in model.n for m in model.m) == 0
    model.nodel_rt = pyen.Constraint(model.d, model.g, model.w, model.s, rule=nodel_rt)    
    
    def power_adjust(model, g, s):
        return (-model.r_max[g], model.Pg_rt[g,s], model.r_max[g])
    model.rw_power_adjust = pyen.Constraint(model.g, model.s, rule=power_adjust)
    
    def gen_limits_rt(model, g, s):
        return (0, model.Pg_da[g] + model.Pg_rt[g,s], model.pg_max[g])
    model.gen_limits_rt = pyen.Constraint(model.g, model.s, rule=gen_limits_rt)
    
    def load_shed_rt(model, d, s):
        return model.L_shed[d,s] <= model.dem[d]
    model.load_shed_rt = pyen.Constraint(model.d, model.s, rule=load_shed_rt)
    
    def spill_rt(model, w, s):
        return model.P_spill[w,s] <= model.w_s[s]
    model.spill_rt = pyen.Constraint(model.w, model.s, rule=spill_rt)
    
    def flow_RT_calc(model, n, m, s):
        return model.f_RT[n,m,s] == model.b_max[n,m]*(model.theta_rt[n,s] - model.theta_rt[m,s])
    model.flow_RT_calc = pyen.Constraint(model.n, model.m, model.s, rule=flow_RT_calc)
    
    def flow_RT_bound(model, n, m, s):
        return (-model.f_max[n,m], model.f_RT[n,m,s], model.f_max[n,m])
    model.flow_RT_bound = pyen.Constraint(model.n, model.m, model.s, rule=flow_RT_bound)
    
    
    # Objective
    def obj_rule(model):
        cost_rt = 0.0
        cost_net = 0.0
        for s in model.s:
            cost_rt += sum(model.cg[g]*model.Pg_rt[g,s] for g in model.g) + \
                       sum(model.penalty[d]*model.L_shed[d,s] for d in model.d)
        cost_net = cost_rt
        return cost_net
    model.obj = pyen.Objective(sense=pyen.minimize, rule=obj_rule, doc='Sum costs by cost type')
    
    
    # Optimize
    opt = SolverFactory("glpk")
    results = opt.solve(model, tee=False)
    # results.write() 
    # if results.solver.status:
    #     model.pprint()
     
    return model