# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:41:54 2021

@author: Babu Kumaran Nalini (ga47jes)
"""

import pyomo.core as pyen
from pyomo.opt import SolverFactory
import numpy as np

def bilevel_opt(g_data,Fmax,Bmax,Dem,leader):
    
    # Create model
    model = pyen.ConcreteModel()
    
    # Sets
    g_leader = g_data.iloc[leader]
    g_leader = g_leader.reset_index(drop=True)
    g_follower = g_data.drop(leader)
    g_follower = g_follower.reset_index(drop=True)
    
    # Sets
    model.I = pyen.Set(ordered=True, initialize=np.arange(0,len(g_leader)))
    model.neg_I = pyen.Set(ordered=True, initialize=np.arange(0,len(g_follower)))
    model.K = pyen.Set(ordered=True, initialize=np.arange(0,len(Dem)))  
 
    # Parameters
    model.PGmax_leader = pyen.Param(model.I, initialize=g_leader['Pmax'].to_dict())         # Leader
    model.Cost_leader = pyen.Param(model.I, initialize=g_leader['C'].to_dict()) 
    model.PGmax_rival = pyen.Param(model.neg_I, initialize=g_follower['Pmax'].to_dict())    # Rival
    model.Cost_rival = pyen.Param(model.neg_I, initialize=g_follower['C'].to_dict())
    model.PDmax = pyen.Param(model.K, initialize=Dem['Load'].to_dict())                     # Demand
    model.Cost_demand = pyen.Param(model.K, initialize=Dem['Cost'].to_dict())
    model.M = 10000                                                                         # BigM
    
    # Variables
    model.lambda_ = pyen.Var(initialize=0, within=pyen.Reals)
    model.offer_i = pyen.Var(model.I, initialize=0, within=pyen.NonNegativeReals)
    
    # Variables I
    model.p_i = pyen.Var(model.I, initialize=0, within=pyen.NonNegativeReals)
    model.p_neg_i = pyen.Var(model.neg_I, initialize=0, within=pyen.NonNegativeReals)
    model.mu_dashed_i = pyen.Var(model.I, initialize=0, within=pyen.NonNegativeReals)
    model.mu_bar_i = pyen.Var(model.I, initialize=0, within=pyen.NonNegativeReals)
    model.mu_dashed_neg_i = pyen.Var(model.neg_I, initialize=0, within=pyen.NonNegativeReals)
    model.mu_bar_neg_i = pyen.Var(model.neg_I, initialize=0, within=pyen.NonNegativeReals)
    model.phi_bar_i = pyen.Var(model.I, initialize=0, within=pyen.Binary)
    model.phi_dashed_i = pyen.Var(model.I, initialize=0, within=pyen.Binary)
    model.phi_bar_neg_i = pyen.Var(model.neg_I, initialize=0, within=pyen.Binary)
    model.phi_dashed_neg_i = pyen.Var(model.neg_I, initialize=0, within=pyen.Binary)
    
    # Variables K
    model.dem_k = pyen.Var(model.K, initialize=0, within=pyen.NonNegativeReals)
    model.mu_dashed_k = pyen.Var(model.K, initialize=0, within=pyen.NonNegativeReals)
    model.mu_bar_k = pyen.Var(model.K, initialize=0, within=pyen.NonNegativeReals)
    model.phi_bar_k = pyen.Var(model.K, initialize=0, within=pyen.Binary)
    model.phi_dash_k = pyen.Var(model.K, initialize=0, within=pyen.Binary)
    
    # Constraints
    def bid_constraint(model, k):
        return (-model.Cost_demand[k] + model.mu_bar_k[k] - model.mu_dashed_k[k]
                + model.lambda_ == 0)
    model.bid_const = pyen.Constraint(model.K, rule=bid_constraint)
    
    def offer_constraint(model, i):
        return (model.offer_i[i] + model.mu_bar_i[i] - model.mu_dashed_i[i]
                - model.lambda_ == 0)
    model.offer_const = pyen.Constraint(model.I, rule=offer_constraint)
    
    def rival_offer_constraint(model, neg_i):
        return (model.Cost_rival[neg_i] + model.mu_bar_neg_i[neg_i]
                - model.mu_dashed_neg_i[neg_i] - model.lambda_ == 0)
    model.rival_offer_const = pyen.Constraint(model.neg_I, rule=rival_offer_constraint)
    
    def power_balance_constraint(model):
        return (sum(model.dem_k[k] for k in model.K)
                - sum(model.p_i[i] for i in model.I)
                - sum(model.p_neg_i[neg_i] for neg_i in model.neg_I) == 0)
    model.power_balance_const = pyen.Constraint(rule=power_balance_constraint)
       
    def d_k_upper_M(model, k):
        return model.PDmax[k] - model.dem_k[k] <= model.phi_bar_k[k]*model.M
    model.d_k_upper_M = pyen.Constraint(model.K, rule=d_k_upper_M)
    
    def p_i_upper_M(model, i):
        return model.PGmax_leader[i] - model.p_i[i] <= model.phi_bar_i[i]*model.M
    model.p_i_upper_M = pyen.Constraint(model.I, rule=p_i_upper_M)
    
    def p_neg_i_upper_M(model, neg_i):
        return model.PGmax_rival[neg_i] - model.p_neg_i[neg_i] <= model.phi_bar_neg_i[neg_i]*model.M
    model.p_neg_i_upper_M = pyen.Constraint(model.neg_I, rule=p_neg_i_upper_M)
    
    def mu_bar_k_M(model, k):
        return model.mu_bar_k[k] <= (1-model.phi_bar_k[k])*model.M
    model.mu_bar_k_M = pyen.Constraint(model.K, rule=mu_bar_k_M)
    
    def mu_bar_i_M(model, i):
        return model.mu_bar_i[i] <= (1-model.phi_bar_i[i])*model.M
    model.mu_bar_i_M = pyen.Constraint(model.I, rule=mu_bar_i_M)
    
    def mu_bar_neg_i_M(model, neg_i):
        return model.mu_bar_neg_i[neg_i] <= (1-model.phi_bar_neg_i[neg_i])*model.M
    model.mu_bar_neg_i_M = pyen.Constraint(model.neg_I, rule=mu_bar_neg_i_M)
    
    def mu_dashed_k_M(model, k):
        return model.mu_dashed_k[k] <= (1-model.phi_dash_k[k])*model.M
    model.mu_dashed_k_M = pyen.Constraint(model.K, rule=mu_dashed_k_M)
    
    def mu_dashed_i_M(model, i):
        return model.mu_dashed_i[i] <= (1-model.phi_dashed_i[i])*model.M
    model.mu_dashed_i_M = pyen.Constraint(model.I, rule=mu_dashed_i_M)
    
    def mu_dashed_neg_i_M(model, neg_i):
        return model.mu_dashed_neg_i[neg_i] <= (1-model.phi_dashed_neg_i[neg_i])*model.M
    model.mu_dashed_neg_i_M = pyen.Constraint(model.neg_I, rule=mu_dashed_neg_i_M)
    
    def d_k_lower_M(model, k):
        return model.dem_k[k] <= model.phi_dash_k[k]*model.M
    model.d_k_lower_M = pyen.Constraint(model.K, rule=d_k_lower_M)
    
    def p_i_lower_M(model, i):
        return model.p_i[i] <= model.phi_dashed_i[i]*model.M
    model.p_i_lower_M = pyen.Constraint(model.I, rule=p_i_lower_M)
    
    def p_neg_i_lower_M(model, neg_i):
        return model.p_neg_i[neg_i] <= model.phi_dashed_neg_i[neg_i]*model.M
    model.p_neg_i_lower_M = pyen.Constraint(model.neg_I, rule=p_neg_i_lower_M)
    
    def d_k_upper_bound(model, k):
        return model.dem_k[k] <= model.PDmax[k]
    model.d_k_ub = pyen.Constraint(model.K, rule=d_k_upper_bound)
    
    def p_i_upper_bound(model, i):
        return model.p_i[i] <= model.PGmax_leader[i]
    model.p_i_ub = pyen.Constraint(model.I, rule=p_i_upper_bound)
    
    def p_neg_i_upper_bound(model, neg_i):
        return model.p_neg_i[neg_i] <= model.PGmax_rival[neg_i]
    model.p_neg_i_ub = pyen.Constraint(model.neg_I, rule=p_neg_i_upper_bound)
    
    # Objective
    def objective_function(model):
        return(-sum(model.p_i[i]*model.Cost_leader[i] for i in model.I)
               + sum(model.Cost_demand[k]*model.dem_k[k] for k in model.K)
               - sum(model.p_neg_i[neg_i]*model.Cost_rival[neg_i] for neg_i in model.neg_I)
               - sum(model.mu_bar_k[k]*model.PDmax[k] for k in model.K)
               - sum(model.mu_bar_neg_i[neg_i]*model.PGmax_rival[neg_i] for neg_i in model.neg_I))
    model.obj = pyen.Objective(rule=objective_function, sense=pyen.maximize)
    
    # Optimize
    solver = SolverFactory('gurobi', solver_io="python")
    results = solver.solve(model)
    results.write()
    # # Print objective function
    # model.obj.pprint()
    
    return model

