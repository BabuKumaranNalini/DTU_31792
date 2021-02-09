# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 03:20:05 2021

@author: ga47jes
"""

import pyomo.core as pyen
from pyomo.opt import SolverFactory
from pyomo.environ import value as get_value
from pyomo.environ import *
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import time as tm
from datetime import datetime

def create_model(ems_local):
    devices = ems_local['devices']
    
    # Number of time steps
    t_initial = ems_local['time_data']['isteps']
    t_end = ems_local['time_data']['nsteps']
    ts = np.arange(t_initial, t_end)
    
    # Forecast data
    forecast_series = pd.DataFrame.from_dict(ems_local['fcst'])
    
    # create the model object m
    m = pyen.ConcreteModel()
    m.t = pyen.Set(ordered=True, initialize=ts)
    
    # solar
    pv_param = devices['pv']
    m.pv_peak_power = pyen.Param(initialize=pv_param)
    m.solar = pyen.Param(m.t, initialize=1, mutable=True)
    
    # Flex demand
    f_load = devices['flex_load']
    m.fload_pow = pyen.Param(initialize=f_load['maxpow'])
    m.fload_ene = pyen.Param(initialize=f_load['maxene'])
    
    # Gas heater
    g_heat = devices['gas_heater']
    m.g_heat_maxpow = pyen.Param(initialize=g_heat['maxpow'])
    m.g_heat_eta = pyen.Param(initialize=g_heat['eta'])
    
    # Electric heater
    e_heat = devices['ele_heater']
    m.e_heat_maxpow = pyen.Param(initialize=e_heat)
    
    # Price profile
    m.import_price, m.export_price, m.gas_price = (pyen.Param(m.t, initialize=1, mutable=True) for i in range(3))
    
    # Load profile
    m.heat_load, m.elec_load = (pyen.Param(m.t, initialize=1, mutable=True) for i in range(2))
    
    # Get forecasting time series
    for t in m.t:
        m.import_price[t] = forecast_series.loc[t]['ele_price_in']
        m.export_price[t] = forecast_series.loc[t]['ele_price_out']
        m.gas_price[t] = forecast_series.loc[t]['gas_price']
        m.heat_load[t] = forecast_series.loc[t]['load_heat']
        m.elec_load[t] = forecast_series.loc[t]['load_elec']
        m.solar[t] = forecast_series.loc[t]['solar_power']
        
    # Declare variables
    m.g_heat_cap, m.e_heat_cap, m.elec_import, m.elec_export, m.flex_load, m.PV_cap, m.const_load = (pyen.Var(m.t, within=pyen.NonNegativeReals) for i in range(7))
    m.costs = pyen.Var(m.t, within=pyen.Reals)
    
    
    def elec_balance_rule(m, t):
        return m.elec_import[t] + m.PV_cap[t] - m.elec_export[t] - m.elec_load[t] - m.e_heat_cap[t] - m.flex_load[t] == 0

    m.elec_power_balance = pyen.Constraint(m.t, rule=elec_balance_rule, doc='elec_balance')
    
    def heat_balance_rule(m, t):
        return m.g_heat_cap[t] + m.e_heat_cap[t] - m.heat_load[t] == 0

    m.heat_power_balance = pyen.Constraint(m.t, rule=heat_balance_rule, doc='heat_balance')
        
    def cost_sum_rule(m, t):
        return m.costs[t] == 0.25 * (m.g_heat_cap[t] * m.gas_price[t] / m.g_heat_eta
                                     + m.elec_import[t] * m.import_price[t] 
                                     - m.elec_export[t] * m.export_price[t])
    
    m.cost_sum = pyen.Constraint(m.t, rule=cost_sum_rule)
    
    # Constraints    
    # Gas heater
    def gas_heater_max_cap_rule(m, t):
        return m.g_heat_cap[t] <= m.g_heat_maxpow

    m.gas_heater_max_cap_def = pyen.Constraint(m.t, rule=gas_heater_max_cap_rule)
    
    # Electric heater
    def ele_heater_max_cap_rule(m, t):
        return m.e_heat_cap[t] <= m.e_heat_maxpow

    m.ele_heater_max_cap_def = pyen.Constraint(m.t, rule=ele_heater_max_cap_rule)
    
    # Flexible load
    def flex_load_max_rule(m, t):
        return m.flex_load[t] <= m.fload_pow

    m.flex_load_max_def = pyen.Constraint(m.t, rule=flex_load_max_rule)

    def flex_load_energy_rule(m, t):           
        return sum(m.flex_load[t]/4 for t in m.t) == m.fload_ene

    m.flex_load_energy_def = pyen.Constraint(m.t, rule=flex_load_energy_rule)
    
    # PV
    def pv_max_cap_rule(m, t): 
        return m.PV_cap[t] <= m.pv_peak_power*m.solar[t]

    m.pv_max_cap_def = pyen.Constraint(m.t, rule=pv_max_cap_rule)
    
    # elec_import
    def elec_import_rule(m, t):
        return m.elec_import[t] <= 50 * 5000

    m.elec_import_def = pyen.Constraint(m.t, rule=elec_import_rule)

    # elec_export
    def elec_export_rule(m, t):
        return m.elec_export[t] <= 50 * 5000

    m.elec_export_def = pyen.Constraint(m.t, rule=elec_export_rule)
    
    def obj_rule(m):
        # Return sum of total costs over all cost types.
        # Simply calculates the sum of m.costs over all m.cost_types.
        return pyen.summation(m.costs)

    m.obj = pyen.Objective(sense=pyen.minimize, rule=obj_rule, doc='Sum costs by cost type')
    return m
    
def solve_model(m):
    results = SolverFactory('glpk').solve(m)
    results.write()    
    # if results.solver.status:
    #     m.pprint()    
    print('\n Cost optimal = ',m.obj())
    return m
    

def extract_res(m, ems):
    timesteps = np.arange(ems['time_data']['isteps'], ems['time_data']['nsteps'])
    length = len(timesteps)
    
    elec_import, elec_export, elec_demand, heat_demand, pv_power, \
    e_heat, g_heat, pv_pv2demand, pv_pv2grid, ele_heater, flex_load, \
    gas_heater = (np.zeros(length) for i in range(12))
        
    i = 0;
    for idx in timesteps:
        elec_import[i] = get_value(m.elec_import[idx])
        elec_export[i] = get_value(m.elec_export[idx])
        elec_demand[i] = get_value(m.elec_load[idx])
        heat_demand[i] = get_value(m.heat_load[idx])
        ele_heater[i] = get_value(m.e_heat_cap[idx])
        gas_heater[i] = get_value(m.g_heat_cap[idx])
        flex_load[i] = get_value(m.flex_load[idx])
        pv_power[i] = get_value(m.solar[idx]*m.pv_peak_power)
        pv_pv2demand[i] = min(pv_power[i], elec_demand[i])
        pv_pv2grid[i] = pv_power[i] - pv_pv2demand[i]
        i += 1
        
    ems['optplan'] = { 
            'PV_power': list(pv_power), 
            'pv_pv2demand': list(pv_pv2demand), 
            'pv_pv2grid': list(pv_pv2grid),
            'grid_import': list(elec_import),
            'Elec_demand': list(elec_demand), 
            'grid_export': list(elec_export),
            'Heat_demand': list(heat_demand),
            'ele_heater': list(ele_heater),
            'gas_heater': list(gas_heater),
            'flex_load': list(flex_load)            
          }
        
    return ems
    