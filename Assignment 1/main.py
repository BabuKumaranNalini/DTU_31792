# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:27:45 2021

@author: ga47jes
"""

from read_data import read_data
from primal_ems import create_model, solve_model, extract_res

# configure and read data
ems = {'time_data': {}, 'devices': {}}
ems['time_data']['isteps'] = 0
ems['time_data']['nsteps'] = 96
ems = read_data(ems, 'input_data.xlsx')

# create lp model and execute
model = create_model(ems)
model = solve_model(model)
ems = extract_res(model, ems)