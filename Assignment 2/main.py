# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:19:38 2021

@author: ga47jes
"""

import pandas as pd
from stochastic_model import stochastic_opt
from realtime_model import realtime_opt
from stochastic_getvalue import stochastic_getval
from pyomo.environ import value as get_value
from tqdm import tqdm
from rand_scenario import rand_prob

# Choose the number of training and test dataset (Total set = 500)
n_train = 100
n_test = 500 - n_train

# Choose between constant probability and random probability
# phi = [0.02] * n_train 
phi = rand_prob(n_train) #Using dirichlet function

# Get input data
file = 'input_data.xlsx'
g_data = pd.read_excel(file, sheet_name='gen')
Fmax = pd.read_excel(file, sheet_name='line', header=None)
Bmax = pd.read_excel(file, sheet_name='Bmax', header=None)
Scen = pd.read_excel(file, sheet_name='scen', nrows=n_train)
Dem = pd.read_excel(file, sheet_name='demand')

# Stochastic model
model_da = stochastic_opt(g_data,Fmax,Bmax,Scen,Dem,phi)
cost_da = get_value(model_da.obj)
res_da = stochastic_getval(model_da, len(g_data))

# Real time model
Scen_rt = pd.read_excel(file, sheet_name='scen', skiprows=n_train)
cost_rt = cost_sum_rt = 0.0

for i in tqdm(range(len(Scen_rt))):
    model_rt = realtime_opt(g_data,Fmax,Bmax,Scen_rt.iloc[i,0],Dem,res_da)
    cost_rt = get_value(model_rt.obj)    
    # print('Real time cost for test scenario', i, 'is: ', cost_rt )
    cost_sum_rt += cost_rt
     
# Out of sample cost
out_of_sample = cost_da + cost_sum_rt/len(Scen_rt)   

# Print required results
print('\n\nDay ahead cost for training set with', n_train ,'scenario : ', cost_da)
print('Average real time cost for all test scenario : ', cost_sum_rt/len(Scen_rt))  
print('Out of sample cost: ',out_of_sample)
