     # -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:02:27 2021

@author: Babu Kumaran Nalini (ga47jes)
"""
import pandas as pd
from bilevel_prgm import bilevel_opt
from get_value import get_val

# Get input data
file = 'input_data_24bus.xlsx'
g_data = pd.read_excel(file, sheet_name='gen')
Fmax = pd.read_excel(file, sheet_name='line', header=None)
Bmax = pd.read_excel(file, sheet_name='Bmax', header=None)
Dem = pd.read_excel(file, sheet_name='demand')

# Select leader generator
leader = [0,20]  # Choose generators which are the leaders
model = bilevel_opt(g_data,Fmax,Bmax,Dem,leader)
model = get_val(model, g_data, leader)
