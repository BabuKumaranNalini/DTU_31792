# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 18:33:32 2021

@author: ga47jes
"""

from pyomo.environ import value as get_value
import numpy as np
import pandas as pd

def stochastic_getval(model, n):
    Pg_da, Pw_da = (np.zeros(n) for i in range(2))
    Pw_da[0] = get_value(model.Pw_da[0])
    for i in range(0,2):
        Pg_da[i] = get_value(model.Pg_da[i])
    # res_da = dict({'Pg_da':Pg_da,'Pw_da':Pw_da})
    res_da = pd.DataFrame({'Pg_da':Pg_da, 'Pw_da':Pw_da})
    return res_da
    
        
    