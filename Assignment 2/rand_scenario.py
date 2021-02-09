# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 17:12:47 2021

@author: ga47jes
"""

import random 
import numpy as np, numpy.random

def rand_scenario():
    res = [random.randrange(1, 7000, 1) for i in range(500)]
    res = [x / 100 for x in res]
    return res

def rand_prob(n):
    rand_dist = np.random.dirichlet(np.ones(n),size=1)
    rand_prob = rand_dist.tolist()[0]
    return rand_prob