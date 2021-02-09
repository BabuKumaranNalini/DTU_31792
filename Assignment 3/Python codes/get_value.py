# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:47:26 2021

@author: ga47jes
"""

def get_val(model, g_data, leader):
    # Get value
    print('\n-----------------------')
    print('Optimization Results:')
    print('-----------------------')
    
    objective_value = model.obj()
    print('Objective value: ', objective_value)
    
    market_price = getattr(model, 'lambda_')
    print('Market price: ', market_price.value)
    
    leader_price = getattr(model, 'offer_i')
    print('\n# Leader cost:')
    for idx in leader_price:
        print('Leader_{}: {}'.format(idx, leader_price[idx].value))
    
    leader_production = getattr(model, 'p_i')
    print('\n# Leader Production:')
    for idx in leader_production:
        print('Leader_{}: {}'.format(idx, leader_production[idx].value))
    
    print('\n# Leader Revenue:')
    for idx in leader_production:
        leader_profit = leader_production[idx].value*(market_price.value - g_data.loc[leader[idx]]['C'])
        print('Leader_{}: {}'.format(idx, leader_profit))
    
    rival_production = getattr(model, 'p_neg_i')
    print('\n# Rival Production:')
    for idx in rival_production:
        print('Rival '+ str(idx)+': ', rival_production[idx].value)
    
    demands = getattr(model, 'dem_k')
    print('\n# Consumption:')
    for idx in demands:
        print('Demand '+ str(idx)+': ', demands[idx].value)
        
    # Check complimentarity
    print('\n--------------------------------')
    print('Complimentarity condition check:')
    print('----------------------------------')
    
    mu_dashed_i = getattr(model, 'mu_dashed_i')
    for idx in mu_dashed_i:
        print('For Leader i = {}:'.format(idx))
        print('mu_dashed_i: ', mu_dashed_i[idx].value)
        print('p_i*mu_dashed_i: ', mu_dashed_i[idx].value*leader_production[idx].value)
        print('\n')
        
    return model