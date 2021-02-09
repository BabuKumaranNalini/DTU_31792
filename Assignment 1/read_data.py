# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:27:56 2021

@author: ga47jes
"""

import pandas as pd

def read_data(ems, path=None):              
           
    # Check for the file type 
    if path.endswith('.xlsx'):
        print('Reading your excel file, please wait!')
        # obtain the spreadsheet data
        xls = pd.ExcelFile(path)
        
        # read properties
        ems['devices']['pv'] = 5
        ems['devices']['gas_heater'] = {}
        ems['devices']['gas_heater']['maxpow'] = 6
        ems['devices']['gas_heater']['eta'] = 0.8
        ems['devices']['ele_heater'] = 6
        ems['devices']['flex_load'] = {}
        ems['devices']['flex_load']['maxene'] = 2
        ems['devices']['flex_load']['maxpow'] = 2
        
        # read forecasting data and write it into ems object
        ts = pd.read_excel(xls, sheet_name='time_series', usecols='B:I', nrows=96)
        ems['fcst'] = read_forecast(ts)
        
    return ems


def read_forecast(excel_data):
    dict_fcst = excel_data.to_dict('dict')
    for key in dict_fcst:
        dict_fcst[key] = list(dict_fcst[key].values())

    return dict_fcst
