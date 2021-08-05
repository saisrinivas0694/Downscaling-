# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:11:50 2021

@author: IIT HYD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#data = pd.read_excel('E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\ap\\ap_wells.xlsx')
#data1 = pd.read_excel('E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\ts\\ts_wells.xlsx')
#data2 = pd.read_excel('E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\Maharastra\\mh_wells.xlsx')
#data3 = pd.read_excel('E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\karnataka\\ka_wells.xlsx')
dt = pd.read_csv('E:/Objective_1/objective_1-20201119T105917Z-003/objective_1/cgwb/GRIDWells/wells_2.csv')
data = pd.read_excel('E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\wells\\all_wells.xlsx')
#data['sort_lat']=data['LAT'].astype('float64')   # 2
#data['sort_long']=data['LON'].astype('float64') 

#data_ap = data.sort_values(['sort_lat', 'sort_long'], ascending=True, inplace=False) # 3
#data_ap.drop(['sort_lat', 'sort_long'], axis='columns', inplace=True) # 4
#data.to_csv(file, sep=your_separator)
min_year, max_year = data.YEAR_OBS.min(), data.YEAR_OBS.max()
data=data.groupby('WLCODE').apply(lambda g: g.set_index("YEAR_OBS").reindex(range(min_year, max_year+1))).drop("WLCODE", axis=1).reset_index()
data_ap = data[(data['YEAR_OBS']>=2002) & (data['YEAR_OBS']<=2017)]
p1=data_ap.groupby('WLCODE')['YEAR_OBS'].count()
A = dt['WLCODE']
data_ap =data_ap.set_index('WLCODE')
data_ap2 = data_ap.loc[A,:]

#k1 = data_ap1.groupby('WLCODE')['LAT','LON'].nunique()
#data_ap_index  = data_ap.set_index('WLCODE')
#data_ap2 = p1.loc('W04562')
#p2 = data_ap.drop_duplicates('WLCODE')
#k1 = p2[['LAT','LON','WLCODE']]
wells_ap = data_ap2[['LAT','LON','YEAR_OBS','POMKH','PREMON','MONSOON','POMRB']]
wells_ap_values = wells_ap[['POMKH','PREMON','MONSOON','POMRB']]

#min_year, max_year = data.YEAR_OBS.min(), data.YEAR_OBS.max()
#
#data=data.groupby('WLCODE').apply(lambda g: g.set_index("YEAR_OBS").reindex(range(min_year, max_year+1))).drop("WLCODE", axis=1).reset_index()
wells_ap_values.to_excel('E:/Objective_1/objective_1-20201119T105917Z-003/objective_1/cgwb/GRIDWells/griddedwells.xlsx')