# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 12:46:08 2021

@author: IIT HYD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_excel('E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\ap\\ap_wells.xlsx')
data1 = pd.read_excel('E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\ts\\ts_wells.xlsx')
data2 = pd.read_excel('E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\Maharastra\\mh_wells.xlsx')
data3 = pd.read_excel('E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\karnataka\\ka_wells.xlsx')

data_ap1 = data[(data['YEAR_OBS']>=2002) & (data['YEAR_OBS']<=2017)]
p1=data_ap1.groupby('WLCODE')['YEAR_OBS'].count()
#k1 = data_ap1.groupby('WLCODE')['LAT','LON'].nunique()
data_ap_index  = data_ap1.set_index('WLCODE')
data_ap2 = data_ap_index.drop("W04562")
p2 = data_ap1.drop_duplicates('WLCODE')
k1 = p2[['LAT','LON','WLCODE']]
wells_ap = data_ap2[['LAT','LON','YEAR_OBS','POMKH','PREMON','MONSOON','POMRB']]
wells_ap_values = wells_ap[['POMKH','PREMON','MONSOON','POMRB']]

data_ka1 = data3[(data3['YEAR_OBS']>=2002) & (data3['YEAR_OBS']<=2017)]
p3=data_ka1.groupby('WLCODE')['YEAR_OBS'].count()
#k2 = data_ka1.groupby('WLCODE')['LAT','LON'].nunique()
data_ka_index = data_ka1.set_index('WLCODE')
data_ka2 = data_ka_index.drop("W05316")
p4 = data_ka1.drop_duplicates('WLCODE')
k2 = p4[['LAT','LON','WLCODE']]
wells_ka = data_ka2[['LAT','LON','YEAR_OBS','POMKH','PREMON','MONSOON','POMRB']]
wells_ka_values = wells_ka[['POMKH','PREMON','MONSOON','POMRB']]


data_ts1 = data1[(data1['YEAR_OBS']>=2002) & (data1['YEAR_OBS']<=2017)]
p5=data_ts1.groupby('WLCODE')['YEAR_OBS'].count()
#k3 = data_ts1['LAT','LON','WLCODE'].nunique()
p6 = data_ts1.drop_duplicates('WLCODE')
k3 = p6[['LAT','LON','WLCODE']]
data_ts_index = data_ts1.set_index('WLCODE')
data_ts2 = data_ts_index.drop(["W04918","W04950"])
wells_ts = data_ts2[['LAT','LON','YEAR_OBS','POMKH','PREMON','MONSOON','POMRB']]
wells_ts_values = wells_ts[['POMKH','PREMON','MONSOON','POMRB']]

data_mh1 = data2[(data2['YEAR_OBS']>=2002) & (data2['YEAR_OBS']<=2017)]
p7=data_mh1.groupby('WLCODE')['YEAR_OBS'].count()
#k4 = data_mh1.groupby('WLCODE')['LAT','LON'].nunique()
p8 = data_mh1.drop_duplicates('WLCODE')
k4 = p8[['LAT','LON','WLCODE']]
data_mh_index = data_mh1.set_index('WLCODE')
wells_mh = data_mh1[['LAT','LON','YEAR_OBS','POMKH','PREMON','MONSOON','POMRB']]
wells_mh_values = wells_mh[['POMKH','PREMON','MONSOON','POMRB']]



wells = wells_ap_values.stack(dropna=False)
wells = pd.DataFrame(wells)


wells1 = wells_ka_values.stack(dropna=False)
wells1 = pd.DataFrame(wells1)


wells2 = wells_ts_values.stack(dropna=False)
wells2 = pd.DataFrame(wells2)


wells3 = wells_mh_values.stack(dropna=False)
wells3 = pd.DataFrame(wells3)


wells.to_excel("E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\ap\\ap_wells_gridded.xlsx")
k1.to_excel("E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\ap\\ap_well_location_gridded.xlsx")
wells2.to_excel("E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\ts\\ts_wells_gridded.xlsx")
k3.to_excel("E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\ts\\ts_well_location_gridded.xlsx")

wells1.to_excel("E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\karnataka\\ka_wells_gridded.xlsx")
k2.to_excel("E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\karnataka\\ka_well_location_gridded.xlsx")
wells3.to_excel("E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\Maharastra\\mh_wells_gridded.xlsx")
k4.to_excel("E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\Maharastra\\mh_well_location_gridded.xlsx")




