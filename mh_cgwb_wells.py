# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 20:04:31 2021

@author: IIT HYD
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

import sys
import csv
import glob


# get data file names
path =r'E:\Objective_1\objective_1-20201119T105917Z-003\objective_1\cgwb\Maharastra'
filenames = glob.glob(path + "/*.csv")

dfs = []

df = pd.DataFrame()
for f in filenames:
    data = pd.read_csv(f)
    df = df.append(data)    
df1= df    
df['sort_year'] =df['YEAR_OBS'].astype('float64')
df.sort_values(['sort_year'],ascending = True, inplace=True)
df.drop(['sort_year'],axis='columns',inplace = True)
df['sort_lat']=df['LAT'].astype('float64')   # 2
df['sort_long']=df['LON'].astype('float64') 

df.sort_values(['sort_lat', 'sort_long'], ascending=True, inplace=True) # 3
df.drop(['sort_lat', 'sort_long'], axis='columns', inplace=True) # 4


p5=df.groupby('LAT')['WLCODE'].nunique()
p6 = df.drop_duplicates('WLCODE')
k=p6[['LAT','LON','WLCODE']]



p3 = pd.isnull(df['YEAR_OBS'])
p4 = df.dropna(axis=0)
p7 = p4[(p4.YEAR_OBS >=2002)&(p4.YEAR_OBS <=2018)]
pk=pd.DataFrame()
for k in range(6):
    for i in range(7):
        p8 = p7[(p7.LAT>=(13+k)) & (p7.LAT<=(14+k)) & (p7.LON>=(73+i)) & (p7.LON<=(74+i))]
        pk = pk.append(p8)
p9 = pd.DataFrame(pk.groupby('WLCODE')['YEAR_OBS'].count())
p10 = p9[p9.YEAR_OBS>=13]
well = pd.DataFrame()

for i in p10.index:
    p11 = df[df['WLCODE']==i]
    well = well.append(p11)
#    
#well.to_excel('E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\Maharastra\\mh_wells.xlsx')   
#well_grp=well.groupby('LAT')['WLCODE'].nunique()
#well_ind = well.drop_duplicates('WLCODE')
#well_loc=well_ind[['LAT','LON','WLCODE']]   
#well_loc.to_excel('E:\\Objective_1\\objective_1-20201119T105917Z-003\\objective_1\\cgwb\\Maharastra\\mh_wells_locations.xlsx')
##k.to_excel(r'E:\Objective_1\objective_1-20201119T105917Z-003\objective_1\cgwb\karnataka\karnataka_welllocation.xlsx')