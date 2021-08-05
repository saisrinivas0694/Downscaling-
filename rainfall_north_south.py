# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 13:27:17 2021

@author: IIT HYD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

rainfall_data = pd.read_excel('E:/Objective_1/objective_1-20201119T105917Z-003/objective_1/data_1/precipitation/trmm_rainfall_new_filled_krishna.xlsx')
#rainfall_lat = rainfall_data[['lon 74.500 lat 16.500','lon 77.500 lat 14.500']]

#rainfall =rainfall_data.mean(axis=1)
rainfall_intensity=rainfall_data[['lon 73.500 lat 18.500']]*0.01*24*30
#rainfall_intensity=pd.DataFrame(rainfall_intensity,columns=['B'])
k=pd.DataFrame()
for i in range(16):
    k[i]=np.sum(rainfall_intensity.iloc[0+(i*12):12+(i*12)])
#
#p=pd.DataFrame(np.zeros([12,16]))
#p1 =list()
#for i in np.arange(16):
#    p1 = pd.DataFrame(rainfall_intensity['B'][0+(i*12):12+(i*12)])
#    
#
#test_list = np.arange(16)
#  
## printing original list  
#print("The original list : " +  str(test_list)) 
#  
## declaring magnitude of repetition 
#K = 12
#  
## using list comprehension 
## repeat elements K times 
#res =  [ele for ele in test_list for i in range(K)] 
#  
## printing result  
#print("The list after adding elements :  " + str(res)) 
#
#rainfall_intensity['K'] = res
#
#for i in np.arange(16):
#    rslt_df = pd.DataFrame(rainfall_intensity['B'].loc[rainfall_intensity['K'] == i])
#    p = pd.concat([p,rslt_df.set_index(p.index)],axis=1,ignore_index = True)
#    
#p=p.drop(np.arange(16),axis=1)
#
#p.columns = np.arange(2002,2018)
#p.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
#p1=p.transpose()
##p.to_excel(r'E:/Objective_1/objective_1-20201119T105917Z-003/objective_1/data_1/precipitation/monthly_variation_rainfall.xlsx')
#
#o=pd.melt(p)
#o.columns = ["Time (Year)","Rainfall"]
#plt.figure(figsize=(20,10))
#bplot=sns.boxplot(x="Time (Year)", y="Rainfall", data=o,width=0.5,
#                 palette="colorblind")
#bplot=sns.stripplot(x="Time (Year)", y="Rainfall", data=o,
#                   jitter=True, 
#                   marker='o', 
#                   alpha=0.75,
#                   color='black')
#bplot.set_xlabel("Time (Year)", 
#                fontsize=20)
# 
#bplot.set_ylabel("Rainfall",
#                fontsize=20)
# 
#bplot.tick_params(labelsize=10)
#
#plot_file_name="boxplot_and_swarmplot_with_seaborn.tiff"
# 
## save as jpeg
#bplot.figure.savefig(plot_file_name,
#                    format='tiff',
#                    dpi=600,)