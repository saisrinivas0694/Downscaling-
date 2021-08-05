# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 00:53:36 2021

@author: IIT HYD
"""

import pandas as pd
import numpy as np

data = pd.read_excel("E:/Objective_1/objective_1-20201119T105917Z-003/objective_1/data_quarter/Krishna/Columnwisearrange/canopint_inst.xlsx")
z=pd.DataFrame()
for i in range(560):
  p= data.iloc[:,1+i] - np.mean(data.iloc[:,1+i])
  z=z.append(p)  