# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:15:48 2021

@author: IIT HYD
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_excel('E:/Objective_1/objective_1-20201119T105917Z-003/objective_1/trail_data/TRAIL_SETUP.xlsx',header=0, index_col=0)
data['GWSA']=data['TWSA']*15-data['SM']-data['CWSA']-data['SURFACE RUNOFF']
K=np.corrcoef(data['GWSA'],data['NDVI'])  
# create the dataset based on the wet year, normal year and dry year

data_dry = data.iloc[0:21,:]
data_dry=data_dry.append(data.iloc[117:129,:])
data_wet = data.iloc[33:45,:]
data_wet=data_wet.append(data.iloc[57:69,:])
data_wet=data_wet.append(data.iloc[93:105,:])
data_normal = data.iloc[21:33,:]
data_normal=data_normal.append(data.iloc[45:57,:])
data_normal=data_normal.append(data.iloc[69:93,:])
data_normal=data_normal.append(data.iloc[106:117,:])
data_normal=data_normal.append(data.iloc[128:183,:])

# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(data_dry['TWSA'],data_dry['NDVI'])
# plt.annotate("R = {:.2f}".format(np.corrcoef(data_dry['TWSA'],data_dry['NDVI'])[0,1]),(-12,0.15),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(data_dry['TWSA'],data_dry['NDVI'], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Terrestrial Water Storage Anomalies (cm)",fontdict=font)
# plt.ylabel("NDVI",fontdict=font)
# fig.savefig("Dry_Year_twsa_ndvi.tif",dpi=600)

# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(data_normal['TWSA'],data_normal['NDVI'])
# plt.annotate("R = {:.2f}".format(np.corrcoef(data_normal['TWSA'],data_normal['NDVI'])[0,1]),(-12,0.15),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(data_normal['TWSA'],data_normal['NDVI'], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Terrestrial Water Storage Anomalies (cm)",fontdict=font)
# plt.ylabel("NDVI",fontdict=font)
# fig.savefig("normal_Year_twsa_ndvi.tif",dpi=600)

# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(data_wet['TWSA'],data_wet['NDVI'])
# plt.annotate("R = {:.2f}".format(np.corrcoef(data_wet['TWSA'],data_wet['NDVI'])[0,1]),(-12,0.15),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(data_wet['TWSA'],data_wet['NDVI'], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Terrestrial Water Storage Anomalies (cm)",fontdict=font)
# plt.ylabel("NDVI",fontdict=font)
# fig.savefig("wet_Year_twsa_ndvi.tif",dpi=600)


# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(data_dry['TWSA'],data_dry['SM'])
# plt.annotate("R = {:.2f}".format(np.corrcoef(data_dry['TWSA'],data_dry['SM'])[0,1]),(-12,150),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(data_dry['TWSA'],data_dry['SM'], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Terrestrial Water Storage Anomalies (cm)",fontdict=font)
# plt.ylabel("SM (mm)",fontdict=font)
# fig.savefig("Dry_Year_twsa_SM.tif",dpi=600)


# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(data_normal['TWSA'],data_normal['SM'])
# plt.annotate("R = {:.2f}".format(np.corrcoef(data_normal['TWSA'],data_normal['SM'])[0,1]),(-12,150),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(data_normal['TWSA'],data_normal['SM'], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Terrestrial Water Storage Anomalies (cm)",fontdict=font)
# plt.ylabel("SM (mm)",fontdict=font)
# fig.savefig("normal_Year_twsa_SM.tif",dpi=600)


# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(data_wet['TWSA'],data_wet['SM'])
# plt.annotate("R = {:.2f}".format(np.corrcoef(data_wet['TWSA'],data_wet['SM'])[0,1]),(-12,150),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(data_wet['TWSA'],data_wet['SM'], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Terrestrial Water Storage Anomalies (cm)",fontdict=font)
# plt.ylabel("SM (mm)",fontdict=font)
# fig.savefig("wet_Year_twsa_SM.tif",dpi=600)



r_dry_ndvi=np.corrcoef(data_dry['TWSA'],data_dry['NDVI'])
r_wet_ndvi=np.corrcoef(data_wet['TWSA'],data_wet['NDVI'])
r_normal_ndvi=np.corrcoef(data_normal['TWSA'],data_normal['NDVI'])
r_dry_sm=np.corrcoef(data_dry['TWSA'],data_dry['SM'])
r_wet_sm=np.corrcoef(data_wet['TWSA'],data_wet['SM'])
r_normal_sm=np.corrcoef(data_normal['TWSA'],data_normal['SM'])
r_dry_ndvi1=np.corrcoef(data_dry['GWSA'],data_dry['NDVI'])
r_wet_ndvi1=np.corrcoef(data_wet['GWSA'],data_wet['NDVI'])
r_normal_ndvi1=np.corrcoef(data_normal['GWSA'],data_normal['NDVI'])
r_dry_sm1=np.corrcoef(data_dry['GWSA'],data_dry['SM'])
r_wet_sm1=np.corrcoef(data_wet['GWSA'],data_wet['SM'])
r_normal_sm1=np.corrcoef(data_normal['GWSA'],data_normal['SM'])
#Lagged NDVI Time series
laggedndvi_1 = data['NDVI'].shift(1)
laggedndvi_2 = data['NDVI'].shift(2)
laggedndvi_3 = data['NDVI'].shift(3)
#lagged gwsa time series
gwsa1 = data['GWSA'].iloc[1:,]
gwsa2 = data['GWSA'].iloc[2:,]
gwsa3 = data['GWSA'].iloc[3:,]

r_lagged_1 = np.corrcoef(gwsa1,laggedndvi_1.iloc[1:,])
r_lagged_2 = np.corrcoef(gwsa2,laggedndvi_2.iloc[2:,])
r_lagged_3 = np.corrcoef(gwsa3,laggedndvi_3.iloc[3:,])

# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(gwsa1,laggedndvi_1.iloc[1:,])
# plt.annotate("R = {:.2f}".format(np.corrcoef(gwsa1,laggedndvi_1.iloc[1:,])[0,1]),(-200,0.15),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(gwsa1,laggedndvi_1.iloc[1:,], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Groundwater Storage Anomalies (mm)",fontdict=font)
# plt.ylabel("NDVI",fontdict=font)
# fig.savefig("gwsa_ndvi_lagged1.tif",dpi=600)



# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(gwsa2,laggedndvi_2.iloc[2:,])
# plt.annotate("R = {:.2f}".format(np.corrcoef(gwsa2,laggedndvi_2.iloc[2:,])[0,1]),(-200,0.15),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(gwsa2,laggedndvi_2.iloc[2:,], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Groundwater Storage Anomalies (mm)",fontdict=font)
# plt.ylabel("NDVI",fontdict=font)
# fig.savefig("gwsa_ndvi_lagged2.tif",dpi=600)

# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(gwsa3,laggedndvi_3.iloc[3:,])
# plt.annotate("R = {:.2f}".format(np.corrcoef(gwsa3,laggedndvi_3.iloc[3:,])[0,1]),(-200,0.15),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(gwsa3,laggedndvi_3.iloc[3:,], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Groundwater Storage Anomalies (mm)",fontdict=font)
# plt.ylabel("NDVI",fontdict=font)
# fig.savefig("gwsa_ndvi_lagged3.tif",dpi=600)







# Dry years

laggedndvi_1_dry = data_dry['NDVI'].shift(1)
laggedndvi_2_dry = data_dry['NDVI'].shift(2)
laggedndvi_3_dry = data_dry['NDVI'].shift(3)
#lagged gwsa time series
gwsa1_dry = data_dry['GWSA'].iloc[1:,]
gwsa2_dry = data_dry['GWSA'].iloc[2:,]
gwsa3_dry = data_dry['GWSA'].iloc[3:,]

r_lagged_1_dry = np.corrcoef(gwsa1_dry,laggedndvi_1_dry.iloc[1:,])
r_lagged_2_dry = np.corrcoef(gwsa2_dry,laggedndvi_2_dry.iloc[2:,])
r_lagged_3_dry = np.corrcoef(gwsa3_dry,laggedndvi_3_dry.iloc[3:,])


# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(gwsa1_dry,laggedndvi_1_dry.iloc[1:,])
# plt.annotate("R = {:.2f}".format(np.corrcoef(gwsa1_dry,laggedndvi_1_dry.iloc[1:,])[0,1]),(-200,0.15),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(gwsa1_dry,laggedndvi_1_dry.iloc[1:,], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Groundwater Storage Anomalies (mm)",fontdict=font)
# plt.ylabel("NDVI",fontdict=font)
# fig.savefig("gwsa_ndvi_lagged1_dry.tif",dpi=600)



# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(gwsa2_dry,laggedndvi_2_dry.iloc[2:,])
# plt.annotate("R = {:.2f}".format(np.corrcoef(gwsa2_dry,laggedndvi_2_dry.iloc[2:,])[0,1]),(-200,0.15),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(gwsa2_dry,laggedndvi_2_dry.iloc[2:,], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Groundwater Storage Anomalies (mm)",fontdict=font)
# plt.ylabel("NDVI",fontdict=font)
# fig.savefig("gwsa_ndvi_lagged2_dry.tif",dpi=600)

# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(gwsa3_dry,laggedndvi_3_dry.iloc[3:,])
# plt.annotate("R = {:.2f}".format(np.corrcoef(gwsa3_dry,laggedndvi_3_dry.iloc[3:,])[0,1]),(-200,0.15),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(gwsa3_dry,laggedndvi_3_dry.iloc[3:,], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Groundwater Storage Anomalies (mm)",fontdict=font)
# plt.ylabel("NDVI",fontdict=font)
# fig.savefig("gwsa_ndvi_lagged3_dry.tif",dpi=600)







#Normal Year

laggedndvi_1_normal = data_normal['NDVI'].shift(1)
laggedndvi_2_normal = data_normal['NDVI'].shift(2)
laggedndvi_3_normal = data_normal['NDVI'].shift(3)
#lagged gwsa time series
gwsa1_normal = data_normal['GWSA'].iloc[1:,]
gwsa2_normal = data_normal['GWSA'].iloc[2:,]
gwsa3_normal = data_normal['GWSA'].iloc[3:,]

r_lagged_1_normal = np.corrcoef(gwsa1_normal,laggedndvi_1_normal.iloc[1:,])
r_lagged_2_normal = np.corrcoef(gwsa2_normal,laggedndvi_2_normal.iloc[2:,])
r_lagged_3_normal = np.corrcoef(gwsa3_normal,laggedndvi_3_normal.iloc[3:,])


# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(gwsa1_normal,laggedndvi_1_normal.iloc[1:,])
# plt.annotate("R = {:.2f}".format(np.corrcoef(gwsa1_normal,laggedndvi_1_normal.iloc[1:,])[0,1]),(-200,0.15),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(gwsa1_normal,laggedndvi_1_normal.iloc[1:,], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Groundwater Storage Anomalies (mm)",fontdict=font)
# plt.ylabel("NDVI",fontdict=font)
# fig.savefig("gwsa_ndvi_lagged1_normal.tif",dpi=600)



# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(gwsa2_normal,laggedndvi_2_normal.iloc[2:,])
# plt.annotate("R = {:.2f}".format(np.corrcoef(gwsa2_normal,laggedndvi_2_normal.iloc[2:,])[0,1]),(-200,0.15),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(gwsa2_normal,laggedndvi_2_normal.iloc[2:,], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Groundwater Storage Anomalies (mm)",fontdict=font)
# plt.ylabel("NDVI",fontdict=font)
# fig.savefig("gwsa_ndvi_lagged2_normal.tif",dpi=600)

# font ={'family':'times new roman','color':'black','weight':'normal','size':30}
# fig=plt.figure(figsize=(20,10))
# plt.scatter(gwsa3_normal,laggedndvi_3_normal.iloc[3:,])
# plt.annotate("R = {:.2f}".format(np.corrcoef(gwsa3_normal,laggedndvi_3_normal.iloc[3:,])[0,1]),(-200,0.15),fontsize=30)
# axes = plt.gca()
# m, b = np.polyfit(gwsa3_normal,laggedndvi_3_normal.iloc[3:,], 1)
# X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
# plt.plot(X_plot, m*X_plot + b, '-')
# plt.xlabel("Groundwater Storage Anomalies (mm)",fontdict=font)
# plt.ylabel("NDVI",fontdict=font)
# fig.savefig("gwsa_ndvi_lagged3_normal.tif",dpi=600)






#wet year

laggedndvi_1_wet = data_wet['NDVI'].shift(1)
laggedndvi_2_wet = data_wet['NDVI'].shift(2)
laggedndvi_3_wet = data_wet['NDVI'].shift(3)
#lagged gwsa time series
gwsa1_wet = data_wet['GWSA'].iloc[1:,]
gwsa2_wet = data_wet['GWSA'].iloc[2:,]
gwsa3_wet = data_wet['GWSA'].iloc[3:,]

r_lagged_1_wet = np.corrcoef(gwsa1_wet,laggedndvi_1_wet.iloc[1:,])
r_lagged_2_wet = np.corrcoef(gwsa2_wet,laggedndvi_2_wet.iloc[2:,])
r_lagged_3_wet = np.corrcoef(gwsa3_wet,laggedndvi_3_wet.iloc[3:,])

#font ={'family':'times new roman','color':'black','weight':'normal','size':30}
#fig=plt.figure(figsize=(20,10))
#plt.scatter(gwsa1_wet,laggedndvi_1_wet.iloc[1:,])
#plt.annotate("R = {:.2f}".format(np.corrcoef(gwsa1_wet,laggedndvi_1_wet.iloc[1:,])[0,1]),(-200,0.15),fontsize=30)
#axes = plt.gca()
#m, b = np.polyfit(gwsa1_wet,laggedndvi_1_wet.iloc[1:,], 1)
#X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
#plt.plot(X_plot, m*X_plot + b, '-')
#plt.xlabel("Groundwater Storage Anomalies (mm)",fontdict=font)
#plt.ylabel("NDVI",fontdict=font)
#fig.savefig("gwsa_ndvi_lagged1_wet.tif",dpi=600)
#
#
#
#font ={'family':'times new roman','color':'black','weight':'normal','size':30}
#fig=plt.figure(figsize=(20,10))
#plt.scatter(gwsa2_wet,laggedndvi_2_wet.iloc[2:,])
#plt.annotate("R = {:.2f}".format(np.corrcoef(gwsa2_wet,laggedndvi_2_wet.iloc[2:,])[0,1]),(-200,0.15),fontsize=30)
#axes = plt.gca()
#m, b = np.polyfit(gwsa2_wet,laggedndvi_2_wet.iloc[2:,], 1)
#X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
#plt.plot(X_plot, m*X_plot + b, '-')
#plt.xlabel("Groundwater Storage Anomalies (mm)",fontdict=font)
#plt.ylabel("NDVI",fontdict=font)
#fig.savefig("gwsa_ndvi_lagged2_wet.tif",dpi=600)
#
#font ={'family':'times new roman','color':'black','weight':'normal','size':30}
#fig=plt.figure(figsize=(20,10))
#plt.scatter(gwsa3_wet,laggedndvi_3_wet.iloc[3:,])
#plt.annotate("R = {:.2f}".format(np.corrcoef(gwsa3_wet,laggedndvi_3_wet.iloc[3:,])[0,1]),(-200,0.15),fontsize=30)
#axes = plt.gca()
#m, b = np.polyfit(gwsa3_wet,laggedndvi_3_wet.iloc[3:,], 1)
#X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
#plt.plot(X_plot, m*X_plot + b, '-')
#plt.xlabel("Groundwater Storage Anomalies (mm)",fontdict=font)
#plt.ylabel("NDVI",fontdict=font)
#fig.savefig("gwsa_ndvi_lagged3_wet.tif",dpi=600)
#
