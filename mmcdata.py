"""
mmcdata.py
 
This Python source defines the MMCData class 

An instance of this class is a given set of 'observed' (via instrument
or model) timeseries of U,V,W, and other state variables, along with
associated location (lat/lon), time, and elevation characteristics of
the data stream 
"""

#import os
#import sys
from math import *
import collections
import numpy as np
import datetime as dt
#import pickle
from matplotlib import pyplot as plt
from matplotlib import rcParams, cycler
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
#from scipy import interpolate
#from sklearn import linear_model

class MMCData():
    """A given set of 'observed' (via instrument or model) timeseries of
    U,V,W, and other state variables and any 'derived' (via calculation
    methods) data like mean, perturbation, variance, correlations, etc...
    that are attributes (could be defined or missing a value) in a given
    MMCData instance
    """
    def __init__(self,pklData={},**kwargs):
        self.dataDict = collections.defaultdict(list)
        
        #JAS_Trying to get all records... self.dataSetLength = len(pklData)-1
        self.dataSetLength = len(pklData)
        self.dataSetDict = pklData[0]
        self.dataRecordDict = []
        if self.dataSetLength > 0:
            self._read(pklData,**kwargs)

    def _read(self,pklData,convert_ft_to_m=False):
        """Updates dataRecordDict, dataDict, and dataSetDict"""
        time=[]
        datetime=[]
        z=[]
        u=[]
        v=[]
        w=[]
        theta=[]
        pres=[]
        tke=[]
        tau11=[]
        tau12=[]
        tau13=[]
        tau22=[]
        tau23=[]
        tau33=[]
        hflux=[]
        for i in range(1,self.dataSetLength):
            self.dataRecordDict.append(pklData[i][0])
            time.append(pklData[i][0]['time'])
            #datetime.append((dt.datetime.strptime(pklData[i][0]['date']+"_"+pklData[i][0]['time'].strip(), '%Y-%m-%d_%H:%M:%S') - dt.datetime(1970,1,1)).total_seconds())
            dtstr = pklData[i][0]['date'] + "_" + pklData[i][0]['time'].strip()
            datetime.append(dt.datetime.strptime(dtstr, '%Y-%m-%d_%H:%M:%S'))
            z.append(pklData[i][1][:,0])
            u.append(pklData[i][1][:,1])
            v.append(pklData[i][1][:,2])
            w.append(pklData[i][1][:,3])
            theta.append(pklData[i][1][:,4])
            pres.append(pklData[i][1][:,5])
            tke.append(pklData[i][1][:,6])
            tau11.append(pklData[i][1][:,7])
            tau12.append(pklData[i][1][:,8])
            tau13.append(pklData[i][1][:,9])
            tau22.append(pklData[i][1][:,10])
            tau23.append(pklData[i][1][:,11])
            tau33.append(pklData[i][1][:,12])
            hflux.append(pklData[i][1][:,13])

        # Re-cast fields as numpy arrays and add to 'dataDict' object attribute 
        self.dataDict['datetime'] = np.asarray(datetime)
        if convert_ft_to_m:
            # Convert TTU/SWiFT height to meters from feet ;-(
            self.dataDict['z'] = np.asarray(z)*0.3048
        else:
            # Otherwise expect heights in meters as they should be
            self.dataDict['z'] = np.asarray(z)
        self.dataDict['u']     = np.asarray(u)
        self.dataDict['v']     = np.asarray(v)
        self.dataDict['w']     = np.asarray(w)
        self.dataDict['theta'] = np.asarray(theta)
        self.dataDict['pres']  = np.asarray(pres)
        self.dataDict['tke']   = np.asarray(tke)
        self.dataDict['tau11'] = np.asarray(tau11)
        self.dataDict['tau12'] = np.asarray(tau12)
        self.dataDict['tau13'] = np.asarray(tau13)
        self.dataDict['tau22'] = np.asarray(tau22)
        self.dataDict['tau23'] = np.asarray(tau23)
        self.dataDict['tau33'] = np.asarray(tau33)
        self.dataDict['hflux'] = np.asarray(hflux)
        self.dataDict['wspd']  = np.sqrt(np.square(self.dataDict['u'])+np.square(self.dataDict['v']))
        self.dataDict['wdir']  = (270.0-np.arctan2(self.dataDict['v'],self.dataDict['u'])*180./np.pi)%360 
        #Declare and initialize to 0 the *_mean arrays
        self.dataDict['u_mean']     = np.zeros(self.dataDict['u'].shape)
        self.dataDict['v_mean']     = np.zeros(self.dataDict['u'].shape)
        self.dataDict['w_mean']     = np.zeros(self.dataDict['u'].shape)
        self.dataDict['theta_mean'] = np.zeros(self.dataDict['u'].shape)
        self.dataDict['tke_mean']   = np.zeros(self.dataDict['u'].shape)
        self.dataDict['hflux_mean'] = np.zeros(self.dataDict['u'].shape)
        self.dataDict['uu_mean']    = np.zeros(self.dataDict['u'].shape)
        self.dataDict['uv_mean']    = np.zeros(self.dataDict['u'].shape)
        self.dataDict['uw_mean']    = np.zeros(self.dataDict['u'].shape)
        self.dataDict['vv_mean']    = np.zeros(self.dataDict['u'].shape)
        self.dataDict['vw_mean']    = np.zeros(self.dataDict['u'].shape)
        self.dataDict['ww_mean']    = np.zeros(self.dataDict['u'].shape)
        self.dataDict['wt_mean']    = np.zeros(self.dataDict['u'].shape)
        self.dataDict['wspd_mean']  = np.zeros(self.dataDict['u'].shape)
        self.dataDict['wdir_mean']  = np.zeros(self.dataDict['u'].shape)
        self.dataDict['shear_mean'] = np.zeros(self.dataDict['u'].shape)
 
          
    def getDataSetDict(self):
        return self.dataSetDict
    
    def getDataSetFieldShape(self):
        return self.dataDict['u'].shape
 
    def getRecordDict(self,recNum):
        return self.dataRecordDict[recNum]
 
    def setRunningMeans(self,windowLength,levels):
    #def getDataSetRunningMean(self,windowLength,levels, start_datetime,stop_datetime):
        for k in range(levels):
           #print("setRunningMeans: k = {:d}".format(k))
           self.dataDict['u_mean'][:,k] = running_mean(self.dataDict['u'][:,k],windowLength)
           self.dataDict['v_mean'][:,k] = running_mean(self.dataDict['v'][:,k],windowLength)
           self.dataDict['w_mean'][:,k] = running_mean(self.dataDict['w'][:,k],windowLength)
           self.dataDict['theta_mean'][:,k] = running_mean(self.dataDict['theta'][:,k],windowLength) 
           self.dataDict['tke_mean'][:,k] = running_mean(self.dataDict['tke'][:,k],windowLength)
           self.dataDict['hflux_mean'][:,k] = running_mean(self.dataDict['hflux'][:,k],windowLength)
           self.dataDict['uu_mean'][:,k] = running_mean( np.square(np.subtract(self.dataDict['u'][:,k],self.dataDict['u_mean'][:,k])),windowLength)
           self.dataDict['uv_mean'][:,k] = running_mean( np.multiply(np.subtract(self.dataDict['u'][:,k],self.dataDict['u_mean'][:,k]),
                                                                 np.subtract(self.dataDict['v'][:,k],self.dataDict['v_mean'][:,k])) ,windowLength)
           self.dataDict['uw_mean'][:,k] = running_mean( np.multiply(np.subtract(self.dataDict['u'][:,k],self.dataDict['u_mean'][:,k]),
                                                                 np.subtract(self.dataDict['w'][:,k],self.dataDict['w_mean'][:,k])) ,windowLength)
           self.dataDict['vv_mean'][:,k] = running_mean( np.square(np.subtract(self.dataDict['v'][:,k],self.dataDict['v_mean'][:,k])),windowLength)
           self.dataDict['vw_mean'][:,k] = running_mean( np.multiply(np.subtract(self.dataDict['v'][:,k],self.dataDict['v_mean'][:,k]),
                                                                 np.subtract(self.dataDict['w'][:,k],self.dataDict['w_mean'][:,k])) ,windowLength)
           self.dataDict['ww_mean'][:,k] = running_mean( np.square(np.subtract(self.dataDict['w'][:,k],self.dataDict['w_mean'][:,k])),windowLength)
           self.dataDict['wt_mean'][:,k] = running_mean( np.multiply(np.subtract(self.dataDict['w'][:,k],self.dataDict['w_mean'][:,k]),
                                                                 np.subtract(self.dataDict['theta'][:,k],self.dataDict['theta_mean'][:,k])) ,windowLength)
        self.dataDict['wspd_mean'] = np.sqrt(np.square(self.dataDict['u_mean'])+np.square(self.dataDict['v_mean']))
        #self.dataDict['wdir_mean'] = np.arctan2(self.dataDict['v_mean'],self.dataDict['u_mean'])*180./np.pi+180.0   #From Branko's original, but this seems incorrect...
        self.dataDict['wdir_mean'] = (270.0-np.arctan2(self.dataDict['v_mean'],self.dataDict['u_mean'])*180./np.pi)%360  
 #       self.dataDict['shear_mean']=np.sqrt(np.square(self.uw_mean)+np.square(self.vw_mean))
 
    #
    # Plotting functions (TODO: move to plotting.py)
    #
 
    def plotDataSetByKey(self,xVarKey,yVarKey):
        plt.figure()
        plt.plot(self.dataDict[xVarKey],self.dataDict[yVarKey],'bo-')
        #plt.show(block=False)
        plt.draw()
        #plt.show()
        #plt.pause(0.0001) 
 
    def plotObsVsModelProfileAsSubplot(self,fig,axs,fldString,obsData,obsIndepVar,obsLabel,modelData,modelIndepVar,modelLabel):
        #Set the Marker styles
        obs_marker_style = dict(color='r', linestyle='None', marker='s', markersize=5, markerfacecolor='None')
        model_marker_style = dict(color='b', linestyle='--', marker='o', markersize=3, markerfacecolor='None')
        #Setup up the shared y-axis ticks and labels
        yticks=np.arange(0,251,50)
        ylabels=[]
        ylabels.extend(str(z) for z in range(0,251,50))
        #Compute the standard deviation of obsField
        obs_std = np.std(obsData,axis=0)
        model_std = np.std(modelData,axis=0)
        #Find the x-axis Min,Max, and Interval
        deltaXp = np.nanmax(np.append(np.abs(np.nanmax(np.mean(obsData,axis=0))- \
                                             np.nanmax(np.mean(modelData,axis=0))) \
                             ,np.nanmax(obs_std)))
        deltaXm = np.nanmax(np.append(np.abs(np.nanmin(np.mean(obsData,axis=0))- \
                                             np.nanmin(np.mean(modelData,axis=0))) \
                             ,np.nanmax(obs_std)))
        deltaX = np.nanmax(np.append(deltaXp,deltaXm))
        xTickMin = np.floor(np.nanmin(np.mean(obsData,axis=0))-deltaX)
        xTickMax = np.ceil(np.nanmax(np.mean(obsData,axis=0))+deltaX)
        xTickInterval = np.round_((xTickMax-xTickMin)/3.0,0)
        #print the x-axis characteristics
        print('{:s}'.format(fldString+" x-axis traits..."))
        print('{:s}'.format("xTickMin ="+str(xTickMin)))
        print('{:s}'.format("xTickMax ="+str(xTickMax)))
        print('{:s}'.format("xTickInterval ="+str(xTickInterval)))
        #Setup the x-axis ticks and labels
        xticks=np.arange(xTickMin,xTickMax+xTickInterval,xTickInterval)
        xlabels=[]
        xlabels.extend(str(u).split('.')[0] for u in xticks.tolist())
        #Plot the observations (and uncertainty via errorbars)
        axs.errorbar(np.mean(obsData,axis=0), obsIndepVar, xerr=obs_std, capsize=2, \
                     label=obsLabel, **obs_marker_style)
        axs.errorbar(np.mean(modelData,axis=0), modelIndepVar, xerr=model_std, capsize=2, \
                     label=modelLabel, **model_marker_style)
        #axs.plot(np.mean(modelData,axis=0), modelIndepVar, \
        #         label=modelLabel, **model_marker_style)
        axs.set(xticks=xticks,xticklabels=xlabels,yticks=yticks,yticklabels=ylabels)
        #Format and beautify the axes ticks and limits
        axs.yaxis.set_minor_locator(AutoMinorLocator(4))
        axs.xaxis.set_minor_locator(AutoMinorLocator(4))
        axs.tick_params(direction='in',top=True,right=True,length=10, width=2, which='major')
        axs.tick_params(direction='in',top=True,right=True,length=5, width=1, which='minor')
        axs.tick_params(axis='both', which='major', labelsize=8)
        axs.tick_params(axis='both', which='minor', labelsize=6)
        axs.set_ylim(0.0,np.max(obsIndepVar+50))
        axs.set_xlim(xTickMin,xTickMax)
       
    def plotObsVsModelTimeSeriesAsSubplot(self,fig,axs,fldString, \
                                          obsData,obsIndepVar,obsLabel,obsLevs, \
                                          modelData,modelIndepVar,modelLabel,modelLevs):
        obs_marker_style = dict(linestyle='-')
        model_marker_style = dict(linestyle=':')
        for lev in range(obsData.shape[1]):
           axs.plot(obsIndepVar,obsData[:,lev],label = obsLabel+": "+str(obsLevs[0,lev]),**obs_marker_style)
        for lev in range(modelData.shape[1]):
           axs.plot(modelIndepVar,modelData[:,lev],label = modelLabel+": "+str(modelLevs[0,lev]),**model_marker_style)
 
    def plotSingleSourceTimeSeriesAsSubplot(self,fig,axs,fldString, \
                                            fldData,fldIndepVar,fldLabel,fldLevs):
        fld_marker_style = dict(linestyle='-')
        for lev in range(fldData.shape[1]):
           axs.plot(fldIndepVar,fldData[:,lev],label = fldLabel+": "+str(fldLevs[0,lev]),**fld_marker_style)

#####END OF the MMC_CLASS

### Utility functions for MMC class
def linearly_interpolate_nans(y):
    # Fit a linear regression to the non-nan y values

    # Create X matrix for linreg with an intercept and an index
    X = np.vstack((np.ones(len(y)), np.arange(len(y))))

    # Get the non-NaN values of X and y
    X_fit = X[:, ~np.isnan(y)]
    y_fit = y[~np.isnan(y)].reshape(-1, 1)

    # Estimate the coefficients of the linear regression
    beta = np.linalg.lstsq(X_fit.T, y_fit, rcond=None)[0]

    # Fill in all the nan values using the predicted coefficients
    y.flat[np.isnan(y)] = np.dot(X[:, np.isnan(y)].T, beta)
    return y

def running_mean(x, N):
    M = len(x)
    bad = np.where(np.isnan(x))
    B = bad[0].size
    if (B > 0) and (float(B)/float(M) <= 0.1):
       x = linearly_interpolate_nans(x)
    if (B > 0) and (float(B)/float(M) > 0.1):
       sys.exit("More than 10% data is NaN!")
    y = x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    xavg = (cumsum[N:] - cumsum[:-N]) / N
    for i in range(0,floor(N/2)):
        xavg = np.insert(xavg,i,np.nanmean(y[i:i+N]))
    for i in range(M-floor(N/2)+1,M):
        xavg = np.append(xavg,np.nanmean(y[i-N:i]))
    return xavg

def running_mean2(x,N):
    xavg=[]
    M=len(x)
    for i in range(0,floor(N/2)):
        xavg.append(np.nanmean(x[i:i+N]))
    for i in range(floor(N/2),M-floor(N/2)):
        xavg.append(np.nanmean(x[i-floor(N/2):i+floor(N/2)]))
    for i in range(M-floor(N/2),M):
        xavg.append(np.nanmean(x[i-N:i]))
    return xavg


