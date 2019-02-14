'''
  What it does: This dictionary contains functions for reading,
                manipulating, and probing WRF (or netCDF) files.

  Who made it: patrick.hawbecker@nrel.gov
  When: 5/11/18
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm    
from netCDF4 import Dataset as ncdf
import pickle
import subprocess


# Find the dimensions of the given WRF file
def getdims(wrff):
    try:
        nx = wrff.dimensions['west_east'].size
    except KeyError:
        print 'No x-dimension'; nx = []
    try:
        ny = wrff.dimensions['south_north'].size
    except KeyError:
        print 'No y-dimension'; ny = []
    try:
        nz = wrff.dimensions['bottom_top'].size
    except KeyError:
        print 'No z-dimension'; nz = []
    try:
        nt = wrff.dimensions['Time'].size
    except KeyError:
        print 'No t-dimension'; nt = []
    return nt,nz,ny,nx

# Get average (over all x,y) heights; staggered and unstaggered
def getavgheight(wrff):
    nt = wrff.dimensions['Time'].size
    try:
        nz = wrff.dimensions['bottom_top'].size
    except KeyError:
        print 'No z-dimension'; return []
    if nt == 1:
        ph  = wrff.variables['PH'][0,:,:,:]
        phb = wrff.variables['PHB'][0,:,:,:]
        hgt = wrff.variables['HGT'][0,:,:]
        z   = np.mean(np.mean(((ph+phb)/9.81) - hgt,axis=1),axis=1)
        zs  = (z[1:] + z[:-1])*0.5
    else:
        z = np.zeros((nt,nz+1))
        for tt in range(0,nt):
            ph  = wrff.variables['PH'][tt,:,:,:]
            phb = wrff.variables['PHB'][tt,:,:,:]
            hgt = wrff.variables['HGT'][tt,:,:]
            z[tt,:] = np.mean(np.mean(((ph+phb)/9.81) - hgt,axis=1),axis=1)
        zs  = (z[:,1:] + z[:,:-1])*0.5
    return z,zs

# Get heights for all x,y,z
def getheight(wrff):
    try:
        nz = wrff.dimensions['bottom_top'].size
    except KeyError:
        print 'No z-dimension'; return []
    ph  = wrff.variables['PH'][0,:,:,:]
    phb = wrff.variables['PHB'][0,:,:,:]
    hgt = wrff.variables['HGT'][0,:,:]

    z   = ((ph+phb)/9.81) - hgt
    zs  = (z[1:,:,:] + z[:-1,:,:])*0.5
    return z,zs

# Get model height at a specific i,j
def getheightloc(wrff,y,x):
    nt = wrff.dimensions['Time'].size
    try:
        nz = wrff.dimensions['bottom_top'].size
    except KeyError:
        print 'No z-dimension'; return []
    if nt == 1:
        ph  = wrff.variables['PH'][0,:,y,x]
        phb = wrff.variables['PHB'][0,:,y,x]
        hgt = wrff.variables['HGT'][0,y,x]
        z   = ((ph+phb)/9.81) - hgt
        zs  = (z[1:] + z[:-1])*0.5
    else:
        z = np.zeros((nt,nz+1))
        for tt in range(0,nt):
            ph  = wrff.variables['PH'][tt,:,y,x]
            phb = wrff.variables['PHB'][tt,:,y,x]
            hgt = wrff.variables['HGT'][tt,y,x]
            z[tt,:] = ((ph+phb)/9.81) - hgt
        zs  = (z[:,1:] + z[:,:-1])*0.5
    return z,zs

# Return all files from a given directory starting with "fstr"
def getwrffiles(fdir,fstr,returnFileNames=True):
    #fdir = file directory; fstr = file string structure (e.g. 'wrfout')
    nwrffs = subprocess.check_output('cd %s && ls %s*' % (fdir,fstr), shell=True).split()
    nt = np.shape(nwrffs)[0]
    if returnFileNames==True:
        return nwrffs,nt
    else:
        return nt

# Return latitude and longitude
def latlon(wrff):
    lat = wrff.variables['XLAT'][0,:,:]
    lon = wrff.variables['XLONG'][0,:,:]
    return lat,lon

#=====================================#
# - - - - - - TOWER DATA  - - - - - - #
# Get the names and locations of all towers in directory (fdir)
def gettowers(fdir,tstr):
    f = open('%s%s' % (fdir,tstr))
    nt = sum(1 for line in f)-3; f.close()
    f = open('%s%s' % (fdir,tstr))
    f.readline(); f.readline(); f.readline()
    tname = []; tij = np.zeros((2,nt))
    for tt in range(0,nt):
        line = f.readline().split()
        tname.append(line[1])
        tij[0,tt] = line[2]; tij[1,tt] = line[3]
    return tname,tij

# Get the i,j location of the given tower
def twrlocij(twrf):
    twr = open(twrf,'r')
    header = twr.readline().replace('(',' ').replace(')',' ').replace(',',' ').split()
    twr.close()
    stni = int(header[6]) - 1
    stnj = int(header[7]) - 1
    return stni,stnj

# Get the lat/lon location of the given tower
def twrlocll(twrf):
    twr = open(twrf,'r')
    header = twr.readline().replace('(',' ').replace(')',' ').replace(',',' ').split()
    twr.close()
    stnlon = float(header[9])
    stnlat = float(header[8])
    return stnlat,stnlon

# Tower class: put tower data into an object variable
class tower():
    def __init__(self,fstr):
        self.fstr = fstr
        self.getvars()
        self.getdata()
    def getvars(self): # find available vars
        varns = subprocess.check_output('ls %s*' % (self.fstr), shell=True).split() 
        nvars = np.shape(varns)[0] # Number of variables
        for vv in range(0,nvars): # Loop over all variables
            varns[vv] = varns[vv].replace(self.fstr,'')
        self.varns = varns # variable names
        self.nvars = nvars # number of variables
    def getdata(self): # Get all the data
        for vv in range(0,self.nvars): # Loop over all variables
            if self.varns[vv] != 'TS': # TS is different structure...
                # Get number of times
                f = open('%s%s' % (self.fstr,self.varns[vv]))
                nt = sum(1 for line in f)-1; f.close()
                # Open again for reading
                f = open('%s%s' % (self.fstr,self.varns[vv]))
                self.header = f.readline().split() # Header information
                for tt in np.arange(0,nt): # Loop over times
                    line = f.readline().split() # reading one profile
                    if tt == 0: # Initialize and get number of heights
                        nz = np.shape(line)[0]-1
                        var = np.zeros((nt,nz))
                        ttime = np.zeros((nt))
                    var[tt,:] = line[1:] # First element is time
                    ttime[tt] = np.float(line[0])
                self.nt   = nt # Number of times
                self.time = ttime # time
                self.nz = nz # Number of heights
                # Set each of the variables to their own name
                if self.varns[vv] == 'PH':
                    self.ph = var
                elif self.varns[vv] == 'QV':
                    self.qv = var
                elif self.varns[vv] == 'TH':
                    self.th = var
                elif self.varns[vv] == 'UU':
                    self.uu = var
                elif self.varns[vv] == 'VV':
                    self.vv = var
                elif self.varns[vv] == 'WW':
                    self.ww = var
            elif self.varns[vv] == 'TS': # Time series are surface variables
                                         # (no height component)
                f = open('%s%s' % (self.fstr,self.varns[vv])) # Number of times
                nt = sum(1 for line in f)-1; f.close()
                f = open('%s%s' % (self.fstr,self.varns[vv])) # Open to get vars
                f.readline() # Skip header
                for tt in np.arange(0,nt): # Loop over all times
                    line = f.readline().split() # One time, all surface variables
                    if tt == 0: # Initialize number of variables
                        nv = np.shape(line)[0]-2
                        var = np.zeros((nt,nv))
                    var[tt,:] = line[2:]
                self.ts = var # Need to look up what tslist outputs to know which
                              # vars are where...

#=====================================#

# Convert WRF times to year, month, day, hour
def wrftimes2hours(wrff):
    nt = np.shape(wrff.variables['Times'][:])[0]
    if nt == 1:
        time = ''.join(wrff.variables['Times'][0])
        year = np.float(time[:4]);    month = np.float(time[5:7])
        day  = np.float(time[8:10]);  hour  = np.float(time[11:13])
        minu = np.float(time[14:16]); sec   = np.float(time[17:19])
        hours = hour + minu/60.0 + sec/(60.0*60.0)
    else:
        year = np.asarray([]); month = np.asarray([])
        day  = np.asarray([]); hour  = np.asarray([])
        minu = np.asarray([]); sec   = np.asarray([])
        for tt in np.arange(0,nt):
            time = ''.join(wrff.variables['Times'][tt])
            year  = np.append(year,np.float(time[:4]))
            month = np.append(month,np.float(time[5:7]))
            day   = np.append(day,np.float(time[8:10]))
            hour  = np.append(hour,np.float(time[11:13]))
            minu  = np.append(minu,np.float(time[14:16]))
            sec   = np.append(sec,np.float(time[17:19]))
        hours = hour + minu/60.0 + sec/(60.0*60.0)
    return [year,month,day,hours]

# Get i,j location from given wrf file and lat/long
def latlon2ij(wrff,latoi,lonoi):
    lat,lon = latlon(wrff)
    dist    = ((lat-latoi)**2 + (lon-lonoi)**2)**0.5
    jj,ii   = np.where(dist==np.min(dist))
    return ii[0],jj[0]

# Unstagger 2D variable on given axis
def unstagger2d(var,ax):
    if ax == 0:
        varu = (var[:-1,:] + var[1:,:])/2.0
    if ax == 1:
        varu = (var[:,:-1] + var[:,1:])/2.0
    return varu

# Unstagger 3D variable on given axis
def unstagger3d(var,ax):
    if ax == 0:
        varu = (var[:-1,:,:] + var[1:,:,:])/2.0
    if ax == 1:
        varu = (var[:,:-1,:] + var[:,1:,:])/2.0
    if ax == 2:
        varu = (var[:,:,:-1] + var[:,:,1:])/2.0
    return varu

# Unstagger 4D variable on given axis
def unstagger4d(var,ax):
    if ax == 1:
        varu = (var[:,:-1,:,:] + var[:,1:,:,:])/2.0
    if ax == 2:
        varu = (var[:,:,:-1,:] + var[:,:,1:,:])/2.0
    if ax == 3:
        varu = (var[:,:,:,:-1] + var[:,:,:,1:])/2.0
    return varu

