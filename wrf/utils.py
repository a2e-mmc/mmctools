'''
  What it does: This dictionary contains functions for reading,
                manipulating, and probing WRF (or netCDF) files.

  Who made it: patrick.hawbecker@nrel.gov
  When: 5/11/18

  Notes:
  - Utility functions should automatically handle input data in either
    netCDF4.Dataset or xarray.Dataset formats.

'''
from __future__ import print_function
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_wrf_dims(wrfdata):
    ''' Find the dimensions of the given WRF file'''
    try:
        nx = wrfdata.dimensions['west_east'].size
    except KeyError:
        print('No x-dimension'); nx = []
    try:
        ny = wrfdata.dimensions['south_north'].size
    except KeyError:
        print('No y-dimension'); ny = []
    try:
        nz = wrfdata.dimensions['bottom_top'].size
    except KeyError:
        print('No z-dimension'); nz = []
    try:
        nt = wrfdata.dimensions['Time'].size
    except KeyError:
        print('No t-dimension'); nt = []
    return nt,nz,ny,nx

def get_avg_height(wrfdata):
    '''Get average (over all x,y) heights; staggered and unstaggered'''
    nt = wrfdata.dimensions['Time'].size
    try:
        nz = wrfdata.dimensions['bottom_top'].size
    except KeyError:
        print('No z-dimension'); return []
    if nt == 1:
        ph  = wrfdata.variables['PH'][0,:,:,:]
        phb = wrfdata.variables['PHB'][0,:,:,:]
        hgt = wrfdata.variables['HGT'][0,:,:]
        zs  = np.mean(np.mean(((ph+phb)/9.81) - hgt,axis=1),axis=1)
        z   = (zs[1:] + zs[:-1])*0.5
    else:
        zs = np.zeros((nt,nz+1))
        for tt in range(0,nt):
            ph  = wrfdata.variables['PH'][tt,:,:,:]
            phb = wrfdata.variables['PHB'][tt,:,:,:]
            hgt = wrfdata.variables['HGT'][tt,:,:]
            zs[tt,:] = np.mean(np.mean(((ph+phb)/9.81) - hgt,axis=1),axis=1)
        z  = (zs[:,1:] + zs[:,:-1])*0.5
    return z,zs

def get_height(wrfdata):
    '''Get heights for all x,y,z'''
    try:
        nz = wrfdata.dimensions['bottom_top'].size
    except KeyError:
        print('No z-dimension'); return []
    ph  = wrfdata.variables['PH'][0,:,:,:]
    phb = wrfdata.variables['PHB'][0,:,:,:]
    hgt = wrfdata.variables['HGT'][0,:,:]

    zs  = ((ph+phb)/9.81) - hgt
    z   = (zs[1:,:,:] + zs[:-1,:,:])*0.5
    return z,zs

def get_height_at_ind(wrfdata,j,i):
    '''Get model height at a specific j,i'''
    nt = wrfdata.dimensions['Time'].size
    try:
        nz = wrfdata.dimensions['bottom_top'].size
    except KeyError:
        print('No z-dimension'); return []
    if nt == 1:
        ph  = wrfdata.variables['PH'][0,:,j,i]
        phb = wrfdata.variables['PHB'][0,:,j,i]
        hgt = wrfdata.variables['HGT'][0,j,i]
        zs  = ((ph+phb)/9.81) - hgt
        z   = (zs[1:] + zs[:-1])*0.5
    else:
        zs = np.zeros((nt,nz+1))
        for tt in range(0,nt):
            ph  = wrfdata.variables['PH'][tt,:,j,i]
            phb = wrfdata.variables['PHB'][tt,:,j,i]
            hgt = wrfdata.variables['HGT'][tt,j,i]
            zs[tt,:] = ((ph+phb)/9.81) - hgt
        z  = (zs[:,1:] + zs[:,:-1])*0.5
    return z,zs

def get_wrf_files(dpath='.',prefix='wrfout',returnFileNames=True,
                  fullpath=False,sort=True):
    '''
    Return all files from a given directory starting with "prefix"
    dpath = file directory; prefix = file string structure (e.g. 'wrfout')
    '''
    nwrffs = glob.glob(os.path.join(dpath,prefix+'*'))
    nt = np.shape(nwrffs)[0]
    if returnFileNames==True:
        if not fullpath:
            nwrffs = [ os.path.split(fpath)[-1] for fpath in nwrffs ]
        if sort:
            nwrffs.sort()
        return nwrffs,nt
    else:
        return nt

def latlon(wrfdata):
    '''Return latitude and longitude'''
    lat = wrfdata.variables['XLAT'][0,:,:]
    lon = wrfdata.variables['XLONG'][0,:,:]
    return lat,lon

def get_tower_names(fdir,tstr):
    '''Get the names and locations of all towers in directory (fdir)'''
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

def twrloc_ij(twr_file_name):
    '''Get the i,j location of the given tower'''
    twr = open(twr_file_name,'r')
    header = twr.readline().replace('(',' ').replace(')',' ').replace(',',' ').split()
    twr.close()
    stni = int(header[6]) - 1
    stnj = int(header[7]) - 1
    return stni,stnj

def twrloc_ll(twr_file_name):
    '''Get the lat/lon location of the given tower'''
    twr = open(twr_file_name,'r')
    header = twr.readline().replace('(',' ').replace(')',' ').replace(',',' ').split()
    twr.close()
    stnlon = float(header[9])
    stnlat = float(header[8])
    return stnlat,stnlon

class Tower():
    ''' 
    Tower class: put tower data into an object variable
    Call with: twr = wrfdict.tower('[path to towers]/[tower abrv.].d0[domain].*')
    '''
    def __init__(self,fstr):
        self.fstr = fstr
        self.getvars()
        self.getdata()
    def getvars(self): # find available vars
        varns = glob.glob('{:s}*'.format(self.fstr))
        nvars = np.shape(varns)[0] # Number of variables
        for vv in range(0,nvars): # Loop over all variables
            varns[vv] = varns[vv].replace(self.fstr,'').replace('.','')
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
                elif self.varns[vv] == 'PP':
                    self.pp = var
            elif self.varns[vv] == 'TS': # Time series are surface variables
                                         # (no height component)
                f = open('%s%s' % (self.fstr,self.varns[vv])) # Number of times
                nt = sum(1 for line in f)-1; f.close()
                f = open('%s%s' % (self.fstr,self.varns[vv])) # Open to get vars
                header = f.readline().replace('(',' ').replace(')',' ').replace(',',' ').split() # Skip header
                self.longname = header[0]
                self.abbr     = header[3]
                self.lat      = header[4]
                self.lon      = header[5]
                self.loci     = header[6]
                self.locj     = header[7]
                self.stationz = header[10]
                for tt in np.arange(0,nt): # Loop over all times
                    line = f.readline().split() # One time, all surface variables
                    if tt == 0: # Initialize number of variables
                        nv = np.shape(line)[0]-2
                        var = np.zeros((nt,nv))
                    var[tt,:] = line[2:]
                self.ts = var # Need to look up what tslist outputs to know which
                              # vars are where...


def wrf_times_to_hours(wrfdata,timename='Times'):
    '''Convert WRF times to year, month, day, hour'''
    nt = np.shape(wrfdata.variables['Times'][:])[0]
    if nt == 1:
        time = ''.join(wrfdata.variables[timename][0])
        year = np.float(time[:4]);    month = np.float(time[5:7])
        day  = np.float(time[8:10]);  hour  = np.float(time[11:13])
        minu = np.float(time[14:16]); sec   = np.float(time[17:19])
        hours = hour + minu/60.0 + sec/(60.0*60.0)
    else:
        year = np.asarray([]); month = np.asarray([])
        day  = np.asarray([]); hour  = np.asarray([])
        minu = np.asarray([]); sec   = np.asarray([])
        for tt in np.arange(0,nt):
            time = ''.join(wrfdata.variables[timename][tt])
            year  = np.append(year,np.float(time[:4]))
            month = np.append(month,np.float(time[5:7]))
            day   = np.append(day,np.float(time[8:10]))
            hour  = np.append(hour,np.float(time[11:13]))
            minu  = np.append(minu,np.float(time[14:16]))
            sec   = np.append(sec,np.float(time[17:19]))
        hours = hour + minu/60.0 + sec/(60.0*60.0)
    return [year,month,day,hours]

def wrf_times_to_datetime(wrfdata,timename='Times',format='%Y-%m-%d_%H:%M:%S'):
    """Convert WRF times to datetime format"""
    timestrs = wrfdata.variables['Times'][:]
    return [ datetime.strptime(s.values.tostring().decode(), format) for s in timestrs ]

def latlon_to_ij(wrfdata,lat,lon):
    '''Get i,j location from given wrf file and lat/long'''
    glat,glon = latlon(wrfdata)
    dist = ((glat-lat)**2 + (glon-lon)**2)**0.5
    jj,ii = np.where(dist==np.min(dist))
    return ii[0],jj[0]

def unstagger2d(var,axis):
    '''Unstagger 2D variable on given axis'''
    if axis == 0:
        varu = (var[:-1,:] + var[1:,:])/2.0
    if axis == 1:
        varu = (var[:,:-1] + var[:,1:])/2.0
    return varu

def unstagger3d(var,axis):
    '''Unstagger 3D variable on given axis'''
    if axis == 0:
        varu = (var[:-1,:,:] + var[1:,:,:])/2.0
    if axis == 1:
        varu = (var[:,:-1,:] + var[:,1:,:])/2.0
    if axis == 2:
        varu = (var[:,:,:-1] + var[:,:,1:])/2.0
    return varu

def unstagger4d(var,axis):
    '''Unstagger 4D variable on given axis'''
    if axis == 1:
        varu = (var[:,:-1,:,:] + var[:,1:,:,:])/2.0
    if axis == 2:
        varu = (var[:,:,:-1,:] + var[:,:,1:,:])/2.0
    if axis == 3:
        varu = (var[:,:,:,:-1] + var[:,:,:,1:])/2.0
    return varu
