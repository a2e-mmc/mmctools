'''
  What it does: This dictionary contains functions for reading,
                manipulating, and probing WRF (or netCDF) files.

  Who made it: patrick.hawbecker@nrel.gov
  When: 5/11/18

  Notes:
  - Utility functions should automatically handle input data in either
    netCDF4.Dataset or xarray.Dataset formats.
  - TODO: as needed, replace these calls with appropriate calls to the
    wrf-python module

'''
from __future__ import print_function
import os, glob
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree
from scipy.interpolate import interp1d, LinearNDInterpolator
import netCDF4

from ..helper_functions import calc_wind


# List of default WRF fields for extract_column_from_wrfdata
default_3D_fields = ['U10','V10','T2','TSK','UST','PSFC','HFX','LH','MUU','MUV','MUT']
default_4D_fields = ['U','V','W','T',
                     'RU_TEND','RU_TEND_ADV','RU_TEND_PGF','RU_TEND_COR','RU_TEND_PHYS',
                     'RV_TEND','RV_TEND_ADV','RV_TEND_PGF','RV_TEND_COR','RV_TEND_PHYS',
                     'T_TEND_ADV',]

# Time-series output variables
# https://github.com/a2e-mmc/WRF/blob/master/run/README.tslist
ts_header = [
    'dom',    # grid ID
    'time',   # forecast time in hours
    'tsID',   # time series ID
    'locx',   # grid location x (nearest grid to the station)
    'locy',   # grid location y (nearest grid to the station)
    'T2',     # 2 m Temperature [K]
    'q2',     # 2 m vapor mixing ratio [kg/kg]
    'u10',    # 10 m U wind (earth-relative)
    'v10',    # 10 m V wind (earth-relative)
    'PSFC',   # surface pressure [Pa]
    'LWd',    # downward longwave radiation flux at the ground (downward is positive) [W/m^2]
    'SWd',    # net shortwave radiation flux at the ground (downward is positive) [W/m^2]
    'HFX',    # surface sensible heat flux (upward is positive) [W/m^2]
    'LFX',    # surface latent heat flux (upward is positive) [W/m^2]
    'TSK',    # skin temperature [K]
    'SLTtop', # top soil layer temperature [K]
    'RAINC',  # rainfall from a cumulus scheme [mm]
    'RAINNC', # rainfall from an explicit scheme [mm]
    'CLW',    # total column-integrated water vapor and cloud variables
    'QFX',    # Vapor flux (upward is positive) [g m^-2 s^-1]
    'UST',    # u* from M-O 
    'U4',     # 4 m U wind (earth-relative)
    'V4',     # 4 m V wind (earth-relative)
]

TH0 = 300.0 # [K] base-state potential temperature by WRF convention

def _get_dim(wrfdata,dimname):
    """Returns the specified dimension, with support for both netCDF4
    and xarray
    """
    if isinstance(wrfdata, netCDF4.Dataset):
        try:
            return wrfdata.dimensions[dimname].size
        except KeyError:
            print('No {:s} dimension'.format(dimname))
            return None
    elif isinstance(wrfdata, xr.Dataset):
        try:
            return wrfdata.dims[dimname]
        except KeyError:
            print('No {:s} dimension'.format(dimname))
            return None
    else:
        raise AttributeError('Unexpected WRF data type')

def _get_dim_names(wrfdata,dimname):
    """Returns dimension names of the specified variable,
    with support for both netCDF4 and xarray
    """
    if isinstance(wrfdata, netCDF4.Dataset):
        try:
            return wrfdata.variables[dimname].dimensions
        except KeyError:
            print('No {:s} dimension'.format(dimname))
            return None
    elif isinstance(wrfdata, xr.Dataset):
        try:
            return wrfdata.variables[dimname].dims
        except KeyError:
            print('No {:s} dimension'.format(dimname))
            return None
    else:
        raise AttributeError('Unexpected WRF data type')

def _get_var(wrfdata,varname):
    """Returns the specified variable, with support for both netCDF4
    and xarray
    """
    if isinstance(wrfdata, netCDF4.Dataset):
        try:
            return wrfdata.variables[varname][:]
        except KeyError:
            print('No variable {:s}'.format(varname))
            return None
    elif isinstance(wrfdata, xr.Dataset):
        try:
            return wrfdata.variables[varname].values
        except KeyError:
            print('No variable {:s}'.format(varname))
            return None
    else:
        raise AttributeError('Unexpected WRF data type')

def get_wrf_dims(wrfdata):
    ''' Find the dimensions of the given WRF file'''
    nx = _get_dim(wrfdata,'west_east')
    ny = _get_dim(wrfdata,'south_north')
    nz = _get_dim(wrfdata,'bottom_top')
    nt = _get_dim(wrfdata,'Time')
    return nt,nz,ny,nx

def get_height(wrfdata,timevarying=False,avgheight=False):
    '''
    Get heights for all [time,]height,latitude,longitude
    If `timevarying` is False, return height for first timestamp if
    `avgheight` is False, otherwise return the average height over all
    times.
    '''
    ph  = _get_var(wrfdata,'PH') # dimensions: (Time, bottom_top_stag, south_north, west_east)
    phb = _get_var(wrfdata,'PHB') # dimensions: (Time, bottom_top_stag, south_north, west_east)
    hgt = _get_var(wrfdata,'HGT') # dimensions: (Time, south_north, west_east)
    zs  = ((ph+phb)/9.81) - hgt[:,np.newaxis,:,:]
    z = unstagger(zs,axis=1)
    if timevarying:
        return z,zs
    elif avgheight:
        return np.mean(z,axis=0), np.mean(zs,axis=0)
    else:
        return z[0,...], zs[0,...]

def get_height_at_ind(wrfdata,j,i):
    '''Get model height at a specific j,i'''
    nt = _get_dim(wrfdata,'Time')
    nz = _get_dim(wrfdata,'bottom_top')
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

def get_unstaggered_var(wrfdata,varname):
    '''
    Extracts and unstaggers the specified variable
    '''
    var = _get_var(wrfdata,varname)
    if var is None:
        return None
    # Use dimension name to determine stagerred axis (staggered dimension name contain 'stag')
    stag_axs = [dim.endswith('_stag') for dim in _get_dim_names(wrfdata,varname)]
    assert(stag_axs.count(True) in [0,1]), \
        'Multiple staggered axis not supported (field ='+field+')'
    try:
        return unstagger(var,stag_axs.index(True))
    except ValueError:
        return var

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


# - - - - - - - - - - - - - - - - - - - - - - - #
#               TOWER UTILITIES                 #
# - - - - - - - - - - - - - - - - - - - - - - - #
def get_tower_header(header_line):
    header = {'longname' : header_line[:26].strip(),
              'domain'   : int(header_line[26:28]),
              'tsid'     : int(header_line[28:31]),
              'abbr'     : header_line[31:37].strip(),
              'lat'      : float(header_line[39:46]),
              'lon'      : float(header_line[47:55]),
              'loci'     : int(header_line[58:62]),
              'locj'     : int(header_line[63:67]),
              'gridlat'  : float(header_line[70:77]),
              'gridlon'  : float(header_line[78:86]),
              'stationz' : float(header_line[88:94])
             }
    return (header)


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
    if twr_file_name[-4:-1] == '.d0':
        twr_file_name = '{}.TS'.format(twr_file_name)
    twr = open(twr_file_name,'r')
    header_line = twr.readline()
    twr.close()
    header = get_tower_header(header_line)
    stni = int(header['loci']) - 1
    stnj = int(header['locj']) - 1
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
    '''

    standard_names = {
        'uu': 'u',
        'vv': 'v',
        'ww': 'w',
        'th': 'theta', # virtual potential temperature
    }

    def __init__(self,fstr,varlist=None):
        """The file-path string should be:
            '[path to towers]/[tower abrv.].d0[domain].*'
        """
        self.time = None
        self.nt = None
        self.nz = None
        self._getvars(fstr,requested_varns=varlist)
        self._getdata()

    def _getvars(self,fstr,requested_varns=None):
        if not fstr.endswith('*'): fstr += '*'
        if requested_varns is None:
            self.filelist = glob.glob(fstr)
            self.nvars = len(self.filelist)
            self.varns = [
                fpath.split('.')[-1] for fpath in self.filelist
            ]
        else:
            self.filelist = []
            self.varns = []
            # convert to uppercase per WRF convention
            requested_varns = [ varn.upper() for varn in requested_varns ]
            # check that requested var names were found in the path
            for varn in requested_varns:
                files = glob.glob(fstr+varn)
                if len(files) == 0:
                    print('requested variable',varn,'not available in file path')
                else:
                    assert (len(files)==1), \
                            'found multiple files for {:s}: {:s}'.format(varn,files)
                    self.varns.append(varn)
                    self.filelist.append(files[0])
        assert len(self.filelist) > 0, 'No TS output found in '+fstr

    def _getdata(self): # Get all the data
        for varn,fpath in zip(self.varns, self.filelist):
            # Get number of times,heights
            with open(fpath) as f:
                for nt,line in enumerate(f.readlines()): pass

            # Read profile data
            if varn != 'TS': # TS is different structure...
                nz = len(line.split()) - 1
                # Open again for reading
                with open(fpath) as f:
                    self.header = f.readline().split() # Header information
                    data = pd.read_csv(f,delim_whitespace=True,header=None).values
                var = data[:,1:]
                setattr(self, varn.lower(), var)
                if self.time is None:
                    self.time = data[:,0]
                else:
                    assert np.all(self.time == data[:,0]), 'tower data at different times'
                if self.nt is None:
                    self.nt = nt # Number of times
                else:
                    assert (self.nt == nt), 'tower data has different number of times'
                if self.nz is None:
                    self.nz = nz # Number of heights
                else:
                    assert (self.nz == nz), 'tower data has different number of heights'

            # Read surface variables (no height component)
            elif varn == 'TS':
                nv = len(line.split()) - 2
                with open(fpath) as f:
                    # Fortran formatted output creates problems when the
                    #   time-series id is 3 digits long and blends with the
                    #   domain number...
                    # FMT='(A26,I2,I3,A6,A2,F7.3,A1,F8.3,A3,I4,A1,I4,A3,F7.3,A1,F8.3,A2,F6.1,A7)')
                    # idx:  0   26 28 31 37,  39,46,  47,55,58,62,63,67,  70,77,  78,86,  88,94
                    header = f.readline()
                    self.longname = header[:26].strip()
                    self.domain   = int(header[26:28])
                    self.tsid     = int(header[28:31])
                    self.abbr     = header[31:37].strip()
                    self.lat      = float(header[39:46])
                    self.lon      = float(header[47:55])
                    self.loci     = int(header[58:62])
                    self.locj     = int(header[63:67])
                    self.gridlat  = float(header[70:77])
                    self.gridlon  = float(header[78:86])
                    self.stationz = float(header[88:94])
                    tsdata = pd.read_csv(f,delim_whitespace=True,header=None,names=ts_header)
                    tsdata = tsdata.drop(columns=['dom','time','tsID','locx','locy'])
                    for name,col in tsdata.iteritems(): 
                        setattr(self, name.lower(), col.values)
                    self.ts_varns = list(tsdata.columns)

    def _create_datadict(self,varns,unstagger=False,staggered_vars=['ph']):
        """Helper function for to_dataframe()"""
        datadict = {}
        for varn in varns:
            tsdata = getattr(self,varn)
            if unstagger:
                if varn in staggered_vars:
                    # need to destagger these quantities
                    datadict[varn] = ((tsdata[:,1:] + tsdata[:,:-1]) / 2).ravel()
                elif varn == 'th':
                    # theta is a special case
                    #assert np.all(tsdata[:,-1] == TH0), 'Unexpected nonzero value for theta'
                    # drop the trailing 0 for already unstaggered quantities
                    datadict[varn] = tsdata[:,:-1].ravel()
                else:
                    # other quantities already unstaggered
                    #if not varn == 'ww':
                    #    # don't throw a warning if w is already unstaggered by the code
                    #    assert np.all(tsdata[:,-1] == 0), 'Unexpected nonzero value for '+varn
                    # drop the trailing 0 for already unstaggered quantities
                    datadict[varn] = tsdata[:,:-1].ravel()
            else:
                # use data as is
                datadict[varn] = tsdata.ravel()
        return datadict

    def to_dataframe(self,start_time,
                     time_unit='h',time_step=None,
                     unstagger=True,
                     heights=None,height_var='height',agl=False,
                     exclude=['ts']):
        """Convert tower time-height data into a dataframe.

        Treatment of the time-varying height coordinates is summarized
        below:

        heights  height_var  resulting height index
        -------  ----------  ------------------------------------------
        None     n/a         levels (Int64Index)
        None     1d array    assumed constant in time (Float64Index)
        input    1d array    interpolate to input heights, assuming
                             constant heights in time (Float64Index)
        input    2d array    interpolate to input heights at all times
                             accounting for change in heights over time
                             (Float64Index)
        
        Note: TS profiles are output at staggered locations for _all_
        variables, regardless of whether they are staggered or not.
        For unstaggered (i.e., cell-centered) quantities, the last value
        will be 0. For consistency--if interpolation heights are not
        provided--all quantities will be unstaggered by default. If
        interpolation heights are provided, then both staggered and
        unstaggered data will be used for interpolation, and the output
        can be at arbitrary heights.

        Example usage:
        ```
        # output with Int64Index
        mytower = Tower('/path/to/prefix.d03.*')
        mytower.to_dataframe(start_time='2013-11-08 12:00')

        # output with approximately constant heights
        mytower = Tower('/path/to/prefix.d03.*')
        mytower.height = np.mean(mytower.ph, axis=0) # average over time
        mytower.height -= mytower.stationz  # make above ground level
        mytower.to_dataframe(start_time='2013-11-08 12:00')

        # interpolated output from data with approx constant heights
        myheights = np.arange(5,2000,10)
        mytower = Tower('/path/to/prefix.d03.*')
        mytower.height = np.mean(mytower.ph, axis=0) - mytower.stationz
        mytower.to_dataframe(start_time='2013-11-08 12:00',
                             heights=myheights)

        # interpolated output from data with time-varying heights,
        #   i.e., (geopotential height, 'ph'), at heights a.g.l.
        mytower = Tower('/path/to/prefix.d03.*')
        mytower.to_dataframe(start_time='2013-11-08 12:00',
                             heights=myheights, height_var='ph', agl=True)
        ```
        
        Parameters
        ----------
        start_time: str or pd.Timestamp
            The datetime index is constructed from a pd.TimedeltaIndex
            plus this start_time, where the timedelta index is formed by
            the saved time array.
        time_unit: str, optional
            Timedelta unit for constructing datetime index, only used if
            time_step is None.
        time_step: float or None, optional
            Time-step size, in seconds, to override the output times in
            the data files. Used in conjunction with start_time to form
            the datetime index. May be useful if times in output files
            do not have sufficient precision.
        unstagger: bool, optional
            Unstagger all variables so that all quantities are output at
            the correct height; only used if heights are not specified
        heights : array-like or None, optional
            If None, then use integer levels for the height index,
            otherwise interpolate to the same heights at all times.
        height_var : str, optional
            Name of attribute with actual height values to form the
            height index. If heights is None, then this must match the
            number of height levels; otherwise, this may be constant
            or variable in time.
        agl : bool, optional
            Heights by default are specified above sea level; if True,
            then the "stationz" attribute is used to convert to heights
            above ground level (AGL).  This only applies if heights are
            specified.
        exclude : list, optional
            List of fields to excldue from the output dataframe. By
            default, the surface time-series data ('ts') are excluded.
        """
        # convert varname list to lower case
        varns0 = [ varn.lower() for varn in self.varns ]
        # remove excluded vars
        varns = [ varn for varn in varns0 if not varn in exclude ]
        # setup index
        start_time = pd.to_datetime(start_time)
        if time_step is None:
            times = start_time + pd.to_timedelta(self.time, unit=time_unit)
            times = times.round(freq='1ms')
            times.name = 'datetime'
        else:
            timedelta = pd.to_timedelta(time_step, unit='s')
            # note: time is in hours and _single-precision_
            Nsteps0 = np.round(self.time[0] / (time_step/3600))
            toffset0 = Nsteps0 * timedelta
            times = pd.date_range(start=start_time+toffset0,
                                  freq=timedelta,
                                  periods=self.nt,
                                  name='datetime')
            times = times.round(freq='1ms')
        # combine (and interpolate) time-height data
        # - note 1: self.[varn].shape == self.height.shape == (self.nt, self.nz)
        # - note 2: arraydata.shape == (self.nt, len(varns)*self.nz)
        if heights is None:
            if unstagger:
                nz = self.nz - 1
            else:
                nz = self.nz
            datadict = self._create_datadict(varns,unstagger)
            if hasattr(self, height_var):
                # heights (constant in time) were separately calculated
                z = getattr(self, height_var)
                if unstagger:
                    z = (z[1:] + z[:-1]) / 2
                assert (len(z.shape) == 1) and (len(z) == nz), \
                        'tower '+height_var+' attribute should correspond to fixed height levels'
            else:
                # heights will be an integer index
                z = np.arange(nz)
            idx = pd.MultiIndex.from_product([times,z],names=['datetime','height'])
            df = pd.DataFrame(data=datadict,index=idx)
        else:
            from scipy.interpolate import interp1d
            z = np.array(heights) # interpolation heights
            zt_stag = getattr(self, height_var) # z(t)
            if agl:
                zt_stag -= self.stationz
            varns_unstag = varns.copy()
            varns_unstag.remove('ph')
            varns_stag = ['ph']
            if len(zt_stag.shape) == 1:
                # approximately constant height (with time)
                assert len(zt_stag) == self.nz
                zt_unstag = (zt_stag[1:] + zt_stag[:-1]) / 2
                df_unstag = pd.DataFrame(
                    data=self._create_datadict(varns_unstag,unstagger=True,staggered_vars=['ph']),
                    index=pd.MultiIndex.from_product([times,zt_unstag],
                                                     names=['datetime','height'])
                )
                df_stag = pd.DataFrame(
                    data=self._create_datadict(varns_stag),
                    index=pd.MultiIndex.from_product([times,zt_stag],
                                                     names=['datetime','height'])
                )
                # now unstack the times to get a height index
                def interp_to_heights(df):
                    unstacked = df.unstack(level=0)
                    interpfun = interp1d(unstacked.index, unstacked.values, axis=0,
                                         bounds_error=False,
                                         fill_value='extrapolate')
                    interpdata = interpfun(z)
                    unstacked = pd.DataFrame(data=interpdata,
                                             index=z, columns=unstacked.columns)
                    unstacked.index.name = 'height'
                    df = unstacked.stack()
                    return df.reorder_levels(order=['datetime','height']).sort_index()
                df_unstag = interp_to_heights(df_unstag)
                df_stag = interp_to_heights(df_stag)
                df = pd.concat((df_stag,df_unstag), axis=1)
            else:
                # interpolate for all times
                assert zt_stag.shape == (self.nt, self.nz), \
                        'heights should correspond to time-height indices'
                datadict = {}
                zt_unstag = (zt_stag[:,1:] + zt_stag[:,:-1]) / 2
                for varn in varns_unstag:
                    newdata = np.empty((self.nt, len(z)))
                    tsdata = getattr(self,varn)
                    #if varn == 'th':
                    #    # theta is a special case
                    #    assert np.all(tsdata[:,-1] == TH0)
                    #elif not varn == 'ww':
                    #    # if w has already been destaggered by wrf
                    #    assert np.all(tsdata[:,-1] == 0)
                    for itime in range(self.nt):
                        interpfun = interp1d(zt_unstag[itime,:],
                                             tsdata[itime,:-1],
                                             bounds_error=False,
                                             fill_value='extrapolate')
                        newdata[itime,:] = interpfun(z)
                    datadict[varn] = newdata.ravel()
                for varn in varns_stag:
                    newdata = np.empty((self.nt, len(z)))
                    tsdata = getattr(self,varn)
                    for itime in range(self.nt):
                        interpfun = interp1d(zt_stag[itime,:],
                                             tsdata[itime,:],
                                             bounds_error=False,
                                             fill_value='extrapolate')
                        newdata[itime,:] = interpfun(z)
                    datadict[varn] = newdata.ravel()
                idx = pd.MultiIndex.from_product([times,z], names=['datetime','height'])
                df = pd.DataFrame(data=datadict,index=idx)

        # standardize names
        df.rename(columns=self.standard_names, inplace=True)
        return df

    def to_xarray(self,start_time,
                  time_unit='h',time_step=None,
                  heights=None,height_var='height',agl=False,
                  structure='ordered',
                  **kwargs):
        """Convert tower time-height data into a xarray dataset.
        
        Treatment of the time-varying height coordinates is summarized
        below:

        heights  height_var  resulting height index
        -------  ----------  ------------------------------------------
        None     n/a         levels (Int64Index)
        None     1d array    assumed constant in time (Float64Index)
        input    1d array    interpolate to input heights, assuming
                             constant heights in time (Float64Index)
        input    2d array    interpolate to input heights at all times
                             accounting for change in heights over time
                             (Float64Index)
        
        Parameters
        ----------
        start_time: str or pd.Timestamp
            The datetime index is constructed from a pd.TimedeltaIndex
            plus this start_time, where the timedelta index is formed by
            the saved time array.
        time_unit: str
            Timedelta unit for constructing datetime index, only used if
            time_step is None.
        time_step: float or None
            Time-step size, in seconds, to override the output times in
            the data files. Used in conjunction with start_time to form
            the datetime index. May be useful if times in output files
            do not have sufficient precision.
        heights : array-like or None
            If None, then use integer levels for the height index,
            otherwise interpolate to the same heights at all times.
        height_var : str
            Name of attribute with actual height values to form the
            height index. If heights is None, then this must match the
            number of height levels; otherwise, this may be constant
            or variable in time.
        agl : bool, optional
            Heights by default are specified above sea level; if True,
            then the "stationz" attribute is used to convert to heights
            above ground level (AGL).  This only applies if heights are
            specified.
        """
        df = self.to_dataframe(start_time,
                time_unit=time_unit, time_step=time_step,
                heights=heights, height_var=height_var, agl=agl,
                **kwargs)
        ds = df.to_xarray()

        # update height dimension
        if heights is None:
            # no interpolation, heights are indicies
            ds = ds.rename_dims({'height':'k'})
            ds = ds.rename_vars({'height':'k'})
        else:
            # interpolation performed, drop height_var
            ds = ds.drop_vars([height_var])

        # update coords and dims
        if structure == 'ordered':
            ds = ds.assign_coords(i=self.loci, j=self.locj)
            ds = ds.expand_dims(['j','i'],axis=[2,3])
            # Add station coordinates as data variables:
            ds['lat'] = (['j','i'],  [[self.gridlat]])
            ds['lon'] = (['j','i'],  [[self.gridlon]])
            ds['zsurface'] = (['j','i'],  [[self.stationz]])
            # Add ts data (no k/height dim)
            for varn in self.ts_varns:
                varn = varn.lower()
                tsvar = getattr(self, varn)
                ds[varn] = (['datetime','j','i'], tsvar[:,np.newaxis,np.newaxis])
        elif structure == 'unordered':
            ds = ds.assign_coords(station=self.abbr)
            ds = ds.expand_dims(['station'],axis=[2])
            # Add station coordinates as data variables:
            ds['i'] = (['station'],  [self.loci])
            ds['j'] = (['station'],  [self.locj])
            ds['lat'] = (['station'],  [self.gridlat])
            ds['lon'] = (['station'],  [self.gridlon])
            ds['zsurface'] = (['station'],  [self.stationz])
            # Add ts data (no k/height dim)
            for varn in self.ts_varns:
                varn = varn.lower()
                tsvar = getattr(self, varn)
                ds[varn] = (['datetime','station'], tsvar[:,np.newaxis])

        return ds

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
    if hasattr(timestrs[0],'values'):
        # xarray
        return [ datetime.strptime(s.values.tostring().decode(), format) for s in timestrs ]
    else:
        # netcdf
        return [ datetime.strptime(s.tostring().decode(), format) for s in timestrs ]

def latlon_to_ij(wrfdata,lat,lon):
    '''Get i,j location from given wrf file and lat/long'''
    glat,glon = latlon(wrfdata)
    dist = ((glat-lat)**2 + (glon-lon)**2)**0.5
    jj,ii = np.where(dist==np.min(dist))
    return ii[0],jj[0]

def unstagger(var,axis):
    '''Unstagger ND variable on given axis'''
    # left_indx: (:,:,etc.) and replace ":" at ax with ":-1"
    left_indx = [slice(0,-1) if axi==axis else slice(None) for axi in range(var.ndim)]
    # right_indx: (:,:,etc.) and replace ":" at ax with "1:"
    right_indx = [slice(1,None) if axi==axis else slice(None) for axi in range(var.ndim)]
    return (var[tuple(left_indx)] + var[tuple(right_indx)])/2.0

def add_surface_plane(var,plane=None):
    tdim,zdim,ydim,xdim = var.shape
    if plane is None:
        plane = np.zeros((tdim,1,ydim,xdim))
        return np.concatenate((plane,var), axis = 1)
    else:
        return np.concatenate((np.reshape(plane,(tdim,1,ydim,xdim)),var), axis = 1)

def extract_column_from_wrfdata(fpath, coords,
                                Ztop=2000., Vres=5.0,
                                T0=TH0,
                                spatial_filter='interpolate',L_filter=0.0,
                                additional_fields=[],
                                verbose=False,
                                **kwargs):
    """
    Extract a column of time-height data for a specific site from a
    4-dimensional WRF output file

    The site specific data is obtained by applying a spatial filter to
    the WRF data and subsequently interpolating to an equidistant set 
    of vertical levels representing a microscale vertical grid.
    The following spatial filtering types are supported:
    - 'interpolate': interpolate to site coordinates
    - 'nearest':     use nearest WRF grid point
    - 'average':     average over an area with size L_filter x L_filter,
                     centered around the site

    Usage
    ====
    coords : list or tuple of length 2
        Latitude and longitude of the site for which to extract data
    Ztop : float
        Top of the microscale grid [m]
    Vres : float
        Vertical grid resolution of the microscale grid [m]
    T0 : float
        Reference temperature for WRF perturbation temperature [K]
    spatial_filter : 'interpolate', 'nearest' or 'average'
        Type of spatial filtering
    L_filter : float
        Length scale for spatial averaging [m]
    additional_fields : list
        Additional fields to be processed
    """
    import utm
    assert(spatial_filter in ['nearest','interpolate','average']),\
            'Spatial filtering type "'+spatial_filter+'" not recognised'

    # Load WRF data
    ds = xr.open_dataset(fpath,**kwargs)
    tdim, zdim, ydim, xdim = get_wrf_dims(ds)
    
    
    #---------------------------
    # Preprocessing
    #---------------------------
    
    # Extract WRF grid resolution
    dx_meso = ds.attrs['DX']
    assert(dx_meso == ds.attrs['DY'])
    
    
    # Number of additional points besides nearest grid point to perform spatial filtering
    if spatial_filter == 'interpolate':
        Nadd = 1
    elif spatial_filter == 'average':
        Nadd = int(np.ceil(0.5*L_filter/dx_meso+1.0e-6)) # +eps to make sure Nadd*dxmeso > L_filter/2
    else:
        Nadd = 0
        
    
    # Setup microscale grid data
    site_X, site_Y, site_zonenumber, _ = utm.from_latlon(coords[0],coords[1])

    if spatial_filter == 'average':
        Navg = 6 #Number of interpolation points to compute average
        xmicro = np.linspace(-L_filter/2,L_filter/2,Navg,endpoint=True)+site_X
        ymicro = np.linspace(-L_filter/2,L_filter/2,Navg,endpoint=True)+site_Y
    else: # 'interpolate' or 'nearest'
        xmicro = site_X
        ymicro = site_Y
    
    zmicro = np.linspace(0,Ztop,1+int(Ztop/Vres))
    Zmicro, Ymicro, Xmicro = np.meshgrid(zmicro, ymicro, xmicro, indexing='ij')

    #2D and 3D list of points
    XYmicro = np.array((Ymicro[0,:,:].ravel(),Xmicro[0,:,:].ravel())).T
    XYZmicro = np.array((Zmicro.ravel(),Ymicro.ravel(),Xmicro.ravel())).T
    

    # Check whether additional fields are 3D or 4D and append to corresnponding list of fields
    fieldnames_3D = default_3D_fields
    fieldnames_4D = default_4D_fields
    for field in additional_fields:
        try:
            ndim = len(ds[field].dims)
        except KeyError:
            print('The additional field "'+field+'" is not available and will be ignored.')
        else:
            if len(ds[field].dims) == 3:
                if field not in fieldnames_3D:
                    fieldnames_3D.append(field)
            elif len(ds[field].dims) == 4:
                if field not in fieldnames_4D:
                    fieldnames_4D.append(field)
            else:
                raise Exception('Field "'+field+'" is not 3D or 4D, not sure how to process this field.')
    
    
    #---------------------------
    # Load data
    #---------------------------
    
    WRFdata = {}
    
    # Cell-centered coordinates
    XLAT = ds.variables['XLAT'].values     # WRF indexing XLAT[time,lat,lon]
    XLONG = ds.variables['XLONG'].values
    
    # Height above ground level
    WRFdata['Zagl'],_ = get_height(ds,timevarying=True)
    WRFdata['Zagl']   = add_surface_plane(WRFdata['Zagl'])
    
    # 3D fields. WRF indexing Z[tdim,ydim,xdim]
    for field in fieldnames_3D:
        WRFdata[field] = get_unstaggered_var(ds,field)
    
    # 4D fields. WRF indexing Z[tdim,zdim,ydim,xdim]
    for field in fieldnames_4D:
        WRFdata[field] = get_unstaggered_var(ds,field)
        if WRFdata[field] is None:
            continue
            
        # 4D field specific processing
        if field == 'T':
            # Add T0, set surface plane to TSK
            WRFdata[field] += T0
            WRFdata[field] = add_surface_plane(WRFdata[field],plane=WRFdata['TSK'])
        elif field in ['U','V','W']:
            # Set surface plane to zero (no slip)
            WRFdata[field] = add_surface_plane(WRFdata[field])
        else:
            WRFdata[field] = add_surface_plane(WRFdata[field],plane=WRFdata[field][:,0,:,:])

    # clean up empty fields
    for name in list(WRFdata.keys()):
        if WRFdata[name] is None:
            del WRFdata[name]
            try:
                fieldnames_3D.remove(name)
            except ValueError: pass
            try:
                fieldnames_4D.remove(name)
            except ValueError: pass
    
    
    #---------------------------
    # Extract site data
    #---------------------------
    sitedata = {}
    sitedata['Zagl'] = zmicro
    
    # Nearest grid points to the site
    points = np.array((XLAT[0,:,:].ravel(),XLONG[0,:,:].ravel())).T
    tree = KDTree(points)
    dist, index = tree.query(np.array(coords),1) # nearest grid point
    inear = int( index % xdim )       # index to XLONG
    jnear = int((index-inear) / xdim) # index to XLAT
    
    
    # Extract data and apply spatial filter if necessary
    if spatial_filter == 'nearest':
        # - 3D fields
        for field in fieldnames_3D:
            sitedata[field] = WRFdata[field][:,jnear,inear]
               
        # - 4D fields
        for field in fieldnames_4D:
            sitedata[field] = np.zeros((tdim,zmicro.size))
        
        # Interpolate to microscale z grid at every time
        for t in range(tdim):
            Zmeso = WRFdata['Zagl'][t,:,jnear,inear].squeeze()
            wrf_data_combined  = np.array([WRFdata[field][t,:,jnear,inear].squeeze() for field in fieldnames_4D]).T
            site_data_combined = interp1d(Zmeso,wrf_data_combined,axis=0)(zmicro)
            for l, field in enumerate(fieldnames_4D):
                sitedata[field][t,:] = site_data_combined[:,l]
            
            
    else: # 'interpolate' or 'average'
        # Coordinates of a subset of the WRF grid in UTM-projected cartesian system
        NN = 1+2*Nadd
        Xmeso = np.zeros((NN,NN))
        Ymeso = np.zeros((NN,NN))
        for i,ii in enumerate(range(inear-Nadd,inear+Nadd+1)):
            for j,jj in enumerate(range(jnear-Nadd,jnear+Nadd+1)):
                Xmeso[j,i], Ymeso[j,i], _, _ = utm.from_latlon(XLAT[0,jj,ii],
                                                               XLONG[0,jj,ii],
                                                               force_zone_number = site_zonenumber)
                
        Xmeso = np.repeat(Xmeso[np.newaxis, :, :], zdim+1, axis=0)
        Ymeso = np.repeat(Ymeso[np.newaxis, :, :], zdim+1, axis=0)
        XYmeso = np.array((np.ravel(Ymeso[0,:,:]),np.ravel(Xmeso[0,:,:]))).T
        
        
        #Initialize fields
        for field in fieldnames_3D: sitedata[field] = np.zeros((tdim))
        for field in fieldnames_4D: sitedata[field] = np.zeros((tdim,zmicro.size))
            
        #Perform 3D interpolation to microscale grid for every time
        for t in range(tdim):
            # 3D fields
            slice3d = (t,slice(jnear-Nadd,jnear+Nadd+1),slice(inear-Nadd,inear+Nadd+1))
            wrf_data_combined  = np.array([WRFdata[field][slice3d].ravel() for field in fieldnames_3D]).T
            site_data_combined = LinearNDInterpolator(XYmeso,wrf_data_combined)(XYmicro)
            for l, field in enumerate(fieldnames_3D):
                if spatial_filter == 'interpolate':
                    sitedata[field][t] = site_data_combined[0,l]
                elif spatial_filter == 'average':
                    sitedata[field][t] = np.mean(site_data_combined[:,l])
            
            
            # 4D fields
            slice4d = (t,range(zdim+1),slice(jnear-Nadd,jnear+Nadd+1),slice(inear-Nadd,inear+Nadd+1))
            Zmeso = WRFdata['Zagl'][slice4d]
            XYZmeso = np.array((Zmeso.ravel(),Ymeso.ravel(),Xmeso.ravel())).T
            
            wrf_data_combined  = np.array([WRFdata[field][slice4d].ravel() for field in fieldnames_4D]).T
            site_data_combined = LinearNDInterpolator(XYZmeso,wrf_data_combined)(XYZmicro)
            for l, field in enumerate(fieldnames_4D):
                if spatial_filter == 'interpolate':
                    sitedata[field][t,:] = site_data_combined[:,l]
                elif spatial_filter == 'average':
                    sitedata[field][t,:] = np.mean(site_data_combined[:,l].reshape(Zmicro.shape), axis=(1,2))

                    
    #---------------------------
    # Store data in xarray
    #---------------------------
    
    coords = {'Time': ds['XTIME'].values,
              'height': zmicro}
    
    data_vars = {}
    for field in fieldnames_3D:
        data_vars[field] = ('Time',
                            sitedata[field],
                            {'description': ds[field].attrs['description'].lower(),
                             'units': ds[field].attrs['units']}
                           )
    for field in fieldnames_4D:
        data_vars[field] = (['Time','height'],
                            sitedata[field],
                            {'description': ds[field].attrs['description'].lower(),
                             'units': ds[field].attrs['units']}
                           )
        
    
    xn = xr.Dataset(data_vars=data_vars,coords=coords)
    
    # Rename T to theta and adjust description
    xn = xn.rename({'T':'theta'})
    xn['theta'].attrs['description'] = 'potential temperature'

    return xn


def combine_towers(fdir, restarts, simulation_start, fname,
                   structure='ordered', time_step=None,
                   dx=12.0, dy=12.0,
                   heights=None, height_var='heights', agl=False,
                   verbose=True, **kwargs):
    '''
    Combine together tslist files in time where, if there is any overlap, the later file
    will overwrite the earlier file. This makes the assumption that all of the tslist 
    files are stored in separate directories but named the same (default in WRF is to 
    name them the same). Each restart directory must have a simulation_start string to
    specify when the timing starts (use same time if run was an actual restart, use WRF
    start time if you are combining several runs).

    fdir             = 'path/to/restart/directories/'
    restarts         = ['restart_dir_1', 'restart_dir_2', 'restart_dir_3']
                       or None (for single run with output in fdir)
    simulation_start = ['2000-01-01 00:00','2000-01-01 00:00','2000-01-01 00:00']
                       or '2000-01-01 00:00' for a single run
    fname            = ['t0001.d02'] (Note: this is the prefix for the tower + domain)
    structure        = 'ordered' or 'unordered'
    '''
    if not isinstance(simulation_start,(list,tuple)):
        simulation_start = [simulation_start]
    if restarts is None:
        restarts = ['.']
    assert len(simulation_start) == len(restarts), 'restarts and simulation_start are not equal'
    for rst,restart in enumerate(restarts):
        if verbose:
            print('restart: {}'.format(restart))
        data = []
        for ff in fname:
            if verbose:
                print('starting {}'.format(ff))
            fpath = os.path.join(fdir,restart,ff)
            tow = Tower(fpath)
            ds = tow.to_xarray(start_time=simulation_start[rst],
                               time_step=time_step,
                               structure=structure,
                               heights=heights,
                               height_var=height_var,
                               agl=agl,
                               **kwargs)
            data.append(ds)
        data_block = xr.combine_by_coords(data)
        if np.shape(restarts)[0] > 1:
            if rst == 0:
                data_previous = data_block
            else:
                dataF = data_previous.combine_first(data_block)
                data_previous = dataF
        else:
            dataF = data_block
    if heights is None:
        height_dim = 'k'
        height_var = 'ph'
    else:
        height_dim = 'height'
        height_var = 'height'
    if structure == 'ordered':
    # -------------------------------------------------------       
    #               MMC Format specifications
        dataF = dataF.rename_dims({height_dim:'nz',
                                   'i':'nx',
                                   'j':'ny'})
        xcoord,ycoord = np.meshgrid(dataF.i*dx,dataF.j*dy)
        dataF = dataF.assign_coords(x=(('ny','nx'),xcoord))
        dataF = dataF.assign_coords(y=(('ny','nx'),ycoord))
        dataF = dataF.assign_coords(z=dataF[height_var])
        dataF = dataF.reset_index(['i','j'],drop=True).drop(height_var)
        dataF.attrs['DX'] = dx
        dataF.attrs['DY'] = dy
    elif structure == 'unordered':
        dataF = dataF.rename_dims({height_dim:'nz'})
    dataF = dataF.assign_coords(lat=dataF.lat)
    dataF = dataF.assign_coords(lon=dataF.lon)
    dataF = dataF.assign_coords(zsurface=dataF.zsurface)

    dataF['wspd'],dataF['wdir'] = calc_wind(dataF)

    dataF.attrs['CREATED_FROM'] = fdir

    return dataF


def tsout_seriesReader(fdir, 
                       restarts=[''], 
                       simulation_start_time=None, 
                       domain_of_interest=None,
                       structure='ordered', 
                       time_step=None,
                       heights=None, 
                       height_var='heights', 
                       select_tower=None,
                       **kwargs):
    '''
    This will combine a series of tslist output over time and location based on the
    path to the case (fdir), the restart directories (restarts), a model start time 
    (simulation_start_time), and the domain of interest for the towers
    (domain_of_interest). You can select individual towers or a set of towers by
    specifying a list or array in select_towers. Tower levels can be interpolated
    to specified heights by specifying 'heights' and 'height_var' where height_var
    is the variable that contains height values.

    fdir                  = 'path/to/restart/directories/'
    restarts              = ['tsout_1800_1830','tsout_1830_1900','tsout_1900_1930','tsout_1930_2000']
    simulation_start_time = '2013-11-08 14:00'
    domain_of_interest    = 'd02'
    time_step             = 10.0 
    heights               = [20.0, 50.0, 100.0]
    height_var            = 'ph'
    select_tower          = ['TS1','TS5']
    '''
    if simulation_start_time is None:
        raise ValueError ('simulation_start_time must be in the format of "YYYY-MM-DD HH:MM:SS"')
    if type(restarts) is str:
        restarts = [restarts]
    if domain_of_interest is None:
        raise ValueError('Must specify domain_of_interest as str (e.g., "d02")')
    if type(simulation_start_time) is str:
        simulation_start_time = [simulation_start_time]*len(restarts)
    ntimes = np.shape(restarts)[0]
    floc = '{}{}/*{}.??'.format(fdir,restarts[0],domain_of_interest)
    file_list = glob.glob(floc)
    assert file_list != [], 'No tslist files found in {}. Check kwargs.'.format(floc)
    for ff,file in enumerate(file_list):
        file = file[:-3]
        file_list[ff] = file
    
    for f in file_list: 
        if 'geo_em' in f: file_list.remove(f)

    file_list = np.unique(file_list)
    tower_names = file_list.copy()
    for ff,file in enumerate(file_list):
        tower_names[ff] = file.split('/')[-1]
        
    if not isinstance(select_tower,(list)) and select_tower is not None:
        select_tower = list(select_tower)
            
    if select_tower != None:
        good_towers = []
        for twr in select_tower:
            for twr_n in tower_names:
                if twr in twr_n: good_towers.append(twr_n)
        tower_names = good_towers
    dsF = combine_towers(fdir,restarts,simulation_start_time,tower_names,
                         structure=structure, time_step=time_step,
                         heights=heights, height_var=height_var,
                         **kwargs)
    return dsF


def wrfout_seriesReader(wrf_path,wrf_file_filter,
                        specified_heights=None,agl=False,
                        irange=None,jrange=None,hlim_ind=None,
                        temp_var='THM',extra_vars=[],
                        use_dimension_coords=False):
    """
    Construct an a2e-mmc standard, xarrays-based, data structure from a
    series of 3-dimensional WRF output files

    Note: Base state theta= 300.0 K is assumed by convention in WRF,
        this function follow this convention.

    Usage
    ====
    wrfpath : string 
        The path to directory containing wrfout files to be processed
    wrf_file_filter : string-glob expression
        A string-glob expression to filter a set of 4-dimensional WRF
        output files.
    specified_heights : list-like, optional	
        If not None, then a list of static heights to which all data
        variables should be interpolated. Note that this significantly
        increases the data read time.
    agl : bool, optional
        If True, then specified heights are expected to be above ground
        level (AGL) and the interpolation heights will have the local
        elevation subtracted out.
    irange,jrange : tuple, optional
        If not none, then the DataArray ds_subset is further subset in
        the horizontal dimensions, which should speed up execution. The
        tuple should be (idxmin, idxmax), inclusive and 1-based indices
        (as in WRF).
    hlim_ind : int, index, optional
        If not none, then the DataArray ds_subset is further subset by
        vertical dimension, keeping vertical layers 0:hlim_ind. This is
        meant to be used to speed up execution of the code or prevent a
        memory error where the specified_heights argument is not well
        suited (i.e., want a range of non-interpolated heights), and you
        only care about data that are below a certain vertical index.
    extra_vars : list, optional
        List of additional fields to output
    temp_var : str, optional
        Name of moist potential temperature variable, e.g., 'THM' for
        standard WRF output or 'T' MMC auxiliary output
    use_dimension_coords : bool, optional
        If True, then x and y coordinates will match the corresponding
        dimension to facilitate and expedite xarray operations
    """
    import wrf as wrfpy
    dims_dict = {
        'Time':'datetime',
        'bottom_top':'nz',
    }
    if use_dimension_coords:
        dims_dict['west_east'] = 'x'
        dims_dict['south_north'] = 'y'
    else:
        dims_dict['west_east'] = 'nx'
        dims_dict['south_north'] = 'ny'

    ds = xr.open_mfdataset(os.path.join(wrf_path,wrf_file_filter),
                           chunks={'Time': 10},
                           combine='nested',
                           concat_dim='Time')
    dim_keys = ["Time","bottom_top","south_north","west_east"] 
    horiz_dim_keys = ["south_north","west_east"]
    print('Finished opening/concatenating datasets...')

    ds_subset = ds[['XTIME']]
    print('Establishing coordinate variables, x,y,z, zSurface...')
    ds_subset['z'] = wrfpy.destagger((ds['PHB'] + ds['PH']) / 9.8,
                                     stagger_dim=1, meta=True)
    #ycoord = ds.DY * np.tile(0.5 + np.arange(ds.dims['south_north']),
    #                         (ds.dims['west_east'],1))
    #xcoord = ds.DX * np.tile(0.5 + np.arange(ds.dims['west_east']),
    #                         (ds.dims['south_north'],1)) 
    ycoord = ds.DY * (0.5 + np.arange(ds.dims['south_north']))
    xcoord = ds.DX * (0.5 + np.arange(ds.dims['west_east']))
    #ds_subset['y'] = xr.DataArray(np.transpose(ycoord), dims=horiz_dim_keys)
    #ds_subset['x'] = xr.DataArray(xcoord, dims=horiz_dim_keys)
    ds_subset['y'] = xr.DataArray(ycoord, dims='south_north')
    ds_subset['x'] = xr.DataArray(xcoord, dims='west_east')

    # Assume terrain height is static in time even though WRF allows
    # for it to be time-varying for moving grids
    ds_subset['zsurface'] = xr.DataArray(ds['HGT'].isel(Time=0), dims=horiz_dim_keys)
    print('Destaggering data variables, u,v,w...')
    ds_subset['u'] = wrfpy.destagger(ds['U'], stagger_dim=3, meta=True)
    ds_subset['v'] = wrfpy.destagger(ds['V'], stagger_dim=2, meta=True)
    ds_subset['w'] = wrfpy.destagger(ds['W'], stagger_dim=1, meta=True)

    print('Extracting data variables, p,theta...')
    ds_subset['p'] = xr.DataArray(ds['P']+ds['PB'], dims=dim_keys)
    ds_subset['theta'] = xr.DataArray(ds[temp_var]+TH0, dims=dim_keys)

    # extract additional variables if requested
    for var in extra_vars:
        if var not in ds.data_vars:
            print(f'Requested variable "{var}" not in {str(list(ds.data_vars))}')
            continue
        field = ds[var]
        print(f'Extracting {var}...')
        for idim, dim in enumerate(field.dims):
            if dim.endswith('_stag'):
                print(f'  destaggering {var} in dim {dim}...')
                field = wrfpy.destagger(field, stagger_dim=idim, meta=True)
        ds_subset[var] = field

    # subset in horizontal dimensions
    # note: specified ranges are WRF indices, i.e., python indices +1
    if irange is not None:
        assert isinstance(irange, tuple), 'irange should be (imin,imax)'
        ds_subset = ds_subset.isel(west_east=slice(irange[0]-1, irange[1]))
    if jrange is not None:
        assert isinstance(jrange, tuple), 'jrange should be (jmin,jmax)'
        ds_subset = ds_subset.isel(south_north=slice(jrange[0]-1, jrange[1]))

    # clip vertical extent if requested
    if hlim_ind is not None:
        ds_subset = ds_subset.isel(bottom_top=slice(0, hlim_ind))

    # optionally, interpolate to static heights	
    if specified_heights is not None:	
        zarr = ds_subset['z']	
        if agl:
            zarr -= ds_subset['zsurface']
        for var in ds_subset.data_vars:
            if (var == 'z') or ('bottom_top' not in ds_subset[var].dims):
                continue
            print('Interpolating',var)	
            ds_subset[var] = wrfpy.interplevel(ds_subset[var], zarr, specified_heights)	
            if np.any(~np.isfinite(ds_subset[var])):
                print('WARNING: wrf.interplevel() produced NaNs -- make sure requested heights are in range and/or use agl=True')
        ds_subset = ds_subset.drop_dims('bottom_top').rename({'level':'z'})	
        dim_keys[1] = 'z'	
        dims_dict.pop('bottom_top')
        print(dims_dict)

    # calculate derived variables
    print('Calculating derived data variables, wspd, wdir...')
    ds_subset['wspd'] = xr.DataArray(np.sqrt(ds_subset['u']**2 + ds_subset['v']**2),
                                     dims=dim_keys)
    ds_subset['wdir'] = xr.DataArray(180. + np.arctan2(ds_subset['u'],ds_subset['v'])*180./np.pi,
                                     dims=dim_keys)
    
    # rename coord variable for time and assign ccordinates 
    ds_subset = ds_subset.rename({'XTIME': 'datetime'})  #Rename after defining the component DataArrays in the DataSet
    if specified_heights is None:
        ds_subset = ds_subset.assign_coords(z=ds_subset['z'])
    ds_subset = ds_subset.assign_coords(x=ds_subset['x'],
                                        y=ds_subset['y'],
                                        zsurface=ds_subset['zsurface'])
    ds_subset = ds_subset.rename_vars({'XLAT':'lat', 'XLONG':'lon'})
    for olddim,newdim in dims_dict.copy().items():
        if newdim in ds_subset.coords:
            # have to swap dim instead of renaming if it already exists
            ds_subset = ds_subset.swap_dims({olddim: newdim})
            dims_dict.pop(olddim)
    ds_subset = ds_subset.rename_dims(dims_dict)
    
    return ds_subset


def wrfout_slices_seriesReader(wrf_path, wrf_file_filter,
                               specified_heights=None,
                               do_slice_vars=True,
                               do_surf_vars=False,
                               vlist=None):
    """
    Construct an a2e-mmc standard, xarrays-based, data structure from a
    series of WRF slice output files

    Note: Base state theta= 300.0 K is assumed by convention in WRF,
          and this function follows this convention.                                                                                     
    Usage
    ====
    wrfpath : string
        The path to directory containing wrfout files to be processed
    wrf_file_filter : string-glob expression
        A string-glob expression to filter a set of 4-dimensional WRF
        output files.
    specified_heights : list-like, optional
        If not None, then a list of static heights to which all data
        variables should be interpolated. Note that this significantly
        increases the data read time.
    do_slice_vars: Logical (default True), optional
       If true, then the slice variables (SLICES_U, SLICES_V, SLICES_W,
       SLICES_T) are read for the specified height (or for all heights
       if 'specified_heights = None')
    do_surf_vars: Logical (default False), optional
       If true, then the surface variables (UST, HFX, QFX, SST, SSTK)
       will be added to the file
    vlist: List-like, default None (optional)
       If not none, then set do_slice_vars and do_surf_vars to False,
       and only variables in the list 'vlist' are read
    """
    dims_dict = {
        'Time':'datetime',
        'num_slices':'nz_slice',
        'south_north': 'ny',
        'west_east':'nx',
    }

    ds = xr.open_mfdataset(os.path.join(wrf_path,wrf_file_filter),
                           chunks={'Time': 10},
                           combine='nested',
                           concat_dim='Time')

    ds = ds.assign_coords({"SLICES_Z": ds.SLICES_Z.isel(Time=1)})
    ds = ds.swap_dims({'num_slices': 'SLICES_Z'})

    dim_keys = ["Time","bottom_top","south_north","west_east"]
    horiz_dim_keys = ["south_north","west_east"]
    print('Finished opening/concatenating datasets...')
    #print(ds.dims)                                                                                                               
    ds_subset = ds[['Time']]
    print('Establishing coordinate variables, x,y,z, zSurface...')
    ycoord = ds.DY * (0.5 + np.arange(ds.dims['south_north']))
    xcoord = ds.DX * (0.5 + np.arange(ds.dims['west_east']))
    ds_subset['z'] = xr.DataArray(specified_heights, dims='num_slices')

    ds_subset['y'] = xr.DataArray(ycoord, dims='south_north')
    ds_subset['x'] = xr.DataArray(xcoord, dims='west_east')

    if vlist is not None:
        print("vlist not None, setting do_slice_vars and do_surf_vars to False")
        print("Does not support specified_heights argument, grabing all available heights")
        do_slice_vars = False
        do_surf_vars = False
        print("Extracting variables")
        for vv in vlist:
            print(vv)
            ds_subset[vv] = ds[vv]

    if do_slice_vars:
        print("Doing slice variables")
        print('Grabbing u, v, w, T')
        if specified_heights is not None:
            if len(specified_heights) == 1:
                print("One height")
                #print(ds.dims)
                #print(ds.coords)
                ds_subset['u'] = ds['SLICES_U'].sel(SLICES_Z=specified_heights)
                ds_subset['v'] = ds['SLICES_V'].sel(SLICES_Z=specified_heights)
                ds_subset['w'] = ds['SLICES_W'].sel(SLICES_Z=specified_heights)
                ds_subset['T'] = ds['SLICES_T'].sel(SLICES_Z=specified_heights)
            else:
                print("Multiple heights")
                ds_subset['u'] = ds['SLICES_U'].sel(SLICES_Z=specified_heights)
                ds_subset['v'] = ds['SLICES_V'].sel(SLICES_Z=specified_heights)
                ds_subset['w'] = ds['SLICES_W'].sel(SLICES_Z=specified_heights)
                ds_subset['T'] = ds['SLICES_T'].sel(SLICES_Z=specified_heights)
        else:
            ds_subset['u'] = ds['SLICES_U']
            ds_subset['v'] = ds['SLICES_V']
            ds_subset['w'] = ds['SLICES_W']
            ds_subset['T'] = ds['SLICES_T']

        print('Calculating derived data variables, wspd, wdir...')
        #print((ds_subset['u'].ufuncs.square()).values)
        ds_subset['wspd'] = xr.DataArray(
                np.sqrt(ds_subset['u'].values**2 + ds_subset['v'].values**2),
                dims=dim_keys)
        ds_subset['wdir'] = xr.DataArray(
                180. + np.arctan2(ds_subset['u'].values,ds_subset['v'].values)*180./np.pi,
                dims=dim_keys)

    if do_surf_vars:
        print('Extracting 2-D variables (UST, HFX, QFX, SST, SSTSK)')
        ds_subset['UST'] = ds['UST']
        ds_subset['HFX'] = ds['HFX']
        ds_subset['QFX'] = ds['QFX']
        ds_subset['SST'] = ds['SST']
        ds_subset['SSTK'] = ds['SSTK']
    else:
        print("Skipping 2-D variables")

    # assign rename coord variable for time, and assign coordinates
    if specified_heights is None:
        ds_subset = ds_subset.assign_coords(z=ds_subset['SLICES_Z'])
    ds_subset = ds_subset.assign_coords(y=ds_subset['y'])
    ds_subset = ds_subset.assign_coords(x=ds_subset['x'])
    print(ds_subset.dims)
    ds_subset = ds_subset.rename_dims(dims_dict)

    return ds_subset


def write_tslist_file(fname,lat=None,lon=None,i=None,j=None,twr_names=None,twr_abbr=None):
    """
    Write a list of lat/lon or i/j locations to a tslist file that is
    readable by WRF.

    Usage
    ====
    fname : string 
        The path to and filename of the file to be created
    lat,lon,i,j : list or 1-D array
        Locations of the towers. 
        If using lat/lon - locx = lon, locy = lat
        If using i/j     - locx = i,   locy = j
    twr_names : list of strings, optional
        List of names for each tower location. Names should not be
        longer than 25 characters, each. If None, default names will
        be given.
    twr_abbr : list of strings, optional
        List of abbreviations for each tower location. Names should not be
        longer than 5 characters, each. If None, default abbreviations
        will be given.
    """
    if (lat is not None) and (lon is not None) and (i is None) and (j is None):
        header_keys = '# 24 characters for name | pfx |  LAT  |   LON  |'
        twr_locx = lon
        twr_locy = lat
        ij_or_ll = 'll'
    elif (i is not None) and (j is not None) and (lat is None) and (lon is None):
        header_keys = '# 24 characters for name | pfx |   I   |    J   |'
        twr_locx = i
        twr_locy = j
        ij_or_ll = 'ij'
    else:
        print('Please specify either lat&lon or i&j')
        return
    
    header_line = '#-----------------------------------------------#'
    header = '{}\n{}\n{}\n'.format(header_line,header_keys,header_line)
    
    if len(twr_locy) == len(twr_locx):
        ntowers = len(twr_locy)  
    else:
        print('Error - tower_x: {}, tower_y: {}'.format(len(twr_locx),len(twr_locy)))
        return
    
    if not isinstance(twr_names,list):
        twr_names = list(twr_names)    
    if twr_names != None:
        if len(twr_names) != ntowers:
            print('Error - Tower names: {}, tower_x: {}, tower_y: {}'.format(len(twr_names),len(twr_locx),len(twr_locy)))
            return
    else:
        twr_names = []
        for twr in np.arange(0,ntowers):
            twr_names.append('Tower{0:04d}'.format(twr+1))
            
    if not isinstance(twr_abbr,list):
        twr_abbr = list(twr_abbr)                
    if twr_abbr != None:
        if len(twr_abbr) != ntowers:
            print('Error - Tower abbr: {}, tower_x: {}, tower_y: {}'.format(len(twr_abbr),len(twr_locx),len(twr_locy)))
            return
        if len(max(twr_abbr,key=len)) > 5:
            print('Tower abbreviations are too large... setting to default names')
            twr_abbr = None
    if twr_abbr==None:
        twr_abbr = []
        for twr in np.arange(0,ntowers):
            twr_abbr.append('T{0:04d}'.format(twr+1))
            
    f = open(fname,'w')
    f.write(header)
            
    for tt in range(0,ntowers):
        if ij_or_ll == 'ij':
            twr_line = '{0:<26.25}{1: <6}{2: <8d} {3: <8d}\n'.format(
                twr_names[tt], twr_abbr[tt], int(twr_locx[tt]), int(twr_locy[tt]))
        else:
            twr_line = '{0:<26.25}{1: <6}{2:.7s}  {3:<.8s}\n'.format(
                twr_names[tt], twr_abbr[tt], '{0:8.7f}'.format(float(twr_locy[tt])), 
                                             '{0:8.7f}'.format(float(twr_locx[tt])))
        f.write(twr_line)
    f.close()
        
        

import glob
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import time
import matplotlib.colors as colors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

class ErrorWatch():
    
    def __init__(self,
                 working_dir=None,
                 domain_of_interest=None,
                 wait_time=5.0,
                 show_plot=True):
        
        if domain_of_interest is None:
            domain_of_interest = 'd01'
        terrain_cmap = truncate_colormap(plt.cm.terrain,minval=0.25)
        terrain_cmap.set_bad('steelblue')

        
        if working_dir is None:
            raise ValueError('working_dir must be specified')
            
        print('Getting information about the run...')
        datetime_dict = self.get_run_information(working_dir)

        print('Getting information about the domain...')
        domain_ds = self.get_domain_information(working_dir,domain_of_interest)
        self.domain_ds = domain_ds

        print('Initializing the CFL dataset...')
        cfl_ds = self.initialize_dataset(datetime_dict[domain_of_interest],domain_of_interest)
        last_time_step = cfl_ds.datetime[0].data
        
        print('Getting CFL information...')
        cfl_ds,rsl_dict,last_time_step = self.grep_error_logs(cfl_ds,rsl_loc=working_dir,last_time_step=last_time_step)
        self.cfl_ds = cfl_ds

        
        sim_running = self.check_run_status(cfl_ds)
        if show_plot:
            fig = plt.figure(figsize=(9,9))#,constrained_layout=True)
            plt.subplots_adjust(hspace=0.8)
            self.error_watch_plot(fig,cmap=terrain_cmap)
        
        sim_finished_when_starting = True
        
        if sim_running:
            print('It appears the simulation is still running... refreshing every {} seconds'.format(wait_time))
            need_to_iterate = True
        else:
            need_to_iterate = False


        while sim_running:
            cfl_ds,rsl_dict,last_time_step = self.grep_error_logs(cfl_ds,rsl_loc=working_dir,last_time_step=last_time_step)
            self.cfl_ds = cfl_ds
            sim_running = self.check_run_status(cfl_ds)
            if show_plot:
                plt.clf()
                self.error_watch_plot(fig,cmap=terrain_cmap)
                plt.pause(0.05)
            time.sleep(wait_time)

        self.cfl_ds = cfl_ds
        if show_plot:
            if need_to_iterate:
                self.error_watch_plot(fig,cmap=terrain_cmap)
            plt.show()
            
        print('The simulation has stopped. Result: {}'.format(cfl_ds.status))


        

    def error_watch_plot(self,fig,cmap):
        cfl_ds = self.cfl_ds
        domain_ds = self.domain_ds
        gs = fig.add_gridspec(3, 3)
        cfl_ax = fig.add_subplot(gs[0, :])
        xy_ax = fig.add_subplot(gs[1:, :-1])
        z_ax = fig.add_subplot(gs[1:, -1])

        w_plt = cfl_ds.w.where(cfl_ds.w != -999.).dropna(how='any',dim='datetime')
        cfl_plt = cfl_ds.cfl.where(cfl_ds.cfl != -999.).dropna(how='any',dim='datetime')

        #cfl_ax.plot(w_plt.datetime.data,w_plt.data,marker='o',c='b',label='W')
        #cfl_ax.plot(cfl_plt.datetime.data,cfl_plt.data,marker='o',label='CFL',c=t_locs,vmin=cfl_ds.datetime[0],vmax=cfl_ds.datetime[-1])
        
        cfl_ax.scatter(w_plt.datetime.data,w_plt.data,marker='o',c='grey',label='W')
        cfl_ax.scatter(cfl_plt.datetime.data,cfl_plt.data,marker='o',label='CFL',cmap=plt.cm.jet,c=w_plt.datetime.data,vmin=cfl_ds.datetime[0],vmax=cfl_ds.datetime[-1])

        
        progress_s = pd.to_datetime(cfl_ds.datetime[0].data)
        progress_e = pd.to_datetime(cfl_ds.finished_to)

        cfl_ax.set_ylim(0,1.1*np.max((cfl_ds.cfl.max().data,cfl_ds.w.max().data,1.0)))
        cfl_ax.set_xlim(cfl_ds.datetime[0].data,cfl_ds.datetime[-1].data)
        cfl_ax.plot([progress_s,progress_e],[0.0,0.0],c='g',lw=10.0,alpha=0.5)

        #xy_ax.pcolormesh(domain_ds.HGT.where(domain_ds.LANDMASK > 0),cmap=cmap,alpha=0.15)
        xy_ax.contour(domain_ds.HGT,alpha=0.15,cmap=plt.cm.copper,levels=np.arange(0,domain_ds.HGT.max(),200))
        xy_ax.contour(domain_ds.LANDMASK,levels=[0.5],colors='k',alpha=0.15)

        locs_ds = cfl_ds.location.where(cfl_ds.location != None).dropna(how='any',dim='datetime')

        loc_list = []
        t_locs = []
        for locs in locs_ds:
            loc_t = locs.datetime.data
            locs = str(locs.data)
            if '),(' in locs:
                locs = locs.split('),(')
                for loc in locs:
                    loc = loc.replace('(','').replace(')','')
                    loc_list.append(loc)
                    t_locs.append(loc_t)
            else:
                loc = locs.replace('(','').replace(')','')
                loc_list.append(loc)
                t_locs.append(loc_t)


        dz = domain_ds.z[1:] - domain_ds.z[:-1]
        #z_ax.scatter(np.arange(len(domain_ds.z)),domain_ds.z/1000.0,marker='o',color='k')
        z_ax.scatter(np.arange(len(dz)),dz,marker='o',color='k')
        show_scatter = True
        if show_scatter: 

            i_locs = []
            j_locs = []
            k_locs = []
            z_locs = []
            for loc in loc_list:
                loc = loc.split(',')
                i_locs.append(int(loc[0]))
                j_locs.append(int(loc[1]))
                k_locs.append(int(loc[2]))
                z_locs.append(dz.data[int(loc[2])])
                
            xy_ax.scatter(i_locs,j_locs,marker='o',cmap=plt.cm.jet,c=t_locs,vmin=cfl_ds.datetime[0],vmax=cfl_ds.datetime[-1])
            z_ax.scatter(k_locs,z_locs,marker='o',cmap=plt.cm.jet,c=t_locs,vmin=cfl_ds.datetime[0],vmax=cfl_ds.datetime[-1])
                
        cfl_ax.tick_params(labelsize=16,labelbottom=True,bottom=True)
        xticks = pd.date_range(str(cfl_ds.datetime.data[0]),str(cfl_ds.datetime.data[-1]),freq='6h')
        xticks = np.asarray(xticks)
        xtick_str = []
        for xt in np.arange(len(xticks)):
            xtick = pd.to_datetime(xticks[xt])
            xtick_str.append('{0:02d} {1:02d}Z'.format(xtick.day,xtick.hour))
        cfl_ax.set_xticks(xticks)
        cfl_ax.set_xticklabels(xtick_str,rotation=-35)
        #cfl_ax.tick_params(labelbottom=False,labeltop=True,bottom=False,top=True,labelsize=16)
        #cfl_ax.tick_params(axis='x',labelrotation=35.0)
        for label in cfl_ax.get_xticklabels():
            label.set_horizontalalignment('left')
        #cfl_ax.xaxis.set_label_position('top') 
        cfl_ax.set_xlabel('Datetime',size=18)
        cfl_ax.legend(frameon=False,fontsize=16,ncol=2,loc=(0.01,1.01))

        xy_ax.tick_params(labelsize=16)
        xy_ax.set_ylabel('NY',size=16)
        xy_ax.set_xlabel('NX',size=16)

        z_ax.tick_params(labelsize=16,labelleft=False,left=False,labelright=True,right=True)
        z_ax.set_xlabel('NZ',size=16)
        z_ax.set_ylabel('∆z [m]',size=16,rotation=270,labelpad=20)
        z_ax.yaxis.set_label_position('right') 

        plt.suptitle('{} - Current Time: {}'.format(str(cfl_ds.domain),progress_e),size=20)
        plt.draw()


    def remove_commas(self,string,make=None):
        if ',' in string: 
            string = string.replace(',','')
        if make is not None:
            if make == 'int': string = int(string)
            if make == 'float': string = float(string)

        return(string)

    def get_run_information(self,working_dir):
        namelist = open('{}namelist.input'.format(working_dir),'r')
        for line in namelist:
            line = line.split() 
            #print(line)
            if len(line) > 0:
                if 'start_year' in line[0]: start_yr = line[2:]
                if 'start_month' in line[0]:start_mo = line[2:]
                if 'start_day' in line[0]:start_dy = line[2:]
                if 'start_hour' in line[0]:start_hr = line[2:]
                if 'start_minute' in line[0]:start_mn = line[2:]
                if 'start_second' in line[0]:start_sd = line[2:]

                if 'end_year' in line[0]: end_yr = line[2:]
                if 'end_month' in line[0]:end_mo = line[2:]
                if 'end_day' in line[0]:end_dy = line[2:]
                if 'end_hour' in line[0]:end_hr = line[2:]
                if 'end_minute' in line[0]:end_mn = line[2:]
                if 'end_second' in line[0]:end_sd = line[2:]

                if 'max_dom' in line[0]:
                    max_dom = line[2]
                    max_dom = self.remove_commas(max_dom)
                    max_dom = int(max_dom)

                if 'time_step' in line[0]:
                    if line[0] == 'time_step':dt_b = self.remove_commas(line[2],make='int')
                    if 'fract_num' in line[0]: dt_num = self.remove_commas(line[2],make='int')
                    if 'fract_den' in line[0]: dt_den = self.remove_commas(line[2],make='int')
                    if 'ratio' in line[0]: dt_ratio = line[2:]


        namelist.close()
        main_dt = float(dt_b) + float(dt_num)/float(dt_den)
        datetime_dict = {}
        for dd in range(0,max_dom):
            dom = 'd{0:02d}'.format(dd+1)
            dom_dt = main_dt / self.remove_commas(dt_ratio[dd],make='float')
            syr = self.remove_commas(start_yr[dd],make='int')
            smo = self.remove_commas(start_mo[dd],make='int')
            sdy = self.remove_commas(start_dy[dd],make='int')
            shr = self.remove_commas(start_hr[dd],make='int')
            smn = self.remove_commas(start_mn[dd],make='int')
            ssd = self.remove_commas(start_sd[dd],make='int')

            eyr = self.remove_commas(end_yr[dd],make='int')
            emo = self.remove_commas(end_mo[dd],make='int')
            edy = self.remove_commas(end_dy[dd],make='int')
            ehr = self.remove_commas(end_hr[dd],make='int')
            emn = self.remove_commas(end_mn[dd],make='int')
            esd = self.remove_commas(end_sd[dd],make='int')

            start_date = pd.to_datetime('{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}'.format(
                                        syr,smo,sdy,shr,smn,ssd))
            end_date = pd.to_datetime('{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}'.format(
                                        eyr,emo,edy,ehr,emn,esd))


            freq = int(max([1.0,dom_dt]))
            datetime = pd.date_range(start_date,end_date,freq='{0}s'.format(freq))
            datetime_dict[dom] = datetime


        return(datetime_dict)


    def initialize_dataset(self,datetime,domain):
        cfl_ds = xr.Dataset(data_vars={
                                'cfl':(['datetime'],np.zeros(len(datetime))-999.0),
                               'npts':(['datetime'],np.zeros(len(datetime))),
                                  'w':(['datetime'],np.zeros(len(datetime))-999.0),
                           'location':(['datetime'],np.empty(len(datetime),dtype=list)),
                                      },
                            coords={'datetime':datetime},
                            attrs={'domain':domain},
                           )
        cfl_ds.w.astype(str)
        return(cfl_ds)

    def grep_error_logs(self,cfl_ds,rsl_dict=None,rsl_loc=None,last_time_step=None):
        if (rsl_dict is None) and (rsl_loc is None):
            raise ValueError('Please specify the location of rsl files (rsl_loc)')
        if (rsl_dict is None) and (rsl_loc is not None):
            rsl_files = sorted(glob.glob('{}rsl.error*'.format(rsl_loc)))
            rsl_dict = {}
            for log in rsl_files:
                rsl_dict[log] = {'stopping_ind':0}

        rsl_files = list(rsl_dict.keys())

        domain_of_interest = cfl_ds.domain
        sim_complete = False
        sim_crash = False
        
        for ee,elog in enumerate(rsl_files):
            log = open(elog,'r')
            log = log.readlines()[rsl_dict[elog]['stopping_ind']:]

            for line in log:

                if 'w_critical_cfl' in line:
                    line = line.split()
                    dom = line[0]
                    if dom == domain_of_interest:
                        npts = int(line[2])
                        dtime = pd.to_datetime(line[1].replace('_',' '))
                        cfl_ds.sel(datetime=dtime)['npts'] += npts
                if 'w-cfl' in line:
                    if "*******" in line: print(line)
                    line = line.split()
                    dom = line[0]
                    if dom == domain_of_interest:
                        dtime = pd.to_datetime(line[1].replace('_',' '))
                        cfl = float(line[10])
                        w = float(line[8])
                        i = str(line[4])
                        j = int(line[5])
                        k = int(line[6])
                        current_cfl = cfl_ds.sel(datetime=dtime).cfl
                        current_w = cfl_ds.sel(datetime=dtime).w
                        current_loc = cfl_ds.sel(datetime=dtime).location

                        if cfl > current_cfl:
                            cfl_ds.sel(datetime=dtime)['cfl'] *= 0.0
                            cfl_ds.sel(datetime=dtime)['cfl'] += cfl

                        if w > current_w:
                            cfl_ds.sel(datetime=dtime)['w'] *= 0.0
                            cfl_ds.sel(datetime=dtime)['w'] += w

                        if current_loc.data == None:
                            cfl_ds.location.loc[{'datetime':str(dtime)}] = ('({},{},{})'.format(i,j,k))

                        else:
                            current_loc_str = str(current_loc.data)
                            new_loc_str = ','.join([current_loc_str,'({},{},{})'.format(i,j,k)])
                            cfl_ds.location.loc[{'datetime':str(dtime)}]= new_loc_str


                if ee == 0:
                    if 'Timing for main' in line:
                        last_time_step = pd.to_datetime(line.split()[4].replace('_',' '))
                if 'SUCCESS COMPLETE WRF' in line:
                    sim_complete = True
                if 'Program received signal SIGSEGV' in line:
                    sim_complete = True
                    sim_crash = True

                rsl_dict[elog]['stopping_ind'] += 1
                
        if sim_complete: 
            if sim_crash:
                status = 'Crashed'
            else:
                status = 'Complete'
        else:
            status = 'Running'
        cfl_ds.attrs['status'] = status
        cfl_ds.attrs['finished_to'] = last_time_step
        #cfl_ds = cfl_ds.where(cfl_ds != -999.)
        return(cfl_ds,rsl_dict,last_time_step)

    def check_run_status(self,cfl_ds):
        if cfl_ds.status == 'Running':
            sim_running = True
        else:
            sim_running = False
        return(sim_running)

    def get_domain_information(self,working_dir,domain):
        wrfin = xr.open_dataset('{}wrfinput_{}'.format(working_dir,domain)).squeeze()
        hgt = wrfin.HGT
        z = ((wrfin.PH + wrfin.PHB)/9.81 - hgt).mean(dim=['west_east','south_north'])

        domain_ds = wrfin[['HGT','LANDMASK']]
        domain_ds['z'] = z
        return(domain_ds)
