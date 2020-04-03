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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import netCDF4
import xarray as xr
from scipy.spatial import KDTree
from scipy.interpolate import interp1d, LinearNDInterpolator
import wrf as wrfpy

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
]

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

    def to_dataframe(self,start_time,
                     time_unit='h',time_step=None,
                     heights=None,height_var='height',agl=False,
                     exclude=['ts']):
        """Convert tower time-height data into a dataframe.
        
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
            times.name = 'datetime'
        else:
            timestep = pd.to_timedelta(time_step, unit='s')
            endtime = start_time + self.nt*timestep + pd.to_timedelta(np.round(self.time[0],decimals=1),unit='h')
            try:
                times = pd.date_range(start=(start_time+timestep+pd.to_timedelta(np.round(self.time[0],decimals=1),unit='h')),
                                      end=endtime,
                                      periods=self.nt,
                                      name='datetime')
            except TypeError:
                print('Trying with index')
                times = pd.date_range(start=(start_time+timestep+pd.to_timedelta(np.round(self.time[0],decimals=1),unit='h'))[0],
                                      end=endtime[0],
                                      periods=self.nt,
                                      name='datetime')
            
        # combine (and interpolate) time-height data
        # - note 1: self.[varn].shape == self.height.shape == (self.nt, self.nz)
        # - note 2: arraydata.shape == (self.nt, len(varns)*self.nz)
        arraydata = np.concatenate(
            [ getattr(self,varn) for varn in varns ], axis=1
        )
        if heights is None:
            if hasattr(self, height_var):
                # heights (constant in time) were separately calculated
                z = getattr(self, height_var)
                assert (len(z.shape) == 1) and (len(z) == self.nz), \
                        'tower '+height_var+' attribute should correspond to fixed height levels'
            else:
                # heights will be an integer index
                z = np.arange(self.nz)
            columns = pd.MultiIndex.from_product([varns,z],names=[None,'height'])
            df = pd.DataFrame(data=arraydata,index=times,columns=columns).stack()
        else:
            from scipy.interpolate import interp1d
            z = np.array(heights)
            zt = getattr(self, height_var)
            if agl:
                zt -= self.stationz
            if len(zt.shape) == 1:
                # approximately constant height (with time)
                assert len(zt) == self.nz
                columns = pd.MultiIndex.from_product([varns,zt],names=[None,'height'])
                df = pd.DataFrame(data=arraydata,index=times,columns=columns).stack()
                # now unstack the times to get a height index
                unstacked = df.unstack(level=0)
                interpfun = interp1d(unstacked.index, unstacked.values, axis=0,
                                     bounds_error=False,
                                     fill_value='extrapolate')
                interpdata = interpfun(z)
                unstacked = pd.DataFrame(data=interpdata,
                                         index=z, columns=unstacked.columns)
                unstacked.index.name = 'height'
                df = unstacked.stack().reorder_levels(order=['datetime','height']).sort_index()
            else:
                # interpolate for all times
                assert zt.shape == (self.nt, self.nz), \
                        'heights should correspond to time-height indices'
                nvarns = len(varns)
                newarraydatalist = []
                for varn in varns:
                    newarraydata = np.empty((self.nt, len(z)))
                    curfield = getattr(self,varn)
                    for itime in range(self.nt):
                        interpfun = interp1d(zt[itime,:], curfield[itime,:],
                                             bounds_error=False,
                                             fill_value='extrapolate')
                        newarraydata[itime,:] = interpfun(z)
                    newarraydatalist.append(newarraydata)
                newarraydata = np.concatenate(newarraydatalist, axis=1)
                columns = pd.MultiIndex.from_product([varns,z],names=[None,'height'])
                unstacked = pd.DataFrame(data=newarraydata,index=times,columns=columns)
                df = unstacked.stack()

        # standardize names
        df.rename(columns=self.standard_names, inplace=True)
        return df

    def to_xarray(self,
                  start_time='2013-11-08',time_unit='h',time_step=None,
                  heights=None,height_var='height',
                  exclude=['ts'],
                  structure='ordered'):
        
        df = self.to_dataframe(start_time,time_unit,time_step,heights,height_var,exclude)
        if structure == 'ordered':
            ds = df.to_xarray().assign_coords(i=self.loci).assign_coords(j=self.locj).expand_dims(['j','i'],axis=[2,3])
            ds = ds.reset_index(['height'], drop = True).rename_dims({'height':'k'})
            ds = ds.assign_coords(height=ds.ph)
            ds["lat"] = (['j','i'],  np.ones((1,1))*self.gridlat)
            ds["lon"] = (['j','i'],  np.ones((1,1))*self.gridlon)
            # Add zsurface (station height) as a data variable:
            ds['zsurface'] = (['j','i'],  np.ones((1,1))*self.stationz)
        elif structure == 'unordered':
            ds = df.to_xarray().assign_coords(station=self.abbr).expand_dims(['station'],axis=[2])
            ds = ds.reset_index(['height'], drop = True).rename_dims({'height':'k'})
            ds = ds.assign_coords(height=ds.ph)
            ds["lat"] = (['station'],  [self.gridlat])
            ds["lon"] = (['station'],  [self.gridlon])
            ds['zsurface'] = (['station'],  [self.stationz])
        
        for varn in self.ts_varns:
            tsvar = getattr(self, varn.lower())
            ds[varn.lower()] = (['datetime','j','i'],np.expand_dims(np.expand_dims(tsvar,axis=1),axis=1) )
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
                                T0=300.,
                                spatial_filter='interpolate',L_filter=0.0,
                                additional_fields=[],
                                verbose=False,
                               ):
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
    ds = xr.open_dataset(fpath)
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
        if field is 'T':
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


def combine_towers(fdir, restarts, simulation_start, fname, return_type='xarray', structure='ordered',
                   time_step=None, heights=None, height_var='heights'):
    '''
    Combine together tslist files in time where, if there is any overlap, the later file
    will overwrite the earlier file. This makes the assumption that all of the tslist 
    files are stored in separate directories but named the same (default in WRF is to 
    name them the same). Each restart directory must have a simulation_start string to
    specify when the timing starts (use same time if run was an actual restart, use WRF
    start time if you are combining several runs).

    fdir             = 'path/to/restart/directories/'
    restarts         = ['restart_dir_1', 'restart_dir_2', 'restart_dir_3']
    simulation_start = ['2000-01-01 00:00','2000-01-01 00:00','2000-01-01 00:00']
    fname            = ['t0001.d02'] (Note: this is the prefix for the tower + domain)
    return_type      = 'xarray' or 'dataframe'
    structure        = 'ordered' or 'unordered'

    This will work with a pandas df or an xarray ds/da
    '''
    for rst,restart in enumerate(restarts):
        if np.size(simulation_start) == 1:
            sim_start = simulation_start
        elif np.size(simulation_start) == np.size(restarts):
            sim_start = simulation_start[rst]
        else:
            raise ValueError('restarts and simulation_start are not equal')
        print('restart: {}'.format(restart))
        data = []
        for ff in fname:
            
            print('starting {}'.format(ff))
            if return_type == 'xarray':
                data.append(Tower('{}{}/{}'.format(fdir,restart,ff)).to_xarray(start_time=sim_start,
                                                                            time_step=time_step,
                                                                            structure=structure,
                                                                            heights=heights,
                                                                            height_var=height_var))
            elif return_type == 'dataframe':
                data.append(Tower('{}{}/{}'.format(fdir,restart,ff)).to_dataframe(start_time=sim_start,
                                                                            time_step=time_step,
                                                                            structure=structure,
                                                                            heights=heights,
                                                                            height_var=height_var))
        data_block = xr.combine_by_coords(data)
        if np.shape(restarts)[0] > 1:
            if rst == 0:
                data_previous = data_block

            else:
                dataF = data_block.combine_first(data_previous)
                data_previous = dataF
        else:
            dataF = data_block
    if structure == 'ordered':
    # -------------------------------------------------------       
    #               MMC Format specifications
        dataF = dataF.rename_dims({'k':'nz',
                                   'i':'nx',
                                   'j':'ny'})
        dx,dy = 12.0, 12.0
        xcoord,ycoord = np.meshgrid(dataF.i*dx,dataF.j*dy)
        dataF = dataF.assign_coords(x=(('ny','nx'),xcoord)).assign_coords(y=(('ny','nx'),ycoord))
        dataF = dataF.assign_coords(z=dataF.ph).reset_index(['i','j'],drop=True).drop('ph')
        dataF.attrs['DX'] = dx
        dataF.attrs['DY'] = dy
    elif structure == 'unordered':
        dataF = dataF.rename_dims({'k':'nz'})
    dataF = dataF.assign_coords(lat=dataF.lat).assign_coords(lon=dataF.lon)
    dataF = dataF.assign_coords(zsurface=dataF.zsurface)
    dataF['wspd'] = (dataF['u']**2.0 + dataF['v']**2.0)**0.5
    dataF['wdir'] = 180. + np.degrees(np.arctan2(dataF['u'], dataF['v']))        

    dataF.attrs['SIMULATION_START_DATE'] = sim_start
    dataF.attrs['CREATED_FROM'] = fdir

    # -------------------------------------------------------       
    #dx,dy = 12.0, 12.0
    #xcoord,ycoord = np.meshgrid(dataF.i*dx,dataF.j*dy)
    #dataF = dataF.assign_coords(x=(('j','i'),xcoord)).assign_coords(y=(('j','i'),ycoord)).assign_coords(lat=dataF.lat).assign_coords(lon=dataF.lon)
    #dataF = dataF.assign_coords(z=dataF.ph).drop('ph').assign_coords(k=dataF.k)

    return dataF


def tsout_seriesReader(fdir, restarts, simulation_start_time, domain_of_interest, structure='ordered',
                       time_step=None, heights=None, height_var='heights',select_tower=None):
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
    ntimes = np.shape(restarts)[0]
    floc = '{}{}/*{}.??'.format(fdir,restarts[0],domain_of_interest)
    file_list = glob.glob(floc)

    for ff,file in enumerate(file_list):
        file = file[:-3]
        file_list[ff] = file
    for f in file_list: 
        if 'geo_em' in f: file_list.remove(f)

    file_list = np.unique(file_list)
    tower_names = file_list.copy()
    for ff,file in enumerate(file_list):
        tower_names[ff] = file.split('/')[-1]
        
    if not isinstance(select_tower,(list)):
        select_tower = list(select_tower)
    if select_tower is not None:
        good_towers = []
        for twr in select_tower:
            for twr_n in tower_names:
                if twr in twr_n: good_towers.append(twr_n)
        tower_names = good_towers
    
    dsF = combine_towers(fdir,restarts,simulation_start_time,tower_names,return_type='xarray',
                         structure=structure, time_step=time_step, heights=heights, height_var=height_var)
    return dsF


def wrfout_seriesReader(wrf_path,wrf_file_filter,specified_heights=None):
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
        variables should be	interpolated. Note that this significantly
        increases the data read time.
    """
    TH0 = 300.0 #WRF convention base-state theta = 300.0 K
    dims_dict = {
        'Time':'datetime',
        'bottom_top':'nz',
        'south_north': 'ny',
        'west_east':'nx',
    }

    ds = xr.open_mfdataset(os.path.join(wrf_path,wrf_file_filter),
                           chunks={'Time': 10},
                           combine='nested',
                           concat_dim='Time')
    dim_keys = ["Time","bottom_top","south_north","west_east"] 
    horiz_dim_keys = ["south_north","west_east"]
    print('Finished opening/concatenating datasets...')

    ds_subset = ds[['XTIME']]
    print('Establishing coordinate variables, x,y,z, zSurface...')
    zcoord = wrfpy.destagger((ds['PHB'] + ds['PH']) / 9.8, stagger_dim=1, meta=False)
    #ycoord = ds.DY * np.tile(0.5 + np.arange(ds.dims['south_north']),
    #                         (ds.dims['west_east'],1))
    #xcoord = ds.DX * np.tile(0.5 + np.arange(ds.dims['west_east']),
    #                         (ds.dims['south_north'],1)) 
    ycoord = ds.DY * (0.5 + np.arange(ds.dims['south_north']))
    xcoord = ds.DX * (0.5 + np.arange(ds.dims['west_east']))
    ds_subset['z'] = xr.DataArray(zcoord, dims=dim_keys)
    #ds_subset['y'] = xr.DataArray(np.transpose(ycoord), dims=horiz_dim_keys)
    #ds_subset['x'] = xr.DataArray(xcoord, dims=horiz_dim_keys)
    ds_subset['y'] = xr.DataArray(ycoord, dims='south_north')
    ds_subset['x'] = xr.DataArray(xcoord, dims='west_east')

    # Assume terrain height is static in time even though WRF allows
    # for it to be time-varying for moving grids
    ds_subset['zsurface'] = xr.DataArray(ds['HGT'].isel(Time=0), dims=horiz_dim_keys)
    print('Destaggering data variables, u,v,w...')
    ds_subset['u'] = xr.DataArray(wrfpy.destagger(ds['U'],stagger_dim=3,meta=False),
                                  dims=dim_keys)
    ds_subset['v'] = xr.DataArray(wrfpy.destagger(ds['V'],stagger_dim=2,meta=False),
                                  dims=dim_keys)
    ds_subset['w'] = xr.DataArray(wrfpy.destagger(ds['W'],stagger_dim=1,meta=False),
                                  dims=dim_keys)

    print('Extracting data variables, p,theta...')
    ds_subset['p'] = xr.DataArray(ds['P']+ds['PB'], dims=dim_keys)
    ds_subset['theta'] = xr.DataArray(ds['THM']+TH0, dims=dim_keys)

    # optionally, interpolate to static heights	
    if specified_heights is not None:	
        zarr = ds_subset['z']	
        for var in ['u','v','w','p','theta']:	
            print('Interpolating',var)	
            interpolated = wrfpy.interplevel(ds_subset[var], zarr, specified_heights)	
            ds_subset[var] = interpolated #.expand_dims('Time', axis=0)	
            #print(ds_subset[var])
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
    
    # assign rename coord variable for time, and assign ccordinates 
    ds_subset = ds_subset.rename({'XTIME': 'datetime'})  #Rename after defining the component DataArrays in the DataSet
    if specified_heights is None:
        ds_subset = ds_subset.assign_coords(z=ds_subset['z'])
    ds_subset = ds_subset.assign_coords(y=ds_subset['y'])
    ds_subset = ds_subset.assign_coords(x=ds_subset['x'])
    ds_subset = ds_subset.assign_coords(zsurface=ds_subset['zsurface'])
    ds_subset = ds_subset.rename_vars({'XLAT':'lat', 'XLONG':'lon'})
    #print(ds_subset)
    ds_subset = ds_subset.rename_dims(dims_dict)
    #print(ds_subset)
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
        
        
  
