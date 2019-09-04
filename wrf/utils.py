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
import xarray
import utm # need to pip install!
from scipy.spatial import KDTree
from scipy.interpolate import interp1d, LinearNDInterpolator


# List of default WRF fields for extract_column_from_wrfdata
default_3D_fields = ['U10','V10','T2','TSK','UST','PSFC','HFX','LH','MUU','MUV','MUT']
default_4D_fields = ['U','V','W','T',
                     'RU_TEND','RU_TEND_ADV','RU_TEND_PGF','RU_TEND_COR','RU_TEND_PHYS',
                     'RV_TEND','RV_TEND_ADV','RV_TEND_PGF','RV_TEND_COR','RV_TEND_PHYS',
                     'T_TEND_ADV',]


def read_tslist(fpath):
    return pd.read_csv(fpath,comment='#',delim_whitespace=True,
                       names=['name','prefix','lat','lon'])


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
    elif isinstance(wrfdata, xarray.Dataset):
        try:
            return wrfdata.dims[dimname]
        except KeyError:
            print('No {:s} dimension'.format(dimname))
            return None
    else:
        raise AttributeError('Unexpected WRF data type')

def _get_dim_names(wrfdata,varname):
    """Returns dimension names of the specified variable,
    with support for both netCDF4 and xarray
    """
    if isinstance(wrfdata, netCDF4.Dataset):
        try:
            return wrfdata.variables[varname].dimensions
        except KeyError:
            print('No {:s} dimension'.format(dimname))
            return None
    elif isinstance(wrfdata, xarray.Dataset):
        try:
            return wrfdata.variables[varname].dims
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
    elif isinstance(wrfdata, xarray.Dataset):
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
    '''
    def __init__(self,fstr):
        """The file-path string should be:
            '[path to towers]/[tower abrv.].d0[domain].*'
        """
        self.time = None
        self.nt = None
        self.nz = None
        self._getvars(fstr)
        self._getdata()

    def _getvars(self,fstr):
        if not fstr.endswith('*'):
            fstr += '*'
        self.filelist = glob.glob(fstr)
        self.nvars = len(self.filelist)
        self.varns = []
        for fpath in self.filelist:
            self.varns.append(fpath.split('.')[-1])

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
                    header = f.readline().replace('(',' ').replace(')',' ').replace(',',' ').split()
                    self.longname = header[0]
                    self.abbr     = header[3]
                    self.lat      = float(header[4])
                    self.lon      = float(header[5])
                    self.loci     = int(header[6])
                    self.locj     = int(header[7])
                    self.stationz = float(header[10])
                    # Note: need to look up what tslist outputs to know which
                    # vars are where...
                    self.ts = pd.read_csv(f,delim_whitespace=True,header=None).values[:,2:]
                    assert (self.ts.shape == (nt,nv))

    def to_dataframe(self,
                     start_time='2013-11-08',time_unit='h',time_step=None,
                     heights=None,height_var='height',
                     exclude=['ts']):
        """Convert tower time-height data into a dataframe.
        
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
            height index (if heights is not None).
        exclude : list
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
        else:
            timestep = pd.to_timedelta(time_step, unit='s')
            endtime = start_time + self.nt*timestep
            times = pd.date_range(start=start_time+timestep,
                                  end=endtime,
                                  periods=self.nt)
        # combine data
        if heights is None:
            # heights will be an integer index
            z = np.arange(self.nz)
            columns = pd.MultiIndex.from_product([varns,z],names=[None,'height'])
            arraydata = np.concatenate(
                [ getattr(self,varn) for varn in varns ], axis=1
            )
            df = pd.DataFrame(data=arraydata,index=times,columns=columns)
            return df.stack()
        else:
            from scipy.interpolate import interp1d
            z = np.array(heights)
            # TODO

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
    assert(spatial_filter in ['nearest','interpolate','average']),\
            'Spatial filtering type "'+spatial_filter+'" not recognised'

    # Load WRF data
    xa = xarray.open_dataset(fpath)
    tdim, zdim, ydim, xdim = get_wrf_dims(xa)
    
    
    #---------------------------
    # Preprocessing
    #---------------------------
    
    # Extract WRF grid resolution
    dx_meso = xa.attrs['DX']
    assert(dx_meso == xa.attrs['DY'])
    
    
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
            ndim = len(xa[field].dims)
        except KeyError:
            print('The additional field "'+field+'" is not available and will be ignored.')
        else:
            if len(xa[field].dims) == 3:
                if field not in fieldnames_3D:
                    fieldnames_3D.append(field)
            elif len(xa[field].dims) == 4:
                if field not in fieldnames_4D:
                    fieldnames_4D.append(field)
            else:
                raise Exception('Field "'+field+'" is not 3D or 4D, not sure how to process this field.')
    
    
    #---------------------------
    # Load data
    #---------------------------
    
    WRFdata = {}
    
    # Cell-centered coordinates
    XLAT = xa.variables['XLAT'].values     # WRF indexing XLAT[time,lat,lon]
    XLONG = xa.variables['XLONG'].values
    
    # Height above ground level
    WRFdata['Zagl'],_ = get_height(xa,timevarying=True)
    WRFdata['Zagl']   = add_surface_plane(WRFdata['Zagl'])
    
    # 3D fields. WRF indexing Z[tdim,ydim,xdim]
    for field in fieldnames_3D:
        WRFdata[field] = get_unstaggered_var(xa,field)
        
    
    # 4D fields. WRF indexing Z[tdim,zdim,ydim,xdim]
    for field in fieldnames_4D:
        WRFdata[field] = get_unstaggered_var(xa,field)
            
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
    
    coords = {'Time': xa['XTIME'].values,
              'height': zmicro}
    
    data_vars = {}
    for field in fieldnames_3D:
        data_vars[field] = ('Time',
                            sitedata[field],
                            {'description': xa[field].attrs['description'].lower(),
                             'units': xa[field].attrs['units']}
                           )
    for field in fieldnames_4D:
        data_vars[field] = (['Time','height'],
                            sitedata[field],
                            {'description': xa[field].attrs['description'].lower(),
                             'units': xa[field].attrs['units']}
                           )
        
    
    xn = xarray.Dataset(data_vars=data_vars,coords=coords)
    
    # Rename T to theta and adjust description
    xn = xn.rename({'T':'theta'})
    xn['theta'].attrs['description'] = 'potential temperature'

    return xn
