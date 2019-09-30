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
from datetime import datetime
import netCDF4
import xarray as xr
##JAS temp remove###  import utm # need to pip install!
from scipy.spatial import KDTree
from scipy.interpolate import interp1d, LinearNDInterpolator
import wrf as wrfpy

# List of default WRF fields for extract_column_from_wrfdata
default_3D_fields = ['U10','V10','T2','TSK','UST','PSFC','HFX','LH','MUU','MUV','MUT']
default_4D_fields = ['U','V','W','T',
                     'RU_TEND','RU_TEND_ADV','RU_TEND_PGF','RU_TEND_COR','RU_TEND_PHYS',
                     'RV_TEND','RV_TEND_ADV','RV_TEND_PGF','RV_TEND_COR','RV_TEND_PHYS',
                     'T_TEND_ADV',]

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

def wrfout_seriesReader(wrfpath,wrfFileFilter,desiredHeights=None):
    """
    Construct an a2e-mmc standard, xarrays-based, data structure from a
    series of 3-dimensional WRF output files

    Note: Base state theta= 300.0 K is assumed by convention in WRF,
        this function follow this convention.

    Usage
    ====
    wrfpath : string 
        The path to directory containing wrfout files to be processed
    wrfFileFilter : string-glob expression
        A string-glob expression to filter a set of 4-dimensional WRF
        output files.
    desiredHeights : list-like
        A list of static heights to which all data variables should be
        interpolated. Note that this significantly increases the data
        read time.
    """
    TH0 = 300.0 #WRF convention base-state theta = 300.0 K
    dims_dict = {
        'Time':'datetime',
        'bottom_top':'nz',
        'south_north': 'ny',
        'west_east':'nx',
    }

    ds = xr.open_mfdataset(os.path.join(wrfpath,wrfFileFilter),
                           chunks={'Time': 10},
                           combine='nested',
                           concat_dim='Time')
    dim_keys = ["Time","bottom_top","south_north","west_east"] 
    horiz_dim_keys = ["south_north","west_east"]
    print('Finished opening/concatenating datasets...')

    ds_subset = ds[['XTIME']]
    print('Establishing coordinate variables, x,y,z, zSurface...')
    height = wrfpy.destagger((ds['PHB'] + ds['PH']) / 9.8, stagger_dim=1, meta=False)
    ycoord = ds.DY * np.tile(0.5 + np.arange(ds.dims['south_north']),
                             (ds.dims['west_east'],1))
    xcoord = ds.DX * np.tile(0.5 + np.arange(ds.dims['west_east']),
                             (ds.dims['south_north'],1)) 
    ds_subset['z'] = xr.DataArray(height, dims=dim_keys)
    ds_subset['y'] = xr.DataArray(np.transpose(ycoord), dims=horiz_dim_keys)
    ds_subset['x'] = xr.DataArray(xcoord, dims=horiz_dim_keys)

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
    if desiredHeights is not None:
        zarr = ds_subset['z']
        for var in ['u','v','w','p','theta']:
            print('Interpolating',var)
            interpolated = wrf.interplevel(ds_subset[var], zarr, desiredHeights)
            ds_subset[var] = interpolated.expand_dims('Time', axis=0)
        ds_subset = ds_subset.drop_dims('bottom_top').rename({'level':'z'})
        dim_keys[1] = 'z'
        dims_dict.pop('bottom_top')
        
    # calculate derived variables
    print('Calculating derived data variables, wspd,wdir...')
    ds_subset['wspd'] = xr.DataArray(np.sqrt(ds_subset['u']**2 + ds_subset['v']**2),
                                     dims=dim_keys)
    ds_subset['wdir'] = xr.DataArray(180. + np.arctan2(ds_subset['u'],ds_subset['v'])*180./np.pi,
                                     dims=dim_keys)
    
    # assign 'height' as a coordinate in the dataset
    ds_subset = ds_subset.rename({'XTIME': 'datetime'})  #Rename after defining the component DataArrays in the DataSet
    if desiredHeights is None:
        ds_subset = ds_subset.assign_coords(z=ds_subset['z'])

    ds_subset = ds_subset.assign_coords(y=ds_subset['y'])
    ds_subset = ds_subset.assign_coords(x=ds_subset['x'])
    ds_subset = ds_subset.assign_coords(zsurface=ds_subset['zsurface'])
    ds_subset = ds_subset.rename_vars({'XLAT':'lat', 'XLONG':'lon'})
    ds_subset = ds_subset.rename_dims(dims_dict)
    return ds_subset

