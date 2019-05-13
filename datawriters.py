"""
Standardized data output routines
"""
from netCDF import Dataset

standard_datetime_fmt = '%Y-%m-%d %H:%M:%S'

# for netCDF output
core_variables = ['Times','u','v','w','wspd','wdir','T','p','theta','RH']

def write_to_netCDF(nc_filename, data,
                    ncformat='NETCDF4_CLASSIC',
                    all_variables=False,
                    verbose=True):
    """This will write a new netCDF file from the data provided in a
    dictionary. Several global variables will be set and care must be
    taken that the dictionary variables are named the same as what this
    function expects.
    """
    ncfile = Dataset(nc_filename,'w', format=ncformat, clobber=True)
    for dd,dim in enumerate(data['dims']):
        ncfile.createDimension(data['dimname'][dd],dim)
    for vv,varname in enumerate(data['varn']):
        if all_variables:
            newvar = ncfile.createVariable(varname,
                                           data['vardtype'][vv],
                                           data['vardims'][vv])
            newvar[:] = data['data'][vv]
            newvar.units = data['units'][vv]
        else:
            if varname in core_variables:
                newvar = ncfile.createVariable(varname,
                                               data['vardtype'][vv],
                                               data['vardims'][vv],
                                               fill_value=data['fillValue'])
                newvar[:] = data['data'][vv]
                if verbose:
                    print(varname)
                    print(newvar[newvar == np.nan])
                newvar[newvar == np.nan] = data['fillValue']
                newvar.units = data['units'][vv]

    ncfile.createDimension('nchars',19)
    newvar[:] = data['time']
    ncfile.description = data['description']
    ncfile.station     = data['station']
    ncfile.sensor      = data['sensor']
    ncfile.latitude    = data['latitude']
    ncfile.longitude   = data['longitude']
    ncfile.altitude    = data['altitude']
    ncfile.createdon   = datetime.datetime.now().strftime(standard_datetime_fmt)
    ncfile.createdby   = data['author']

