'''
  What it does: Given a WRF file and a lat/lon pair, several surface
                variables and profiles will be extracted and written
                to a new file.

  Who made it: patrick.hawbecker@nrel.gov 
  When: 2/4/19
'''
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as ncdf
import subprocess
import wrfdict as wrfdict

# - - - - - - - - - - - - #
# User Settings:
stnlat = 45.638004   # Latitude of interest
stnlon = -120.642973 # Longitude of interest
fdir = '/projects/wfip/WFIP2/improved/extracted/' # Location of WRF data
dom  = 1 # WRF Domain of interest
newfname = 'ExtractedWRFdataForWFIP2_d0%d.nc' % dom # New file name
savedir = '/projects/wfip/' # Where to save new output
# - - - - - - - - - - - - #

# Find all wrfout_d0X files to loop over. Expects 1 time per file
wrfoutf = subprocess.check_output('ls %swrfout_d0%d_*00' % (fdir,dom),shell=True).split()
nt = np.shape(wrfoutf)[0] # Number of times

# Initialize time-series variables; 
pblh  = np.zeros((nt)); ustr  = np.zeros((nt))
u10   = np.zeros((nt)); v10   = np.zeros((nt))
T2    = np.zeros((nt)); TH2   = np.zeros((nt))
swdwn = np.zeros((nt)); psfc  = np.zeros((nt))
time  = np.zeros((nt)); wdate = np.zeros((nt))
hfx   = np.zeros((nt))

# - - - WRF Data - - -
cc = 0 # Start counter
for ff in wrfoutf: # Loop over all WRF files
    wrfout     = ncdf(ff)
    year,month,day,hour = wrfdict.wrftimes2hours(wrfout) # Finds the WRF time
    wdate[cc] = year*10000 + month*100 + day # Generates an integer in form YYYMMDD
    time[cc]  = hour
    if cc == 0: # Initialize 2D vars and gather necessary variables
        poii, poij = wrfdict.latlon2ij(wrfout,stnlat,stnlon) # i,j location
                                                # closest to given lat/lon
        z,zs = wrfdict.getheightloc(wrfout,poij,poii) # Get z values at location
        nz = np.shape(zs)[0] # number of heights
        height = zs[:,poij,poii] # tower heights
        # Initialize 2D variables
        u = np.zeros((nt,nz))
        v = np.zeros((nt,nz))
        w = np.zeros((nt,nz))
        T = np.zeros((nt,nz))
        P = np.zeros((nt,nz))
        Q = np.zeros((nt,nz))
    # Load variables from WRF file
    hfx[cc]    = wrfout.variables['HFX'][0,poij,poii]
    pblh[cc]   = wrfout.variables['PBLH'][0,poij,poii]
    psfc[cc]   = wrfout.variables['PSFC'][0,poij,poii]
    ustr[cc]   = wrfout.variables['UST'][0,poij,poii]
    u10[cc]    = wrfout.variables['U10'][0,poij,poii]
    v10[cc]    = wrfout.variables['V10'][0,poij,poii]
    T2[cc]     = wrfout.variables['T2'][0,poij,poii]
    TH2[cc]    = wrfout.variables['TH2'][0,poij,poii]
    swdwn[cc]  = wrfout.variables['SWDOWN'][0,poij,poii]
    # U and V need to be interpolated to cell-center
    u[cc]      = wrfdict.unstagger2d(wrfout.variables['U'][0,:,poij,poii-1:poii+1],ax=1)[:,0]
    v[cc]      = wrfdict.unstagger2d(wrfout.variables['V'][0,:,poij-1:poij+1,poii],ax=1)[:,0]
    # W needs to be unstaggered in the vertical direction
    ws         = wrfout.variables['W'][0,:,poij,poii]
    w[cc]      = (ws[1:] + ws[:-1])*0.5
    # T is perturbation temp... need to add 300.0 K
    T[cc]      = wrfout.variables['T'][0,:,poij,poii]+300.0
    # Pressure is perturbation + base
    P[cc]      = wrfout.variables['P'][0,:,poij,poii]+wrfout.variables['PB'][0,:,poij,poii]
    # Mixing ratio of water vapor
    Q[cc]      = wrfout.variables['QVAPOR'][0,:,poij,poii]

    cc += 1 # increase counter
# Write new NetCDF file
newncdf = ncdf('%s%s' % (savedir,newfname),'w',format='NETCDF4_CLASSIC')
# Create time and height dimensions
newncdf.createDimension('NZ',nz)
newncdf.createDimension('time',nt)
# Set global values
newncdf.location = '(%f, %f)' % (wrfout.variables['XLAT'][0,poij,poii], wrfout.variables['XLONG'][0,poij,poii])
newncdf.elevation = '%f m' % wrfout.variables['HGT'][0,poij,poii]
newncdf.description = \
        'Extracted by Patrick Hawbecker (patrick.hawbecker@nrel.gov) on Feb. 4, 2019 from wrfout files located at %s' % fdir

# Create new variables
times    = newncdf.createVariable('Time',np.float64, ('time',))
dates    = newncdf.createVariable('Date',np.float64, ('time',))
hgts     = newncdf.createVariable('Height',np.float64, ('NZ',))
hfxo     = newncdf.createVariable('HFX',np.float64, ('time',))
pblho    = newncdf.createVariable('PBLH',np.float64, ('time',))
psfco    = newncdf.createVariable('PSFC',np.float64, ('time',))
ustro    = newncdf.createVariable('USTAR',np.float64, ('time',))
u10o     = newncdf.createVariable('U10',np.float64, ('time',))
v10o     = newncdf.createVariable('V10',np.float64, ('time',))
T2o      = newncdf.createVariable('T2',np.float64, ('time',))
TH2o     = newncdf.createVariable('TH2',np.float64, ('time',))
swdwno   = newncdf.createVariable('SWDOWN',np.float64, ('time',))
uout     = newncdf.createVariable('U',np.float64, ('time','NZ',))
vout     = newncdf.createVariable('V',np.float64, ('time','NZ',))
wout     = newncdf.createVariable('W',np.float64, ('time','NZ',))
Tout     = newncdf.createVariable('T',np.float64, ('time','NZ',))
Pout     = newncdf.createVariable('P',np.float64, ('time','NZ',))
Qout     = newncdf.createVariable('Q',np.float64, ('time','NZ',))
# Assign data to new variables
times[:]   = time
dates[:]   = wdate
hgts[:]    = height
hfxo[:]    = hfx
pblho[:]   = pblh 
psfco[:]   = psfc 
ustro[:]   = ustr
u10o[:]    = u10
v10o[:]    = v10
T2o[:]     = T2
TH2o[:]    = TH2
swdwno[:]  = swdwn
uout[:]    = u
vout[:]    = v
wout[:]    = w
Tout[:]    = T
Pout[:]    = P
Qout[:]    = Q
# Write and close the new NetCDF file
newncdf.close()
