"""
Helper functions for calculating standard meteorological quantities
"""
import numpy as np
import pandas as pd
import xarray as xr
import time
import glob
from datetime import datetime,timedelta

# constants
epsilon = 0.622 # ratio of molecular weights of water to dry air


def e_s(T, celsius=False, model='Tetens'):
    """Calculate the saturation vapor pressure of water, $e_s$ [mb]
    given the air temperature ([K] by default).
    """
    T = T.copy()
    if celsius:
        # input is deg C
        T_degC = T
        T = T_degC + 273.15
    else:
        # input is in Kelvin
        T_degC = T - 273.15
    if model == 'Bolton':
        # Eqn 10 from Bolton (1980), Mon. Weather Rev., Vol 108
        # - applicable from -30 to 35 deg C
        svp = 6.112 * np.exp(17.67*T_degC / (T_degC + 243.5))
    elif model == 'Magnus':
        # Eqn 21 from Alduchov and Eskridge (1996), J. Appl. Meteorol., Vol 35
        # - AERK formulation, applicable from -40 to 50 deg C
        svp = 6.1094 * np.exp(17.625*T_degC / (243.04 + T_degC))
    elif model == 'Tetens':
        # Tetens' formula, e.g., from the National Weather Service:
        # https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
        svp = 6.11 * 10**(7.5*T_degC/(237.3+T_degC))
    else:
        raise ValueError('Unknown model: {:s}'.format(model))
    return svp


def T_d(T, RH, celsius=False, model='NWS'):
    """Calculate the dewpoint temperature, $T_d$, from air temperature
    and relative humidity [%]. If celsius is True, input and output 
    temperatures are in degrees Celsius; otherwise, inputs and outputs
    are in Kelvin.
    """
    if model == 'NWS':
        es = e_s(T, celsius, model='Tetens')
        # From National Weather Service, using Tetens' formula:
        # https://www.weather.gov/media/epz/wxcalc/virtualTemperature.pdf
        # - note the expression for vapor pressure is the saturation vapor
        #   pressure expression, with Td instead of T
        e = RH/100. * es
        denom = 7.5*np.log(10) - np.log(e/6.11)
        Td = 237.3 * np.log(e/6.11) / denom
        if not celsius:
            Td += 273.15
    else:
        raise ValueError('Unknown model: {:s}'.format(model))
    return Td


def w_s(T,p,**kwargs):
    """Calculate the saturation mixing ratio, $w_s$ [kg/kg] given the
    air temperature ([K] by default) and station pressure [mb].

    See e_s() for additional information for the kwargs.
    """
    # First calculate the saturation vapor pressure
    es = e_s(T,**kwargs)
    # From Wallace & Hobbs, Eqn 3.63:
    return epsilon * es / (p - es)


def T_to_Tv(T,p=None,RH=None,e=None,w=None,Td=None,
            celsius=False,verbose=False):
    """Convert moist air temperature to virtual temperature.
    
    Formulas based on given total (or "station") pressure (p [mbar]) and
    relative humidity (RH [%]); mixing ratio (w [kg/kg]); or partial
    pressures of water vapor and dry air (e, pd [mbar]); or dewpoint
    temperature (Td).
    """
    T = T.copy()
    if celsius:
        T_degC = T
        T = T_degC + 273.15
    else:
        T_degC = T - 273.15
    if (p is not None) and (RH is not None):
        # saturation vapor pressure of water, e_s [mbar]
        es = e_s(T)
        if verbose:
            # sanity check!
            es_est = e_s(T, model='Bolton')
            print('e_s(T) =',es,'~=',es_est)
            es_est = e_s(T, model='Magnus')
            print('e_s(T) =',es,'~=',es_est)
        # saturation mixing ratio, ws [-]
        ws = w_s(T, p)
        if verbose:
            print('w_s(T,p) =',ws,'~=',epsilon*es/p)
        # mixing ratio, w, from definition of relative humidity
        w = (RH/100.) * ws
        if verbose:
            # we also have specific humidity, q, at this point (not needed)
            q = w / (1+w)
            print('q(T,p,RH) =',q)
        if verbose:
            # sanity check: Wallace & Hobbs, Eqn 3.60
            # - assumes mixing ratio is small (i.e., w^2 ~ 0)
            #   Tv - T = (1-epsilon)/epsilon * wT / (1 + w)
            #          = (1-epsilon)/epsilon * wT * (1 - w + w^2 + ...)
            #         ~= (1-epsilon)/epsilon * wT
            #      Tv ~= T*(1 + ((1-epsilon)/epsilon)*w)
            print('Tv(T,p,RH) ~=',T*(1+0.61*w))
        # Using Wallace & Hobbs, Eqn 3.59
        Tv = T * (w/epsilon + 1) / (1 + w)
    elif (e is not None) and (p is not None):
        # Definition of virtual temperature
        #   Wallace & Hobbs, Eqn 3.16
        Tv = T / (1 - e/p*(1-epsilon))
    elif w is not None:
        # Using Wallace & Hobbs, Eqn 3.59 substituted into 3.16
        Tv = T * (w/epsilon + 1) / (1 + w)
    elif (Td is not None) and (p is not None):
        # From National Weather Service, using Tetens' formula:
        # https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
        Td_degC = Td.copy()
        if not celsius:
            Td_degC -= 273.15
        e = e_s(Td_degC, celsius=True, model='Tetens')
        # Calculate from definition of virtual temperature
        Tv = T_to_Tv(T,e=e,p=p)
    else:
        raise ValueError('Specify (T,RH,p) or (T,e,p) or (T,w), or (T,Td,p)')
    if celsius:
        Tv -= 273.15
    return Tv


def Ts_to_Tv(Ts,**kwargs):
    """TODO: Convert sonic temperature [K] to virtual temperature [K].
    """


def calc_wind(df,u='u',v='v'):
    """Calculate wind speed and direction from horizontal velocity
    components, u and v.
    """
    if isinstance(df,pd.DataFrame):
        fields = df.columns
    elif isinstance(df,xr.Dataset):
        fields = df.variables
    if not all(velcomp in fields for velcomp in [u,v]):
        print(('velocity components u/v not found; '
               'set u and/or v to calculate wind speed/direction'))
    else:
        wspd = np.sqrt(df[u]**2 + df[v]**2)
        wdir = 180. + np.degrees(np.arctan2(df[u], df[v]))
        return wspd, wdir

def calc_uv(df,wspd='wspd',wdir='wdir'):
    """Calculate velocity components from wind speed and direction.
    """
    if isinstance(df,pd.DataFrame):
        fields = df.columns
    elif isinstance(df,xr.Dataset):
        fields = df.variables
    if not all(windvar in fields for windvar in [wspd,wdir]):
        print(('wind speed/direction not found; '
               'set wspd and/or wpd to calculate velocity components'))
    else:
        ang = np.radians(270. - df[wdir])
        u = df[wspd] * np.cos(ang)
        v = df[wspd] * np.sin(ang)
        return u,v


def theta(T, p, p0=1000.):
    """Calculate (virtual) potential temperature [K], theta, from (virtual)
    temperature T [K] and pressure p [mbar] using Poisson's equation.

    Standard pressure p0 at sea level is 1000 mbar or hPa. 

    Typical assumptions for dry air give:
        R/cp = (287 J/kg-K) / (1004 J/kg-K) = 0.286
    """
    return T * (p0/p)**0.286

# create alias for theta for consistency
T_to_theta = theta
def theta_to_T(theta,p,p0=1000.):
    """Calculate (virtual) temperature [K], from (virtual) potential
    temperature, theta, [K] and pressure p [mbar] using Poisson's equation.

    Standard pressure p0 at sea level is 1000 mbar or hPa. 

    Typical assumptions for dry air give:
        R/cp = (287 J/kg-K) / (1004 J/kg-K) = 0.286
    """
    return theta / (p0/p)**0.286    

def covariance(a,b,interval='10min',resample=False,**kwargs):
    """Calculate covariance between two series (with datetime index) in
    the specified interval, where the interval is defined by a pandas
    offset string
    (http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects).

    Notes:
    - The output data will have the same length as the input data by
      default, because statistics are calculated with pd.rolling(). To
      return data at the same intervals as specified, set
      `resample=True`.
    - Covariances may be simultaneously calculated at multiple heights
      by inputting multi-indexed dataframes (with height being the
      second index level)
    - If the inputs have multiindices, this function will return a
      stacked, multi-indexed dataframe.

    Example:
        heatflux = covariance(df['Ts'],df['w'],'10min')
    """
    # handle multiindices
    have_multiindex = False
    if isinstance(a.index, pd.MultiIndex):
        assert isinstance(b.index, pd.MultiIndex), \
               'Both a and b should have multiindices'
        assert len(a.index.levels) == 2
        assert len(b.index.levels) == 2
        # assuming levels 0 and 1 are time and height, respectively
        a = a.unstack() # create unstacked copy
        b = b.unstack() # create unstacked copy
        have_multiindex = True
    elif isinstance(b.index, pd.MultiIndex):
        raise AssertionError('Both a and b should have multiindices')
    # check index
    if isinstance(interval, str):
        # make sure we have a compatible index
        assert isinstance(a.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex))
        assert isinstance(b.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex))
    # now, do the calculations
    if resample:
        a_mean = a.resample(interval).mean()
        b_mean = b.resample(interval).mean()
        ab_mean = (a*b).resample(interval,**kwargs).mean()
    else:
        a_mean = a.rolling(interval).mean()
        b_mean = b.rolling(interval).mean()
        ab_mean = (a*b).rolling(interval,**kwargs).mean()
    cov = ab_mean - a_mean*b_mean
    if have_multiindex:
        return cov.stack()
    else:
        return cov


def power_spectral_density(df,tstart=None,interval=None,window_size='10min',
                           window_type='hanning',detrend='linear',scaling='density',
                           num_overlap=None):
    """
    Calculate power spectral density using welch method and return
    a new dataframe. The spectrum is calculated for every column
    of the original dataframe.

    Notes:
    - Input can be a pandas series or dataframe
    - Output is a dataframe with frequency as index
    """
    from scipy.signal import welch
    
    # Determine time scale
    timevalues = df.index.get_level_values(0)

    if isinstance(timevalues,pd.DatetimeIndex):
        timescale = pd.to_timedelta(1,'s')
    else:
        # Assuming time is specified in seconds
        timescale = 1

    # Determine tstart and interval if not specified
    if tstart is None:
        tstart = timevalues[0]
    if interval is None:
        interval = timevalues[-1] - timevalues[0]
    elif isinstance(interval,str):
        interval = pd.to_timedelta(interval)

    # Update timevalues
    inrange = (timevalues >= tstart) & (timevalues <= tstart+interval)
    timevalues = df.loc[inrange].index.get_level_values(0)

    # Determine sampling rate and samples per window
    dts = np.diff(timevalues.unique())/timescale
    dt  = dts[0]

    if type(window_type) is str:
        nperseg = int( pd.to_timedelta(window_size)/pd.to_timedelta(dt,'s') )
    else:
        nperseg = len(window_type)
    assert(np.allclose(dts,dt)),\
        'Timestamps must be spaced equidistantly'

    # If input is series, convert to dataframe
    if isinstance(df,pd.Series):
        df = df.to_frame()

    spectra = {}
    for col in df.columns:
        f,P = welch(df.loc[inrange,col], fs=1./dt, nperseg=nperseg,
                    detrend=detrend,window=window_type,scaling=scaling,
                    noverlap=num_overlap)    
        spectra[col] = P
    spectra['frequency'] = f
    return pd.DataFrame(spectra).set_index('frequency')
    

def power_law(z,zref=80.0,Uref=8.0,alpha=0.2):
    return Uref*(z/zref)**alpha

def fit_powerlaw(df=None,z=None,U=None,zref=80.0,Uref=None):
    """Calculate power-law exponent to estimate shear.

    Parameters
    ==========
    df : pd.DataFrame, optional
        Calculate from data columns; index should be height values
    U : str or array-like, optional
        An array of wind speeds if dataframe 'df' is not provided speeds
    z : array-like, optional
        An array of heights if dataframe 'df' is not provided
    zref : float
        Power-law reference height
    Uref : float or array-like, optional
        Power-law reference wind speed; if not specified, then the wind
        speeds are evaluated at zref to get Uref

    Returns
    =======
    alpha : float or pd.Series
        Shear exponents
    R2 : float or pd.Series
        Coefficients of determination
    """
    from scipy.optimize import curve_fit
    # generalize all inputs
    if df is None:
        assert (U is not None) and (z is not None)
        df = pd.DataFrame(U, index=z)
    elif isinstance(df,pd.Series):
        df = pd.DataFrame(df)
    # make sure we're only working with above-ground values
    df = df.loc[df.index > 0]
    z = df.index
    logz = np.log(z) - np.log(zref)
    # evaluate Uref at zref, if needed
    if Uref is None:
        Uref = df.loc[zref]
    elif not hasattr(Uref, '__iter__'):
        Uref = pd.Series(Uref,index=df.columns)
    # calculate shear coefficient
    alpha = pd.Series(index=df.columns)
    R2 = pd.Series(index=df.columns)
    def fun(x,*popt):
        return popt[0]*x
    for col,U in df.iteritems():
        logU = np.log(U) - np.log(Uref[col])
        popt, pcov = curve_fit(fun,xdata=logz,ydata=logU,p0=0.14,bounds=(0,1))
        alpha[col] = popt[0]
        U = df[col]
        resid = U - Uref[col]*(z/zref)**alpha[col]
        SSres = np.sum(resid**2)
        SStot = np.sum((U - np.mean(U))**2)
        R2[col] = 1.0 - (SSres/SStot)
    return alpha.squeeze(), R2.squeeze()

def fit_power_law_alpha(z,U,zref=80.0,Uref=8.0):
    """DEPRECATED: use fit_powerlaw instead"""
    from scipy.optimize import curve_fit
    above0 = (z > 0)
    logz = np.log(z[above0]) - np.log(zref)
    logU = np.log(U[above0]) - np.log(Uref)
    fun = lambda logz,alpha: alpha*logz
    popt, pcov = curve_fit(fun, xdata=logz, ydata=logU,
                           p0=0.2, bounds=(0,np.inf))
    alpha = popt[0]
    resid = U - Uref*(z/zref)**alpha
    SSres = np.sum(resid**2)
    SStot = np.sum((U - np.mean(U))**2)
    R2 = 1.0 - (SSres/SStot)
    return alpha, R2

def lowess_mean(ds,win_size,lowess_delta):
    '''
    This will calculate the lowess mean with specified window size
    and lowess delta.
    ds : xarray dataset or data array
    win_size : float
    lowess_delta: float
    '''
    series_length = ds.data.size
    sm_frac = win_size/series_length
    exog = np.arange(len(ds.data))

    init_ds_means = True

    lowess_smth = lowess(ds.data, exog, 
                         frac=sm_frac, 
                         delta=lowess_delta)[:,1]
    return(lowess_smth)

def model4D_calcQOIs(ds,mean_dim,data_type='wrfout', mean_opt='static', lowess_delta=0, lowess_window=None):
    """
    Augment an a2e-mmc standard, xarrays-based, data structure of 
    4-dimensional model output with space-based quantities of interest

    Usage
    ====
    ds : mmc-4D standard xarray DataSet 
        The raw standard mmc-4D data structure 
    mean_dim : string 
        Dimension along which to calculate mean and fluctuating (perturbation) parts
    data_type : string
        Either 'wrfout' or 'ts' for tslist output
    mean_opt : string
        Which technique of calculating the mean do you want to use:
        static = one value (mean over mean_dim)
        lowess = smoothed mean (over mean_dim)
    lowess_delta : float
        delta in the lowess smoothing function. Expecting float relating to
        number of time steps in some time length to average over.
        e.g., wanting 30 min intervals with time step of 0.1 s would yeild
        lowess_delta = 1800.0*0.1 = 18000.0
        Setting lowess_delta = 0 means no linear averaging (default)
    """

    dim_keys = [*ds.dims.keys()]
    print('Calculating means... this may take a while.')
    if data_type == 'ts':
        var_keys = ['u','v','w','theta','wspd','wdir']
    else:
        var_keys = [*ds.data_vars.keys()]

    if mean_opt == 'static':
        print('calculating static means')
        ds_means = ds.mean(dim=mean_dim)
    elif mean_opt == 'lowess':
        print('calculating lowess means')
        init_ds_means = True
        for varn in var_keys:
            print(varn)
            if varn == 'wspd':
                var_str = '{:s}Mean'.format('U')
            else:
                var_str = '{:s}Mean'.format(varn)

            lowess_smth = np.zeros((ds.datetime.data.size, ds.nz.data.size, 
                                    ds.ny.data.size, ds.nx.data.size))
            loop_start = time.time()
            for kk in ds.nz.data:
                k_loop_start = time.time()
                var = ds[varn].isel(nz=kk).values
                for jj in ds.ny.data:
                    for ii in ds.nx.data:
                        lowess_smth[:,kk,jj,ii] = lowess_mean(var[:,jj,ii], 
                                                         win_size=lowess_window,
                                                         delta=lowess_delta)
                print('k-loop: {} seconds'.format(time.time()-k_loop_start))
                loop_end = time.time()
            print('total time: {} seconds'.format(loop_end - loop_start))
            if init_ds_means:
                ds_means = xr.Dataset({varn:(['datetime','nz','ny','nx'], lowess_smth)},
                                       coords=ds.coords)
                init_ds_means = False
            else:
                ds_means[varn] = (['datetime','nz','ny','nx'], lowess_smth)
    else:
        print('Please select "static" or "lowess"')
        return
    ds_perts = ds-ds_means

    ds['uMean'] = ds_means['u']
    ds['vMean'] = ds_means['v']
    ds['wMean'] = ds_means['w']
    ds['thetaMean'] = ds_means['theta']
    ds['UMean'] = ds_means['wspd']
    ds['UdirMean'] = ds_means['wdir']
    if data_type != 'ts':
        ds['pMean'] = ds_means['p']

    print('calculating variances / covariances...')
    ds['uu'] = ds_perts['u']**2
    ds['vv'] = ds_perts['v']**2
    ds['ww'] = ds_perts['w']**2
    ds['uv'] = ds_perts['u']*ds_perts['v']
    ds['uw'] = ds_perts['u']*ds_perts['w']
    ds['vw'] = ds_perts['v']*ds_perts['w']
    ds['wth'] = ds_perts['w']*ds_perts['theta']
    ds['UU'] = ds_perts['wspd']**2
    ds['Uw'] = ds_perts['wspd']*ds_perts['w']
    ds['TKE'] = 0.5*np.sqrt(ds['UU']+ds['ww'])
    ds.attrs['MEAN_OPT'] = mean_opt
    if mean_opt == 'lowess':
        ds.attrs['WINDOW_SIZE'] = lowess_window
        ds.attrs['LOWESS_DELTA'] = lowess_delta
    return ds

def model4D_spectra(ds,spectra_dim,average_dim,vert_levels,horizontal_locs,fld,fldMean):
    """
    Using an a2e-mmc standard, xarrays-based, data structure of 
    4-dimensional model output with space-based quantities of interest,
    calculate energy spectra at specified vertical indices and 
    streamwise horizontal indices, averaged over the time instances in ds.

    Usage
    ====
    ds : mmc-4D standard xarray DataSet 
        The raw standard mmc-4D data structure 
    spectra_dim : string 
        Dimension along which to calculate spectra
    average_dim : string 
        Dimension along which to average spectra
    vert_levels :
        vertical levels over which to calculate spectra
    horizontal_locs : 
        horizontal (non-spectra_dim) locations at which to calculate spectra
    fld : string
        Name of the field in the dataset tocalculate spectra of 
    fldMean : string
        Name of the mean of fld in the dataset
    """
    from scipy.signal import welch
    from scipy.signal.windows import hann, hamming

    print('Averaging spectra (in {:s}) over {:d} instances in {:s}'.format(
                                spectra_dim,ds.dims[average_dim],average_dim))
    nblock = ds.dims[spectra_dim]
    if 'y' in spectra_dim:
        dt = ds.attrs['DY']
    elif 'x' in spectra_dim:
        dt = ds.attrs['DX']
    elif spectra_dim == 'datetime':
        dt = float(ds.datetime[1].data - ds.datetime[0].data)/1e9
        
    fs = 1 / dt
    overlap = 0
    win = hamming(nblock, True) #Assumed non-periodic in the spectra_dim
    
    init_Puuf_cum = True

    for cnt_lvl,level in enumerate(vert_levels): # loop over levels
        print('grabbing a slice...')
        spec_start = time.time()
        series_lvl = ds[fld].isel(nz=level)-ds[fldMean].isel(nz=level)
        series_lvl.name = 'varn'
        print(time.time() - spec_start)
        for cnt_i,iLoc in enumerate(horizontal_locs): # loop over x
            for cnt,it in enumerate(range(ds.dims[average_dim])): # loop over y
                if spectra_dim == 'datetime':
                    series = series_lvl.isel(nx=iLoc,ny=it)
                    if (type(series) == xr.Dataset) or (type(series) == xr.DataArray):
                        series = series.to_dataframe()
                        for key in series.keys():
                            if key != 'varn':
                                series = series.drop([key],axis=1)
                    
                elif 'y' in spectra_dim:
                    series = series_lvl.isel(nx=iLoc,datetime=it)
                else:
                    print('Please choose spectral_dim of \'ny\', or \'datetime\'')
                #f, Pxxfc = welch(series, fs, window=win, noverlap=overlap, 
                #                 nfft=nblock, return_onesided=False, detrend='constant')
                #Pxxf = np.multiply(np.real(Pxxfc),np.conj(Pxxfc))
                
                Pxxf = power_spectral_density(series,window_type=win,detrend='constant')
                if it == 0:
                    if init_Puuf_cum:
                        Puuf_cum = np.zeros((len(vert_levels),len(horizontal_locs),len(Pxxf)))
                        init_Puuf_cum = False
                    Puuf_cum[cnt_lvl,cnt_i,:] = Pxxf.varn
                    sum_count = 1
                else:
                    Puuf_cum[cnt_lvl,cnt_i,:] += Pxxf.varn
                    sum_count += 1
    #Puuf = 2.0*(1.0/cnt)*Puuf_cum[:,:,:(np.floor(ds.dims[spectra_dim]/2).astype(int))]   ###2.0 is to account for the dropping of the negative side of the FFT 
    #f = f[:(np.floor(ds.dims[spectra_dim]/2).astype(int))]
    Puuf = (1.0/sum_count)*Puuf_cum
    f = Pxxf.index.get_level_values('frequency')
    return f,Puuf

def model4D_spatial_spectra(ds,spectra_dim,vert_levels,horizontal_locs,fld,fldMean):
    """
    Using an a2e-mmc standard, xarrays-based, data structure of 
    4-dimensional model output with space-based quantities of interest,
    calculate energy spectra at specified vertical indices and 
    streamwise horizontal indices, averaged over the time instances in ds.

    Usage
    ====
    ds : mmc-4D standard xarray DataSet 
        The raw standard mmc-4D data structure 
    spectra_dim : string 
        Dimension along which to calculate spectra
    vert_levels :
        vertical levels over which to calculate spectra
    horizontal_locs : 
        horizontal (non-spectra_dim) locations at which to calculate spectra
    fld : string
        Name of the field in the dataset tocalculate spectra of 
    fldMean : string
        Name of the mean of fld in the dataset
    """
    from scipy.signal import welch
    from scipy.signal.windows import hann, hamming

    print('Averaging spectra over {:d} time-instances'.format(ds.dims['datetime']))
    nblock = ds.dims[spectra_dim]
    if 'y' in spectra_dim:
        dt = ds.attrs['DY']
    elif 'x' in spectra_dim:
        dt = ds.attrs['DX']
    fs = 1 / dt
    overlap = 0
    win = hamming(nblock, True) #Assumed non-periodic in the spectra_dim
    Puuf_cum = np.zeros((len(vert_levels),len(horizontal_locs),ds.dims[spectra_dim]))

    cnt=0.0
    for it in range(ds.dims['datetime']):
        cnt_lvl = 0
        for level in vert_levels:
            cnt_i = 0
            for iLoc in horizontal_locs:
                series = ds[fld].isel(datetime=it,nz=level,nx=iLoc)-ds[fldMean].isel(datetime=it,nz=level,nx=iLoc)

                f, Pxxfc = welch(series, fs, window=win, noverlap=overlap, nfft=nblock, return_onesided=False, detrend='constant')
                Pxxf = np.multiply(np.real(Pxxfc),np.conj(Pxxfc))
                if it == 0:
                    Puuf_cum[cnt_lvl,cnt_i,:] = Pxxf
                else:
                    Puuf_cum[cnt_lvl,cnt_i,:] = Puuf_cum[cnt_lvl,cnt_i,:] + Pxxf
                    cnt_i = cnt_i+1
            cnt_lvl = cnt_lvl + 1
        cnt = cnt+1.0
    Puuf = 2.0*(1.0/cnt)*Puuf_cum[:,:,:(np.floor(ds.dims['ny']/2).astype(int))]   ###2.0 is to account for the dropping of the negative side of the FFT 
    f = f[:(np.floor(ds.dims['ny']/2).astype(int))]

    return f,Puuf

def model4D_cospectra(ds,spectra_dim,average_dim,vert_levels,horizontal_locs,fldv0,fldv0Mean,fldv1,fldv1Mean):
    """
    Using an a2e-mmc standard, xarrays-based, data structure of 
    4-dimensional model output with space-based quantities of interest,
    calculate cospectra from two fields at specified vertical indices and 
    streamwise horizontal indices, averaged over the dimension specified
    by average_dim.

    Usage
    ====
    ds : mmc-4D standard xarray DataSet 
        The raw standard mmc-4D data structure 
    spectra_dim : string 
        Dimension along which to calculate spectra
    average_dim : string 
        Dimension along which to averag the spectra
    vert_levels :
        vertical levels over which to caluclate spectra
    horizontal_locs : 
        horizontal (non-spectra_dim) locations at which to calculate spectra
    fldv0 : string
        Name of the first field in the dataset in desired cospectra of 
    fldv0Mean : string
        Name of the mean of fldv0 in the dataset
    fldv1 : string
        Name of the second field in the dataset in deisred cospectra  
    fldv1Mean : string
        Name of the mean of fldv1 in the dataset
    """
    from scipy.signal import welch
    from scipy.signal.windows import hann, hamming

    print('Averaging cospectra (in {:s}) over {:d} instances in {:s}'.format(
          spectra_dim,ds.dims[average_dim],average_dim))
    nblock = ds.dims[spectra_dim]
    if 'y' in spectra_dim:
        dt = ds.attrs['DY']
    elif 'x' in spectra_dim:
        dt = ds.attrs['DX']
    elif spectra_dim == 'datetime':
        dt = float(ds.datetime[1].data - ds.datetime[0].data)/1e9

    fs = 1 / dt
    overlap = 0
    win = hamming(nblock, True) #Assumed non-periodic in the spectra_dim
    Puuf_cum = np.zeros((len(vert_levels),len(horizontal_locs),ds.dims[spectra_dim]))
    for cnt_lvl,level in enumerate(vert_levels): # loop over levels
        spec_start = time.time()
        print('Grabbing slices in z')
        series0_lvl = ds[fldv0].isel(nz=level)-ds[fldv0Mean].isel(nz=level)
        series1_lvl = ds[fldv1].isel(nz=level)-ds[fldv1Mean].isel(nz=level)
        for cnt_i,iLoc in enumerate(horizontal_locs): # loop over x
            for cnt,it in enumerate(range(ds.dims[average_dim])): # loop over average dim
                if spectra_dim == 'datetime':
                    series0 = series0_lvl.isel(nx=iLoc,ny=it)
                    series1 = series1_lvl.isel(nx=iLoc,ny=it)
                elif 'y' in spectra_dim:
                    series0 = series0_lvl.isel(nx=iLoc,datetime=it)
                    series1 = series1_lvl.isel(nx=iLoc,datetime=it)
                f, Pxxfc0 = welch(series0, fs, window=win, noverlap=overlap, 
                            nfft=nblock, return_onesided=False, detrend='constant')
                f, Pxxfc1 = welch(series1, fs, window=win, noverlap=overlap, 
                            nfft=nblock, return_onesided=False, detrend='constant')
                Pxxf = (np.multiply(np.real(Pxxfc0),np.conj(Pxxfc1))+
                        np.multiply(np.real(Pxxfc1),np.conj(Pxxfc0)))
                if it == 0:
                    Puuf_cum[cnt_lvl,cnt_i,:] = Pxxf
                else:
                    Puuf_cum[cnt_lvl,cnt_i,:] = Puuf_cum[cnt_lvl,cnt_i,:] + Pxxf
    Puuf = 2.0*(1.0/cnt)*Puuf_cum[:,:,:(np.floor(ds.dims[spectra_dim]/2).astype(int))]  ###2.0 is to account for the dropping of the negative side of the FFT
    f = f[:(np.floor(ds.dims[spectra_dim]/2).astype(int))]

    return f,Puuf

def model4D_spatial_cospectra(ds,spectra_dim,vert_levels,horizontal_locs,fldv0,fldv0Mean,fldv1,fldv1Mean):
    """
    Using an a2e-mmc standard, xarrays-based, data structure of 
    4-dimensional model output with space-based quantities of interest,
    calculate cospectra from two fields at specified vertical indices and 
    streamwise horizontal indices, averaged over the time instances in ds.

    Usage
    ====
    ds : mmc-4D standard xarray DataSet 
        The raw standard mmc-4D data structure 
    spectra_dim : string 
        Dimension along which to calculate spectra
    vert_levels :
        vertical levels over which to caluclate spectra
    horizontal_locs : 
        horizontal (non-spectra_dim) locations at which to calculate spectra
    fldv0 : string
        Name of the first field in the dataset in desired cospectra of 
    fldv0Mean : string
        Name of the mean of fldv0 in the dataset
    fldv1 : string
        Name of the second field in the dataset in deisred cospectra  
    fldv1Mean : string
        Name of the mean of fldv1 in the dataset
    """
    from scipy.signal import welch
    from scipy.signal.windows import hann, hamming

    print('Averaging spectra over {:d} time-instances'.format(ds.dims['datetime']))
    nblock = ds.dims[spectra_dim]
    if 'y' in spectra_dim:
        dt = ds.attrs['DY']
    elif 'x' in spectra_dim:
        dt = ds.attrs['DX']
    fs = 1 / dt
    overlap = 0
    win = hamming(nblock, True) #Assumed non-periodic in the spectra_dim
    Puuf_cum = np.zeros((len(vert_levels),len(horizontal_locs),ds.dims[spectra_dim]))

    cnt=0.0
    for it in range(ds.dims['datetime']):
        cnt_lvl = 0
        for level in vert_levels:
            cnt_i = 0
            for iLoc in horizontal_locs:
                series0 = ds[fldv0].isel(datetime=it,nz=level,nx=iLoc)-ds[fldv0Mean].isel(datetime=it,nz=level,nx=iLoc)
                f, Pxxfc0 = welch(series0, fs, window=win, noverlap=overlap, nfft=nblock, return_onesided=False, detrend='constant')
                series1 = ds[fldv1].isel(datetime=it,nz=level,nx=iLoc)-ds[fldv1Mean].isel(datetime=it,nz=level,nx=iLoc)
                f, Pxxfc1 = welch(series1, fs, window=win, noverlap=overlap, nfft=nblock, return_onesided=False, detrend='constant')

                Pxxf = (np.multiply(np.real(Pxxfc0),np.conj(Pxxfc1))+np.multiply(np.real(Pxxfc1),np.conj(Pxxfc0)))
                if it == 0:
                    Puuf_cum[cnt_lvl,cnt_i,:] = Pxxf
                else:
                    Puuf_cum[cnt_lvl,cnt_i,:] = Puuf_cum[cnt_lvl,cnt_i,:] + Pxxf
                    cnt_i = cnt_i+1
            cnt_lvl = cnt_lvl + 1
        cnt = cnt+1.0
    Puuf = 2.0*(1.0/cnt)*Puuf_cum[:,:,:(np.floor(ds.dims['ny']/2).astype(int))]  ###2.0 is to account for the dropping of the negative side of the FFT
    f = f[:(np.floor(ds.dims['ny']/2).astype(int))]

    return f,Puuf

def model4D_pdfs(ds,pdf_dim,vert_levels,horizontal_locs,fld,fldMean,bins_vector):
    """
    Using an a2e-mmc standard, xarrays-based, data structure of 
    4-dimensional model output with space-based quantities of interest,
    calculate probability distributions at specified vertical indices and 
    streamwise horizontal indices, accumulated over the time instances in ds.

    Usage
    ====
    ds : mmc-4D standard xarray DataSet 
        The raw standard mmc-4D data structure 
    pdf_dim : string 
        Dimension along which to calculate probability distribution functions
    vert_levels :
        vertical levels over which to caluclate probability distribution functions
    horizontal_locs : 
        horizontal (non-pdf_dim) locations at which to calculate probability distribution functions
    fld : string
	Name of the field in the dataset to calculate pdfs on
    fldMean : string
	Name of the mean of fld in the dataset 
    """

    from scipy.stats import skew,kurtosis
    print('Accumulating statistics over {:d} time-instances'.format(ds.dims['datetime']))
    sk_vec=np.zeros((len(vert_levels),len(horizontal_locs)))
    kurt_vec=np.zeros((len(vert_levels),len(horizontal_locs)))
    hist_cum = np.zeros((len(vert_levels),len(horizontal_locs),bins_vector.size-1))
    cnt_lvl = 0
    for level in vert_levels:
        cnt_i = 0
        for iLoc in horizontal_locs:
            dist=np.ndarray.flatten(((ds[fld]).isel(nz=level,nx=iLoc)-(ds[fldMean]).isel(nz=level,nx=iLoc)).values)
            sk_vec[cnt_lvl,cnt_i]=skew(dist)
            kurt_vec[cnt_lvl,cnt_i]=kurtosis(dist)
            hist,bin_edges=np.histogram(dist, bins=bins_vector)
            hist_cum[cnt_lvl,cnt_i,:] = hist
            cnt_i = cnt_i+1
        cnt_lvl = cnt_lvl+1
#    cnt=0.0
#    for it in range(ds.dims['datetime']):
#        ##### If it seems like this is taking a long time, check progress occasionally by uncommenting 2-lines below
#        #if int((it/ds.dims['datetime'])%10*100)%10 == 0:
#        #    print('working...{:2d}% complete'.format(int((it/ds.dims['datetime'])%10*100)))
#        cnt_lvl = 0
#        for level in vert_levels:
#            cnt_i = 0
#            for iLoc in horizontal_locs:
#                y = (ds[fld].isel(datetime=it,nz=level,nx=iLoc)-ds[fldMean].isel(datetime=it,nz=level,nx=iLoc))
#                #y = np.ndarray.flatten(dist.isel(nz=level,nx=iLoc).values)
#                hist,bin_edges=np.histogram(y, bins=bins_vector)
#                if it is 0:
#                    hist_cum[cnt_lvl,cnt_i,:] = hist
#                else:
#                    hist_cum[cnt_lvl,cnt_i,:] = hist_cum[cnt_lvl,cnt_i,:] + hist
#                cnt_i = cnt_i+1
#            cnt_lvl = cnt_lvl+1
#        cnt = cnt+1.0

    return hist_cum, bin_edges, sk_vec, kurt_vec

def model4D_spatial_pdfs(ds,pdf_dim,vert_levels,horizontal_locs,fld,fldMean,bins_vector):
    """
    Using an a2e-mmc standard, xarrays-based, data structure of 
    4-dimensional model output with space-based quantities of interest,
    calculate probability distributions at specified vertical indices and 
    streamwise horizontal indices, aiccumulated over the time instances in ds.

    Usage
    ====
    ds : mmc-4D standard xarray DataSet 
        The raw standard mmc-4D data structure 
    pdf_dim : string 
        Dimension along which to calculate probability distribution functions
    vert_levels :
        vertical levels over which to caluclate probability distribution functions
    horizontal_locs : 
        horizontal (non-pdf_dim) locations at which to calculate probability distribution functions
    fld : string
	Name of the field in the dataset to calculate pdfs on
    fldMean : string
	Name of the mean of fld in the dataset 
    """

    from scipy.stats import skew,kurtosis
    print('Accumulating statistics over {:d} time-instances'.format(ds.dims['datetime']))
    sk_vec=np.zeros((len(vert_levels),len(horizontal_locs)))
    kurt_vec=np.zeros((len(vert_levels),len(horizontal_locs)))
    hist_cum = np.zeros((len(vert_levels),len(horizontal_locs),bins_vector.size-1))
    cnt_lvl = 0
    for level in vert_levels:
        cnt_i = 0
        for iLoc in horizontal_locs:
            dist=np.ndarray.flatten(((ds[fld]).isel(nz=level,nx=iLoc)-(ds[fldMean]).isel(nz=level,nx=iLoc)).values)
            sk_vec[cnt_lvl,cnt_i]=skew(dist)
            kurt_vec[cnt_lvl,cnt_i]=kurtosis(dist)
            cnt_i = cnt_i+1
        cnt_lvl = cnt_lvl+1
    cnt=0.0
    for it in range(ds.dims['datetime']):
        ##### If it seems like this is taking a long time, check progress occasionally by uncommenting 2-lines below
        #if int((it/ds.dims['datetime'])%10*100)%10 == 0:
        #    print('working...{:2d}% complete'.format(int((it/ds.dims['datetime'])%10*100)))
        cnt_lvl = 0
        for level in vert_levels:
            cnt_i = 0
            for iLoc in horizontal_locs:
                y = (ds[fld].isel(datetime=it,nz=level,nx=iLoc)-ds[fldMean].isel(datetime=it,nz=level,nx=iLoc))
                #y = np.ndarray.flatten(dist.isel(nz=level,nx=iLoc).values)
                hist,bin_edges=np.histogram(y, bins=bins_vector)
                if it == 0:
                    hist_cum[cnt_lvl,cnt_i,:] = hist
                else:
                    hist_cum[cnt_lvl,cnt_i,:] = hist_cum[cnt_lvl,cnt_i,:] + hist
                cnt_i = cnt_i+1
            cnt_lvl = cnt_lvl+1
        cnt = cnt+1.0

    return hist_cum, bin_edges, sk_vec, kurt_vec


def reference_lines(x_range, y_start, slopes, line_type='log'):
    '''
    This will generate an array of y-values over a specified x-range for
    the provided slopes. All lines will start from the specified
    location. For now, this is only assumed useful for log-log plots.
    x_range : array
        values over which to plot the lines (requires 2 or more values)
    y_start : float
        where the lines will start in y
    slopes : float or array
        the slopes to be plotted (can be 1 or several)
    '''
    if type(slopes)==float:
        y_range = np.asarray(x_range)**slopes
        shift = y_start/y_range[0]
        y_range = y_range*shift
    elif isinstance(slopes,(list,np.ndarray)):
        y_range = np.zeros((np.shape(x_range)[0],np.shape(slopes)[0]))
        for ss,slope in enumerate(slopes):
            y_range[:,ss] = np.asarray(x_range)**slope
            shift = y_start/y_range[0,ss]
            y_range[:,ss] = y_range[:,ss]*shift
    return(y_range)


def estimate_ABL_height(T=None,Tw=None,uw=None,sanitycheck=True,**kwargs):
    """Estimate the height of the atmospheric boundary layer (ABL) with
    a variety of methods. The recommended approach is Tw during convective
    conditions and uw during stable conditions.

    Parameters
    ==========
    All inputs should be multi-indexed pandas Series, unless otherwise
    specified, with the index levels being datetime (0) and height (1).
    T : 
        Estimate the height of the ABL from the potential temperature
        (T) profile. The height is given as where the gradient of
        potential temperature is smaller than some threshold. Additional
        parameters:
        - threshold (default=0.065 K/m)
            Temperature gradient that describes the inversion layer.
        - zmin (default=0 m)
            Height above ground to start looking for the inversion.
    Tw :
        Estimate the height of the convective boundary layer from the
        instantaneous minima in vertical heat flux <T'w'>.
    uw :
        Estimate the height of the stable boundary layer as the height
        at which the tangential turbulent stress vanishes, where <u'w'>
        is the magnitude of the horizontal components. Additional
        parameters:
        - cutoff (default=0.05)
            Stress value from which the zero-stress height is
            extrapolated.
        Ref: Kosovic & Curry, JAS (2000)
    sanitycheck : boolean, optional
        Perform additional sanity checks (if any).
    kwargs : optional keywords
        Additional method-specific parameters.
    """
    ablh = None
    if T is not None:
        threshold = kwargs.get('threshold',0.065) # [K/m]
        zmin = kwargs.get('zmin',0) # [m]
        T = T.loc[T.index.get_level_values(1) >= zmin].unstack()
        heights = np.array(T.columns)
        dz = np.diff(heights)
        newheights = (heights[:-1] + heights[1:]) / 2
        dTdz = T.diff(axis=1).drop(columns=[heights[0]])
        dTdz.columns = newheights
        dTdz = dTdz.divide(dz, axis=1)
        ablh = dTdz[dTdz >= threshold].apply(lambda s: s.first_valid_index(), axis=1)
    elif Tw is not None:
        ablh = Tw.unstack().idxmin(axis=1)
        if sanitycheck:
            Tw_at_ablh = Tw.loc[[(t,z) for t,z in ablh.iteritems()]]
            assert np.all(Tw_at_ablh <= 0)
    elif uw is not None:
        # assume the turbulent stress maxima occur near the surface and
        # equal u*
        unstacked = uw.unstack()
        ustar = unstacked.max(axis=1)
        uw_norm = unstacked.divide(ustar, axis=0)
        cutoff = kwargs.get('cutoff',0.05)
        z_near0 = uw_norm[uw_norm <= cutoff].apply(lambda s: s.first_valid_index(), axis=1)
        z_near0 = z_near0.fillna(method='bfill').fillna(method='ffill')
        # get near-zero value of uw for extrapolation
        uw_norm = uw_norm.stack(dropna=False)
        uw_norm_near0 = uw_norm.loc[[(t,z) for t,z in z_near0.iteritems()]]
        if sanitycheck:
            assert np.all(uw_norm_near0.loc[~pd.isna(uw_norm_near0)] >= 0) 
        # extrapolate
        ablh = z_near0 / (1 - uw_norm_near0)
        ablh = ablh.reset_index(level=1)[0]
    else:
        raise ValueError('No valid inputs provided')
    ablh.name = 'ABLheight'
    return ablh

def get_nc_file_times(f_dir,
                      f_grep_str,
                      decode_times=True,
                      time_dim='time',
                      get_time_from_fname=False,
                      f_split=[],
                      time_pos=[],
                      time_fmt='%Y%m%d'):
    '''
    Get times from netCDF files and returns dictionary of times associated with the file that it's in:
    dict{'time' : 'file_path'}. This uses xarray to find the times.
    
    This is useful for when you're using different NetCDF datasets that have non-uniform time 
    conventions (i.e., some have 1 time per file, others have multiple.)
    
    f_dir : str
        path to files
    f_grep_str : str
        string to grep the file - should include '*'
    decode_times : bool
        (Default=True) If you want xarray to decode the times (if xarray cannot decode the time, set 
        this to False)
    time_dim : str
        (Default='time') time dimension name
    get_time_from_fname : bool
        if there is no time in the file, you can use the following options to parse the file name
        f_split : list
            The strings (in order) for which the file name should be parsed
        time_pos : list (same dimension as f_split)
            After the string has been split, which index should be taken. Must be same dimension and
            order as f_split to work properly
        time_fmt : str
            (Default '%Y%m%d') Format for the datetime in file.
    '''
    files = sorted(glob.glob('{}{}'.format(f_dir,f_grep_str)))
    num_files = len(files)
    file_times = {}

    for ff,fname in enumerate(files): 
        ncf = xr.open_dataset(fname,decode_times=decode_times)

        #ncf = ncdf(fname,'r')

        if get_time_from_fname:
            assert f_split != [], 'Need to specify how to split the file name.'
            assert time_pos != [], 'Need to specify index of time string after split.'
            f_name = fname.replace(f_dir,'')
            assert len(f_split) == len(time_pos), 'f_split (how to parse the file name) and time_pos (index of time string is after split) must be same size.'
            for split,pos in zip(f_split,time_pos):
                f_name = f_name.split(split)[pos]
            
            f_time = [datetime.strptime(f_name,time_fmt)]
        else:
            if not decode_times:
                nc_times = ncf[time_dim][:].data
                f_time = []
                for ff,nc_time in enumerate(nc_times):
                    time_start = pd.to_datetime(ncf[time_dim].units.replace('seconds since ',''))            
                    f_time.append(datetime(time_start.year, time_start.month, time_start.day) + timedelta(seconds=int(nc_time)))
            else:
                f_time = ncf[time_dim].data

        for ft in f_time:
            ft = pd.to_datetime(ft)
            file_times[ft] = fname
    return (file_times)

def calc_spectra(data,
                 var_oi=None,
                 spectra_dim=None,
                 average_dim=None,
                 level_dim=None,
                 level=None,
                 window='hamming',
                 number_of_windows=1,
                 window_length=None,
                 window_overlap_pct=None,
                 detrend='constant',
                 tstart=None,
                 interval=None
                 ):
    
    '''
    Calculate spectra using the Welch function. This code uses the 
    power_spectral_density function from helper_functions.py. This function
    accepts either xarray dataset or dataArray, or pandas dataframe. Dimensions
    must be 4 or less (time, x, y, z). Returns a xarray dataset with the PSD of
    the variable (f(average_dim, level, frequency/wavelength)) and the frequency 
    or wavelength variables. Averages of the PSD over time or space can easily
    be done with xarray.Dataset.mean(dim='[dimension_name]').
    
    Parameters
    ==========
    data : xr.Dataset, xr.DataArray, or pd.dataframe
        The data that spectra should be calculated over
    var_oi : str
        Variable of interest - what variable should PSD be computed from.
    spectra_dim : str
        Name of the dimension that the variable spans for spectra to be 
        computed. E.g., if you want time spectra, this should be something like
        'time' or 'datetime', if you want spatial spectra, this should be 'x' or 
        'y' (or for WRF, 'south_north' / 'west_east')
    average_dim : str
        Which dimension should be looped over for averaging. Name should be
        similar to what is described in spectra_dim
    level_dim : str (optional)
        If you have a third dimension that you want to loop over, specify the
        dimension name here. E.g., if you want to calculate PSD at several
        heights, level_dim = 'height_dim'
    level : list, array, int (optional)
        If there is a level_dim, what levels should be looped over. Default is 
        the length of level_dim.
    window : 'hamming' or specific window (optional)
        What window should be used for the PSD calculation? If None, no window
        is used in the Welch function (window is all 1's).
    number_of_windows : int (optional)
        Number of windows - determines window length as signal length / int
    window_length : int or str (optional)
        Alternative to number_of_windows, you can directly specify the length
        of the windows as an integer or as a string to be converted to a 
        time_delta. This will overwrite number_of_windows. If using time_delta,
        the window_length cannot be shorter than the data frequency.
    overlap_percent : int (optional)
        Percentage of data overlap with respect to window length.
    detrend : str (optional)
        Should the data be detrended (constant, linear, etc.). See Welch 
        function for more details.
    tstart : datetime (optional)
        If calculating the spectra over only a portion of the data, when will
        the series start (only available for timeseries at the moment).
    interval : str (optional)
        If calculating the spectra over only a portion of the data, how long
        of a segment is considered (only available for timeseries at the 
        moment).
        
        
    Example Call
    ============
    
    psd = calc_spectra(data,                       # data read in with xarray 
                       var_oi='W',                # PSD of 'W' to be computed
                       spectra_dim='west_east',     # Take the west-east line
                       average_dim='south_north',  # Average over north/south
                       level_dim='bottom_top_stag', # Compute over each level
                       level=None)    # level defaults to all levels in array
    
    '''
    from scipy.signal.windows import hamming

    # Datasets, DataArrays, or dataframes
    if not isinstance(data,xr.Dataset):
        if isinstance(data,pd.DataFrame):
            data = data.to_xarray()
        elif isinstance(data,xr.DataArray):
            if data.name is None:
                data.name = var_oi
            data = data.to_dataset()
        else:
            raise ValueError('unsupported type: {}'.format(type(data)))
    
    # Get index for frequency / wavelength:
    spec_index = data.coords[spectra_dim]
    dX = (spec_index.data[1] - spec_index.data[0])
    if isinstance(dX,(pd.Timedelta,np.timedelta64)):
        dX = pd.to_timedelta(dX)#.total_seconds()
    else:
        dX = float(dX)

    # Window length specification:
    if window_length is not None:
        if (isinstance(window_length,str)):
            if isinstance(dX,(pd.Timedelta,np.timedelta64)):
                try:
                    dwindow = pd.to_timedelta(window_length)
                except:
                    raise ValueError('Cannot convert {} to timedelta'.format(window_length))
                    
                if dwindow < dX:
                    raise ValueError('window_length is smaller than data time spacing')
                nblock = int( dwindow/dX )
            else:
                raise ValueError('window_length given as timedelta, but spectra_dim is not datetime...')
        else:
            nblock = int(window_length)
    else:
        nblock = int((len(data[spectra_dim].data))/number_of_windows)
    
    # Create window:
    if window is None:
        window = np.ones(nblock)
    elif (window == 'hamming') or (window == 'hanning'):
        window = hamming(nblock, True) #Assumed non-periodic in the spectra_dim    
    
    # Calculate number of overlapping points:
    if window_overlap_pct is not None:
        if window_overlap_pct > 1:
            window_overlap_pct /= 100.0
        num_overlap = int(nblock*window_overlap_pct)
    else:
        num_overlap = None
    
    # Make sure 'level' is iterable:
    if level is None:
        if level_dim is not None:
            level = data[level_dim].data[:]
        else:
            level = [None]
    elif isinstance(level,(int,float)):
        level = [level]
    level = list(level)
    n_levels = len(level)

    if average_dim is None:
        average_dim_data = [None]
    else:
        average_dim_data = data[average_dim]
    
    for ll,lvl in enumerate(level):
        if lvl is not None:
            spec_dat_lvl = data.sel({level_dim:lvl},method='nearest')
            lvl = spec_dat_lvl[level_dim].data
        else:
            spec_dat_lvl = data.copy()
        for ad,avg_dim in enumerate(average_dim_data):
            if avg_dim is not None:
                spec_dat = spec_dat_lvl.sel({average_dim:avg_dim})
            else:
                spec_dat = spec_dat_lvl.copy()
            if len(list(spec_dat.dims)) > 1:
                dim_list = list(spec_dat.dims)
                dim_list.remove(spectra_dim)
                assert len(dim_list) == 1, 'There are too many dimensions... drop one of {}'.format(dim_list)
                assert len(spec_dat[dim_list[0]].data) == 1, 'Not sure how to parse this dimension, {}, reduce to 1 or remove'.format(dim_list)
                spec_dat = spec_dat.squeeze()
            for varn in list(spec_dat.variables.keys()):
                if (varn != var_oi) and (varn != spectra_dim):
                    spec_dat = spec_dat.drop(varn)
            
            spec_dat_df = spec_dat[var_oi].to_dataframe()
            
            psd = power_spectral_density(spec_dat_df,
                                         window_type=window,
                                         detrend=detrend,
                                         num_overlap=num_overlap,
                                         tstart=tstart,interval=interval)
            psd = psd.to_xarray()
            if avg_dim is not None:
                psd = psd.assign_coords({average_dim:1})
                psd[average_dim] = avg_dim.data            
                psd = psd.expand_dims(average_dim)

                if ad == 0:
                    psd_level = psd
                else:
                    psd_level = psd.combine_first(psd_level)
            else:
                psd_level = psd
                
        if level_dim is not None:
            psd_level = psd_level.assign_coords({level_dim:1})
            psd_level[level_dim] = lvl#.data            
            psd_level = psd_level.expand_dims(level_dim)

        if ll == 0:
            psd_f = psd_level
        else:
            psd_f = psd_level.combine_first(psd_f)
    return(psd_f)


def calcTRI(hgt,window):
    '''
    Terrain Ruggedness Index
    Riley, S. J., DeGloria, S. D., & Elliot, R. (1999). Index that 
        quantifies topographic heterogeneity. intermountain Journal 
        of sciences, 5(1-4), 23-27.
    
    hgt : array
        Array of heights over which TRI will be calculated
    window : int
        Length of window in x and y direction. Must be odd.
    '''
    
    # Window setup:
    assert (window/2.0) - np.floor(window/2.0) != 0.0, 'window must be odd...'
    Hwindow = int(np.floor(window/2))
    
    # Type and dimension check:
    if isinstance(hgt,(xr.Dataset,xr.DataArray)):
        hgt = hgt.data    
    assert len(np.shape(hgt)) == 2, 'hgt must be 2-dimensional. Currently has {} dimensions'.format(len(np.shape(hgt)))
    
    ny,nx = np.shape(hgt)
    tri = np.zeros((ny,nx))
    # Loop over all cells within bounds of window:
    for ii in range(Hwindow+1,nx-Hwindow-1):
        for jj in range(Hwindow+1,ny-Hwindow-1):
            hgt_window = hgt[jj-Hwindow:jj+Hwindow+1,ii-Hwindow:ii+Hwindow+1]
            tri[jj,ii] = (np.sum((hgt_window - hgt[jj,ii])**2.0))**0.5
    return tri


def calcVRM(hgt,window):
    '''
    Vector Ruggedness Measure
    Sappington, J. M., Longshore, K. M., & Thompson, D. B. (2007). 
        Quantifying landscape ruggedness for animal habitat analysis: 
        a case study using bighorn sheep in the Mojave Desert. The 
        Journal of wildlife management, 71(5), 1419-1426.
    
    hgt : array
        Array of heights over which TRI will be calculated
    window : int
        Length of window in x and y direction. Must be odd.
    '''
    import richdem as rd
    
    # Window setup:
    assert (window/2.0) - np.floor(window/2.0) != 0.0, 'window must be odd...'
    Hwndw = int(np.floor(window/2))

    # Type and dimension check:
    if isinstance(hgt,(xr.Dataset,xr.DataArray)):
        hgt = hgt.data    
    assert len(np.shape(hgt)) == 2, 'hgt must be 2-dimensional. Currently has {} dimensions'.format(len(np.shape(hgt)))
    ny,nx = np.shape(hgt)

    # Get slope and aspect:
    hgt_rd = rd.rdarray(hgt, no_data=-9999)
    rd.FillDepressions(hgt_rd, in_place=True)
    slope  = rd.TerrainAttribute(hgt_rd, attrib='slope_riserun')
    aspect = rd.TerrainAttribute(hgt_rd, attrib='aspect')
    
    # Calculate vectors:
    vrm = np.zeros((ny,nx))
    rugz   = np.cos(slope*np.pi/180.0)
    rugdxy = np.sin(slope*np.pi/180.0)
    rugx   = rugdxy*np.cos(aspect*np.pi/180.0)
    rugy   = rugdxy*np.sin(aspect*np.pi/180.0)
    
    # Loop over all cells within bounds of window:
    for ii in range(Hwndw+1,nx-Hwndw-1):
        for jj in range(Hwndw+1,ny-Hwndw-1):
            vrm[jj,ii] = 1.0 - np.sqrt(\
                    np.sum(rugx[jj-Hwndw:jj+Hwndw+1,ii-Hwndw:ii+Hwndw+1])**2.0 + \
                    np.sum(rugy[jj-Hwndw:jj+Hwndw+1,ii-Hwndw:ii+Hwndw+1])**2.0 + \
                    np.sum(rugz[jj-Hwndw:jj+Hwndw+1,ii-Hwndw:ii+Hwndw+1])**2.0)/float(window**2)
    return vrm