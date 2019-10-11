"""
Helper functions for calculating standard meteorological quantities
"""
import numpy as np
import pandas as pd
import xarray as xr


# constants
epsilon = 0.622 # ratio of molecular weights of water to dry air


def e_s(T, celsius=False, model='Tetens'):
    """Calculate the saturation vapor pressure of water, $e_s$ [mb]
    given the air temperature.
    """
    if celsius:
        # input is deg C
        T_degC = T
        T = T + 273.15
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


def w_s(T,p,celsius=False):
    """Calculate the saturation mixing ratio, $w_s$ [kg/kg] given the
    air temperature and station pressure [mb].
    """
    es = e_s(T,celsius)
    # From Wallace & Hobbs, Eqn 3.63
    return epsilon * es / (p - es)


def T_to_Tv(T,p=None,RH=None,e=None,w=None,Td=None,
            celsius=False,verbose=False):
    """Convert moist air temperature to virtual temperature.
    
    Formulas based on given total (or "station") pressure (p [mbar]) and
    relative humidity (RH [%]); mixing ratio (w [kg/kg]); or partial
    pressures of water vapor and dry air (e, pd [mbar]); or dewpoint
    temperature (Td).
    """
    if celsius:
        T_degC = T
        T += 273.15
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
        # Using Wallace & Hobbs, Eqn 3.59
        if verbose:
            # sanity check!
            print('Tv(T,p,RH) ~=',T*(1+0.61*w))
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
        Td_degC = Td
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


def covariance(a,b,interval='10min',resample=False):
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
        ab_mean = (a*b).resample(interval).mean()
    else:
        a_mean = a.rolling(interval).mean()
        b_mean = b.rolling(interval).mean()
        ab_mean = (a*b).rolling(interval).mean()
    cov = ab_mean - a_mean*b_mean
    if have_multiindex:
        return cov.stack()
    else:
        return cov


def power_spectral_density(df,tstart=None,interval=None,window_size='10min',
                           window_type='hanning',detrend='linear',scaling='density'):
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
    nperseg = int( pd.to_timedelta(window_size)/pd.to_timedelta(dt,'s') )
    assert(np.allclose(dts,dt)),\
        'Timestamps must be spaced equidistantly'

    # If input is series, convert to dataframe
    if isinstance(df,pd.Series):
        df = df.to_frame()

    spectra = {}
    for col in df.columns:
        f,P = welch( df.loc[inrange,col], fs=1./dt, nperseg=nperseg,
            detrend=detrend,window=window_type,scaling=scaling)    
        spectra[col] = P
    spectra['frequency'] = f
    return pd.DataFrame(spectra).set_index('frequency')
    

def power_law(z,zref=80.0,Uref=8.0,alpha=0.2):
    return Uref*(z/zref)**alpha

def fit_power_law_alpha(z,U,zref=80.0,Uref=8.0):
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

def model4D_calcQOIs(ds,mean_dim):
    """
    Augment an a2e-mmc standard, xarrays-based, data structure of 
    4-dimensional model output with space-based quantities of interest

    Usage
    ====
    ds : mmc-4D standard xarray DataSet 
        The raw standard mmc-4D data structure 
    mean_dim : string 
        Dimension along which to calculate mean and fluctuating (perturbation) parts
    """

    dim_keys = [*ds.dims.keys()]
    ds_means = ds.mean(dim=mean_dim)
    ds_perts = ds-ds_means

    ds['uMean'] = ds_means['u']
    ds['vMean'] = ds_means['v']
    ds['wMean'] = ds_means['w']
    ds['pMean'] = ds_means['p']
    ds['thetaMean'] = ds_means['theta']
    ds['UMean'] = ds_means['wspd']
    ds['UdirMean'] = ds_means['wdir']

    ds['uu'] = ds_perts['u']**2
    ds['vv'] = ds_perts['v']**2
    ds['ww'] = ds_perts['w']**2
    ds['uv'] = ds_perts['u']*ds_perts['v']
    ds['uw'] = ds_perts['u']*ds_perts['w']
    ds['vw'] = ds_perts['v']*ds_perts['w']
    ds['wth'] = ds_perts['w']*ds_perts['theta']
    ds['UU'] = ds_perts['wspd']**2
    ds['Uw'] = ds_perts['wspd']**2
    ds['TKE'] = 0.5*np.sqrt(ds['UU']+ds['ww'])
    return ds

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
                if it is 0:
                    Puuf_cum[cnt_lvl,cnt_i,:] = Pxxf
                else:
                    Puuf_cum[cnt_lvl,cnt_i,:] = Puuf_cum[cnt_lvl,cnt_i,:] + Pxxf
                    cnt_i = cnt_i+1
            cnt_lvl = cnt_lvl + 1
        cnt = cnt+1.0
    Puuf = 2.0*(1.0/cnt)*Puuf_cum[:,:,:(np.floor(ds.dims['ny']/2).astype(int))]   ###2.0 is to account for the dropping of the negative side of the FFT 
    f = f[:(np.floor(ds.dims['ny']/2).astype(int))]

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
                if it is 0:
                    Puuf_cum[cnt_lvl,cnt_i,:] = Pxxf
                else:
                    Puuf_cum[cnt_lvl,cnt_i,:] = Puuf_cum[cnt_lvl,cnt_i,:] + Pxxf
                    cnt_i = cnt_i+1
            cnt_lvl = cnt_lvl + 1
        cnt = cnt+1.0
    Puuf = 2.0*(1.0/cnt)*Puuf_cum[:,:,:(np.floor(ds.dims['ny']/2).astype(int))]  ###2.0 is to account for the dropping of the negative side of the FFT
    f = f[:(np.floor(ds.dims['ny']/2).astype(int))]

    return f,Puuf

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
                if it is 0:
                    hist_cum[cnt_lvl,cnt_i,:] = hist
                else:
                    hist_cum[cnt_lvl,cnt_i,:] = hist_cum[cnt_lvl,cnt_i,:] + hist
                cnt_i = cnt_i+1
            cnt_lvl = cnt_lvl+1
        cnt = cnt+1.0

    return hist_cum, bin_edges, sk_vec, kurt_vec
