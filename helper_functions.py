"""
Helper functions for calculating standard meteorological quantities
"""
import numpy as np
import pandas as pd


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
    if not all(velcomp in df.columns for velcomp in [u,v]):
        print(('velocity components u/v not found; '
               'set u and/or v to calculate wind speed/direction'))
    else:
        wspd = np.sqrt(df[u]**2 + df[v]**2)
        wdir = 180. + np.degrees(np.arctan2(df[u], df[v]))
        return wspd, wdir

def calc_uv(df,wspd='wspd',wdir='wdir'):
    """Calculate velocity components from wind speed and direction.
    """
    if not all(windvar in df.columns for windvar in [wspd,wdir]):
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

    Example:
        heatflux = covariance(df['Ts'],df['w'],'10min')
    """
    # handle multiindices
    have_multiindex = False
    if isinstance(a.index, pd.MultiIndex):
        assert len(a.index.levels) == 2
        # assuming levels 0 and 1 are time and height, respectively
        a = a.unstack()
        have_multiindex = True
    if isinstance(b.index, pd.MultiIndex):
        assert len(b.index.levels) == 2
        # assuming levels 0 and 1 are time and height, respectively
        b = b.unstack()
        have_multiindex = True
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


def power_spectral_density(df,time_intervals=None,window_size='10min',
                           window_type='hanning',detrend='linear',scaling='density'):
    """
    Calculate power spectral density using welch method in the
    specified time interval(s). The spectrum is calculated for every
    column of the dataframe.

    Notes:
    - time_intervals is a list of N+1 timestamps denoting the start and
      end of N time intervals. The timestamp in the output corresponds to
      the start of the time interval. If time_intervals is not specified,
      the spectrum is calculated for one time interval spanning the entire
      time period
    - Spectra may be calculated at multiple heights at once by
      inputting multi-index dataframes (with height being the second
      index level)
    - Output is a multi-index dataframe with an additional index level
      corresponding to the frequency
    """
    from scipy.signal import welch
    timevalues = df.index.get_level_values(0)

    # Determine sampling rate and samples per window
    if isinstance(timevalues,pd.DatetimeIndex):
        # Divide by pd.timedelta to get a float
        dts = np.diff(timevalues.unique())/pd.to_timedelta(1,'s')
    else:
        # It is assumed that time is specified in seconds
        dts = np.diff(timevalues.unique())

    dt  = dts[0]
    nperseg = int( pd.to_timedelta(window_size)/pd.to_timedelta(dt,'s') )
    assert(np.allclose(dts,dt)),\
        'Timestamps must be spaced equidistantly'

    # If no time_interval specified, use entire signal
    if time_intervals is None:
        time_intervals = [timevalues[0], timevalues[-1]]
    # If time specified as string, convert to DatetimeIndex
    elif isinstance(time_intervals[0],str):
        time_intervals = pd.to_datetime(time_intervals)

    df_list = []

    # multiple heights
    if isinstance(df.index, pd.MultiIndex):
        assert len(df.index.levels) == 2
        # assuming levels 0 and 1 are time and height, respectively

        for t in range(np.asarray(time_intervals).size-1):
            inrange = (timevalues >= time_intervals[t]) & (timevalues <= time_intervals[t+1])
            for hgt in df.index.get_level_values(1).unique():
                spectrum = {}
                for col in df.columns:
                    f,P = welch( df.loc[(inrange,hgt),col], fs=1./dt, nperseg=nperseg,
                        detrend=detrend,window=window_type,scaling=scaling)    
                    spectrum[col] = P
                spectrum['frequency'] = f
                spectrum = pd.DataFrame(spectrum)
                spectrum['datetime'] = time_intervals[t]
                spectrum['height'] = hgt
                df_list.append(spectrum)
        # New index levels
        keys = ['datetime','height','frequency']

    # single height
    else:
        for t in range(np.asarray(time_intervals).size-1):
            inrange = (timevalues >= time_intervals[t]) & (timevalues < time_intervals[t+1])
            spectrum = {}
            for col in df.columns:
                f,P = welch( df.loc[inrange,col], fs=1./dt, nperseg=nperseg,
                    detrend=detrend,window=window_type,scaling=scaling)    
                spectrum[col] = P
            spectrum['frequency'] = f
            spectrum = pd.DataFrame(spectrum)
            spectrum['datetime'] = time_intervals[t]
            df_list.append(spectrum)
        # New index levels
        keys = ['datetime','frequency']

    df_spectra = pd.concat(df_list)
    df_spectra = df_spectra.set_index(keys).sort_index()
    return df_spectra
    

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


