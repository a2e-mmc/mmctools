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


