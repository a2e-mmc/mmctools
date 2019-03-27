"""
Helper functions for calculating standard meteorological quantities
"""
import numpy as np
import pandas as pd

def T_to_Tv(T,p=None,RH=None,w=None,q=None,e=None,pd=None,
            epsilon=0.622,verbose=False):
    """Convert moist air temperature [K] to virtual temperature [K].
    
    Formulas based on given total pressure (p, mbar) and relative
    humidity (RH, %); mixing ratio (w, kg/kg); or partial pressures of
    water vapor and dry air (e, pd). Epsilon is the ratio of the gas
    constants of dry air to water vapor.
    """
    if (p is not None) and (RH is not None):
        # saturation vapor pressure of water, e_s [mbar]
        #   Eqn 10 from Bolton (1980), Mon. Weather Rev., Vol 108
        T_degC = T - 273.15
        es = 6.112 * np.exp(17.67*T_degC / (T_degC + 243.5))
        if verbose:
            # from https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
            # - where do these equations come from?
            # - is it 237.3 or 237.7?
            es_est = 6.11 * 10**(7.5*T_degC/(237.3+T_degC))
            print('e_s(T) =',es,'est',es_est)
        # saturation mixing ratio, ws [-]
        #   Wallace & Hobbs, Eqn 3.63
        ws = epsilon * es / (p - es)
        if verbose:
            print('w_s(T,p) =',ws,'est',epsilon*es/p)
        # mixing ratio, w, from definition of relative humidity
        w = (RH/100.) * ws
        # specific humidity, q (not needed at this point)
        if verbose:
            q = w / (1+w)
            print('q(T,p,RH) =',q)
        # alternative calculation, based on dewpoint
        if verbose:
            # from https://www.weather.gov/media/epz/wxcalc/virtualTemperature.pdf
            # - note this is Wallace & Hobbs' eqn 3.16, also implemented below
            # - note the expression for vapor pressure is similar to the
            #   saturation vapor pressure expression above, with Td instead of T
            # - is it 237.3 or 237.7?
            Td_degC = 237.3 * np.log(es*RH/611) / (7.5*np.log(10) - np.log(es*RH/611))
            print('Td(T,RH) =',Td_degC)
            e_est = 6.11 * 10**(7.5*Td_degC/(237.7+Td_degC))
            denom = 1 - 0.379 * e_est/p
            Tv_est = T / denom
            print('est Tv(T,p,RH) =',Tv_est)
        # Using Wallace & Hobbs, Eqn 3.59
        if verbose:
            print('Tv(T,p,RH) ~=',T*(1+0.61*w))
        Tv = T * (w/epsilon + 1) / (1 + w)
    elif q is not None:
        w = q / (1-q)
        # Using Wallace & Hobbs, Eqn 3.59
        Tv = T * (w/epsilon + 1) / (1 + w)
    elif w is not None:
        # Using Wallace & Hobbs, Eqn 3.59
        Tv = T * (w/epsilon + 1) / (1 + w)
    elif (e is not None) and (pd is not None):
        # Wallace & Hobbs, Eqn 3.16
        Tv = T / (1 - e/pd*(1-epsilon))
    else:
        print('Specify (RH,p) or (q,) or (w,) or (e,pd)')
        Tv = None
    return Tv


def Ts_to_Tv(Ts,**kwargs):
    """TODO: Convert sonic temperature [K] to virtual temperature [K].
    """
    

def covariance(a,b,interval):
    """Calculate covariance between two series (with datetime index) in
    the specified interval, where the interval is defined by a pandas
    offset string
    (http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects).

    Example:
        heatflux = covariance(df['Ts'],df['w'],'10min')
    """
    a_mean = a.rolling(interval).mean()
    b_mean = b.rolling(interval).mean()
    ab_mean = (a*b).rolling(interval).mean()
    return ab_mean - a_mean*b_mean

