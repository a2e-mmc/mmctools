"""
Helper functions for calculating standard meteorological quantities
"""
import numpy as np
import pandas as pd


# constants
epsilon = 0.622 # ratio of molecular weights of water to dry air


def e_s(T, celsius=False, model='Bolton1980'):
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
    if model == 'Bolton1980':
        # Eqn 10 from Bolton (1980), Mon. Weather Rev., Vol 108
        es = 6.112 * np.exp(17.67*T_degC / (T_degC + 243.5))
    elif model == 'NWS':
        # From National Weather Service
        # https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
        # - where do these equations come from?
        # - is it 237.3 or 237.7?
        es = 6.11 * 10**(7.5*T_degC/(237.3+T_degC))
    else:
        raise ValueError('Unknown model: {:s}'.format(model))
    return es


def T_d(T, RH, celsius=False, model='NWS'):
    """Calculate the dewpoint temperature, $T_d$, from air temperature
    and relative humidity [%]. If celsius is True, input and output 
    temperatures are in degrees Celsius; otherwise, inputs and outputs
    are in Kelvin.
    """
    if model == 'NWS':
        if not celsius:
            T -= 273.15
        # from https://www.weather.gov/media/epz/wxcalc/virtualTemperature.pdf
        # - note the expression for vapor pressure is the saturation vapor
        #   pressure expression, with Td instead of T
        # - should the coefficient be 237.3 or 237.7?
        es = e_s(T, model='NWS')
        e = RH/100. * es
        denom = 7.5*np.log(10) - np.log(e/6.11)
        Td = 237.3 * np.log(e/6.11) / denom
        if not celsius:
            Td += 273.15
    else:
        raise ValueError('Unknown model: {:s}'.format(model))
    return Td


def T_to_Tv(T,p=None,RH=None,e=None,w=None,Td=None,
            verbose=False):
    """Convert moist air temperature [K] to virtual temperature [K].
    
    Formulas based on given total (or "station") pressure (p [mbar]) and
    relative humidity (RH [%]); mixing ratio (w [kg/kg]); or partial
    pressures of water vapor and dry air (e, pd [mbar]); or dewpoint
    temperature (Td [K]).
    """
    if (p is not None) and (RH is not None):
        # saturation vapor pressure of water, e_s [mbar]
        T_degC = T - 273.15
        es = e_s(T_degC)
        if verbose:
            # sanity check!
            es_est = e_s(T_degC, model='NWS')
            print('e_s(T) =',es,'est',es_est)
        # saturation mixing ratio, ws [-]
        #   Wallace & Hobbs, Eqn 3.63
        ws = epsilon * es / (p - es)
        if verbose:
            print('w_s(T,p) =',ws,'est',epsilon*es/p)
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
        # From National Weather Service
        # https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
        # - where do these equations come from?
        # - is it 237.3 or 237.7?
        Td_degC = Td - 273.15
        e = 6.11 * 10**(7.5*Td_degC/(237.7+Td_degC))
        # Calculate from definition of virtual temperature
        Tv = T_to_Tv(T,e=e,p=p)
    else:
        raise ValueError('Specify (RH,p) or (e,p) or (w,), or (Td,p)')
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

