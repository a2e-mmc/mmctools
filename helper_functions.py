"""
Helper functions for calculating standard meteorological quantities
"""
import numpy as np
import pandas as pd

def T_to_Tv(T,p=None,RH=None,w=None,e=None,pd=None,epsilon=0.622,verbose=False):
    """Convert moist air temperature [K] to virtual temperature [K].
    
    Formulas based on given total pressure (p, mbar) and relative
    humidity (RH, %); mixing ratio (w, kg/kg); or partial pressures of
    water vapor and dry air (e, pd). Epsilon is the ratio of the gas
    constants of dry air to water vapor.
    """
    if (p is not None) and (RH is not None):
        # saturation vapor pressure of water, es [mbar]
        #   Eqn 10 from Bolton (1980), Mon. Weather Rev., Vol 108
        T_degC = T - 273.15
        es = 6.112 * np.exp(17.67*T_degC / (T_degC + 243.5))
        if verbose:
            print('e_s(T) =',es)
        # saturation mixing ratio, ws [-]
        ws = epsilon * es / (p - es)
        if verbose:
            print('w_s(T,p) =',ws,'est',epsilon*es/p)
        # mixing ratio, w, from definition of relative humidity
        w = (RH/100.) * ws
        # specific humidity, q (not needed at this point)
        if verbose:
            q = w / (1+w)
            print('q(T,p,RH) =',q)
        # Using Wallace & Hobbs, Eqn 3.59
        if verbose:
            print('Tv(T,p,RH) ~=',T*(1+0.61*w))
        Tv = T * (w/epsilon + 1) / (1 + w)
    elif w is not None:
        # Using Wallace & Hobbs, Eqn 3.59
        Tv = T * (w/epsilon + 1) / (1 + w)
    elif (e is not None) and (pd is not None):
        # Wallace & Hobbs, Eqn 3.16
        Tv = T / (1 - e/pd*(1-epsilon))
    else:
        print('Specify (RH,) or (w,) or (e,pd)')
        Tv = None
    return Tv
    
