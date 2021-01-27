"""
# Observation Nudging in WRF

Based on "A Brief Guide to Observation Nudging in WRF" by Brian Reen (Feb 2016)
"""
import numpy as np
import pandas as pd

header = """ {datetime:14s}
  {lat:9.4f} {lon:9.4f} 
  {id:40s}   {namef:40s}   
  {platform:16s}  {source:16s}  {elevation:#8.0f}  {is_sound:>4s}  {bogus:>4s}  {meas_count:5d}
"""

MISSING = -888888.000
EARTHRELATIVE = 129 # wind specification (instead of grid-relative)


def boolstr(b):
    return str(b)[0].upper()


def write_header(f,datetime,lat,lon,stationID,
                 platform,elevation,
                 datatype='OTHER',source='',
                 levels=1,is_sound=True,bogus=False):
    """
    Parameters
    ----------
    f : output object
    datetime : datetime or pd.Timestamp
    lat,lon : float
    stationID : str
    platform : str
        Platform name--WRF checks for specific strings and outputs a
        warning if there's no match
    data type : str
        In some cases, WRF modifies `platform` based on this value
    source : str
        Source of observation, not used
    elevation : float
    is_sound : bool
        Is this a non-surface observation?
    is_bogus : bool
    levels : int
        Number of levels in sounding
    """
    datetime_str = pd.to_datetime(datetime).strftime('%Y%m%d%H%M%S')
    is_sound = boolstr(is_sound)
    bogus = boolstr(bogus)
    f.write(header.format(
        datetime=datetime_str,
        lat=lat, lon=lon,
        id=stationID, namef=datatype,
        platform=platform, source=source,
        elevation=elevation, is_sound=is_sound, bogus=bogus, meas_count=levels,
    ))

def _to_val_qc(val,defaultqc=0):
    if isinstance(val,tuple):
        return val
    elif val is None:
        return (MISSING, MISSING)
    else:
        assert isinstance(val, (int,float,np.float32,np.float64))
        return (val, defaultqc)


def write_surface(f,slp=None,ref_pres=None,height=0,
                  temperature=None,u=None,v=None,
                  rh=None,psfc=None,precip=None):
    """
    Inputs may be floats or tuples (value, qcflag). Wind components are
    assumed to be Earth-relative and all other values are assumed to
    have a QC flag of 0.

    Parameters
    ----------
    f : output object
    slp : float or tuple
        Sea-level pressure [Pa]
    ref_pres : float or tuple
        Reference pressure, not used? [Pa]
    height : float or tuple
        Height of observation, not used? Should match elevation [m MSL]
    temperature: float or tuple
        Temperature [K]
    u_met, v_met : float or tuple
        U,V wind component [m/s]
    rh_data : float or tuple
        Relative humidity [%]
    psfc_data : float or tuple
        Surface pressure [Pa]
    precip_data : float or tuple
        Precipitation information, unused?
    """
    line = ' '
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(slp))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(ref_pres))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(height))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(temperature))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(u,EARTHRELATIVE))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(v,EARTHRELATIVE))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(rh))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(psfc))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(precip))
    line += '\n'
    f.write(line)
    
def write_level(f,pressure=None,height=0,
                temperature=None,u=None,v=None,
                rh=None):
    """
    Inputs may be floats or tuples (value, qcflag). Wind components are
    assumed to be Earth-relative and all other values are assumed to
    have a QC flag of 0.

    Parameters
    ----------
    f : output object
    pressure : float or tuple
        Reference pressure, not used? [Pa]
    height : float or tuple
        Height of observation, not used? Should match elevation [m MSL]
    temperature: float or tuple
        Temperature [K]
    u_met, v_met : float or tuple
        U,V wind component [m/s]
    rh_data : float or tuple
        Relative humidity [%]
    """
    line = ' '
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(pressure))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(height))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(temperature))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(u,EARTHRELATIVE))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(v,EARTHRELATIVE))
    line += '{:11.3f} {:11.3f} '.format(*_to_val_qc(rh))
    line += '\n'
    f.write(line)
    
