"""
Data readers for meteorological towers

Based on https://github.com/NWTC/datatools/blob/master/metmast.py
"""
from collections import OrderedDict
import numpy as np
import pandas as pd

#
# TODO:
# - standardize names ("wspd" vs "windspeed" vs "speed", etc)
# - standardize units
# - standardize quantities (air temperature vs virtual temperature etc)

datetime_name = 'datetime'
date_name = 'date' # for building datetime from separate columns
time_name = 'time' # for building datetime from separate columns
height_name = 'height'
windspeed_name = 'wspd'
winddirection_name = 'wdir'

#
# Data descriptors
# ================
# Use OrderedDict to correctly associate columns with data. Data columns
# are identified as follows:
# - float: scaling factor applied to get the data units (default is 1)
# - callable: function to convert from data to standardized data
# - str: datetime format
# - None: ignore column

Metek_USA1 = OrderedDict(
    v=100, # units are cm/s, i.e., 100*[m/s]
    u=100, 
    w=100,
    T=lambda Ts: 273.15 + Ts/100, # sonic temperature, 100*[deg C]
    time='%H:%M:%S',
)

RMYoung_05106 = OrderedDict(
    ID=None,
    year='%Y',  # 4-digit year
    day='%j',  # julian day number
    time='%H%M',
    HorizontalWind=1, # mean horizontal wind speed
    wspd=1,
    wdir=1,
    wdir_std=1,
    T=lambda Ta: 273.15 + Ta, # air temperature [deg C]
    RH=1,
    P=lambda p: 1.0*p, # barometric pressure, not corrected for sea-level [mb]
    SW_down=1,
)


def read_data(fname, column_spec,
              height=None, multi_index=False,
              datetime_offset=None,
              start=pd.datetime(1990,1,1), end=pd.datetime.today(),
              **kwargs):
    """Read in data (e.g., output from a sonic anemometer) at a height
    on the met mast and standardize outputs
    """
    df = pd.read_csv(fname)

    # set up date/time column
    if datetime_name in column_spec.keys():
        # we have complete information
        datetime_format = column_spec[datetime_name]
        df[datetime_name] = pd.to_datetime(df[datetime_name],
                                           format=datetime_format)
        have_datetime = True
    elif time_name in column_spec.keys():
        # we have time (and optionally, date) information in separate columns
        time_format = column_spec[time_name]
        time = pd.to_timedelta(df[time_name], format=time_format)
        if date_name in column_spec.keys():
            assert(datetime_offset is None)
            date_format = column_spec[date_name]
            date = pd.to_datetime(df[date_name], format=date_format)
            df[datetime_name] = date + time
        elif datetime_offset is not None:
            df[datetime_name] = datetime_offset + time
            have_datetime = True
        else:
            print('Specify datetime_offset for complete datetime')
            df[datetime_name] = time
            have_datetime = False
    else:
        print('No datetime in column spec')

    # trim datetime
    if have_datetime:
        datetime_range = (df[datetime_name] >= start) & (df[datetime_name] <= end)
        df = df.loc[datetime_range]

    # set height column (and multi-index)
    if height is not None:
        df[height_name] = height
    if height and multi_index:
        df = df.set_index([datetime_name,height_name])
    else:
        df = df.set_index(datetime_name)

    # standard calculations
    if not windspeed_name in column_spec.keys():
        # assume we have all velocity components
        df[windspeed_name] = np.sqrt(df['u']**2 + df['v']**2 + df['w']**2)
    if not winddirection_name in column_spec.keys():
        # assume we have u,v velocity components
        df[winddirection_name] = np.degrees(np.arctan2(-df['u'],-df['v']))
        df.loc[df[winddirection_name] < 0, winddirection_name] += 360.0

