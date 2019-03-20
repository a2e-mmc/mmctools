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
    Ts=lambda Ts: 273.15 + Ts/100, # sonic temperature, 100*[deg C]
    time='%H:%M:%S',
)

RMYoung_05106 = OrderedDict(
    ID=None,
    year='%Y',  # 4-digit year
    day='%j',  # julian day number
    time='%H%M',
    HorizontalWind=1, # mean horizontal wind speed [m/s]
    wspd=1, # resultant mean wind speed [m/s]
    wdir=1, # resultant mean of wind direction == arctan(Ueast/Unorth) [deg]
    wdir_std=1, # stdev of wind direction [deg]
    T=lambda Ta: 273.15 + Ta, # air temperature [deg C]
    RH=1, # relative humidity [%]
    P=1, # barometric pressure (not corrected for sea level) [mb]
    SW_down=1, # downwelling SW solar radiation (400-1100nm) [W/m^2]
    T10X=1, # datalogger temperature [deg C]
    p10X=1, # datalogger power [V]
)

Gill_R3_50 = OrderedDict(
    v=1, # North-South, plus to North
    u=1, # East-West, plus to East
    w=1, # Vertical, plus upward
    Ts=lambda Ts: 273.15 + Ts, # virtual sonic temperature [deg C]
)

def read_data(fname, column_spec,
              height=None, multi_index=False,
              datetime_start='', datetime_start_format='',
              datetime=None, datetime_offset=None,
              start=pd.datetime(1990,1,1), end=pd.datetime.today(),
              **kwargs):
    """Read in data (e.g., output from a sonic anemometer) at a height
    on the met mast and standardize outputs

    Inputs
    ------
    fname : str
        Filename passed to pandas.read_csv()
    column_spec : OrderedDict
        Pairs of column names and data formats
    height : float, optional
        Height to associate with this data (column will be added)
    multi_index : bool, optional
        If height is specified, then return a pandas.DataFrame with a
        MultiIndex
    datetime_start : str, optional
        To specify datetime information missing from the data file
    datetime_start_format : str, optional
        If datetime_start is specified, then the format of the provided
        datetime string
    datetime : pandas.DateTimeIndex, optional
        If no date or time information are included in datafile, use
        this specified datetime series
    datetime_offset : float, optional
        Add a time offset (in seconds) that will be converted into a
        timedelta, e.g., for standardizing data that were averaged to
        the beginning or end of an interval
    start,end : str or datetime, optional
        Trim the data down to this specified time range
    **kwargs : optional
        Additional arguments to pass to pandas.read_csv()
    """
    columns = column_spec.keys()
    df = pd.read_csv(fname,names=columns,**kwargs)

    # standardize the data
    datetime_columns = []
    for col,fmt in column_spec.items():
        if fmt==1:
            continue
        elif callable(fmt):
            # apply function to column
            df[col] = df[col].apply(fmt)
        elif isinstance(fmt,float):
            # convert to standard units
            df[col] = df[col] / fmt
        elif isinstance(fmt,str):
            # collect datetime column names
            datetime_columns.append(col)
        elif fmt is None:
            df = df.drop(columns=col)
        else:
            raise TypeError('Unexpected column name/format:',(col,fmt))
    if (len(datetime_columns) == 0) and (datetime is None):
        raise InputError('No datetime data in file; need to specify datetime')
    elif (len(datetime_columns) > 0) and (datetime is not None):
        print('Note: datetime specified; datetime information in datafile ignored')

    # set up date/time column
    if datetime is not None:
        # use user-specified datetime
        df[datetime_name] = datetime
    elif datetime_name in datetime_columns:
        # we have complete information
        datetime_format = column_spec[datetime_name]
        df[datetime_name] = pd.to_datetime(df[datetime_name],
                                           format=datetime_format)
    elif (date_name in datetime_columns) and (time_name in datetime_columns):
        # we have separate date and time columns
        if datetime_start is not None:
            print('Ignored specified datetime_start')
        date_format = column_spec[date_name]
        time_format = column_spec[time_name]
        df[datetime_name] = pd.to_datetime(df[date_name]+df[time_name],
                                           format=date_format+time_format)
    else:
        # try to cobble together datetime information from all text columns
        # - convert datetime columns into string type (so that we can add them
        #   together) and make sure time strings didn't end up truncated as
        #   integers
        #   e.g. %H%M : 01:00 --> 100 (instead of '0100')
        test_strings = ['%H','%M','%S']
        for col in datetime_columns:
            if df[col].dtype == np.int64:
                fmt = column_spec[col]
                if any([s in fmt for s in test_strings]):
                    for strftime_str in test_strings:
                        fmt = fmt.replace(strftime_str,'00')
                    timestrlen = len(fmt)
                    # convert integer column data into zero-padded string
                    zeropadfmt = '{:0' + str(timestrlen) + '}'
                    df[col] = df[col].apply(lambda t: zeropadfmt.format(t))
                else:
                    # convert integer column, e.g., year(, month, day) into str
                    df[col] = df[col].astype(str)
        # - combine all datetime columns into a series
        datetime = datetime_start \
                + df[datetime_columns].apply(lambda cols: ''.join(cols), axis=1)
        # - combine all format strings
        datetime_format = datetime_start_format \
                + ''.join([column_spec[col] for col in datetime_columns])
        # - create datetime series
        df[datetime_name] = pd.to_datetime(datetime, format=datetime_format)
        df = df.drop(columns=datetime_columns)

    # add time offset, e.g., for standardizing data that were averaged to the
    # beginning/end of an interval
    if datetime_offset is not None:
        offset = pd.to_timedelta(datetime_offset,unit='s')
        df[datetime_name] += offset

    # trim datetime
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
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

    return df

