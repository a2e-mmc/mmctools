"""
Data readers for meteorological towers

Based on https://github.com/NWTC/datatools/blob/master/metmast.py
"""
import os
import inspect
from collections import OrderedDict
import numpy as np
import pandas as pd

# TODO: Decide on
# - standardized units
# - standardized quantities (air temperature vs virtual temperature etc)
# - what to do with extra nonstandard quantities
# - should timestamps correspond to the beginning of statistics interval

datetime_name = 'datetime'
date_name = 'date' # for building datetime from separate columns
time_name = 'time' # for building datetime from separate columns
height_name = 'height'
windspeed_name = 'wspd'
winddirection_name = 'wdir'
sonictemperature_name = 'Ts'

standard_output_columns = [
    datetime_name,
    height_name,
    windspeed_name,
    winddirection_name
]


# Data descriptors
# ================
# Use OrderedDict to correctly associate columns with data. Data columns
# are identified as follows:
# - float: scaling factor applied to get the data units (default is 1)
# - callable: function to convert from data to standardized data
# - str: datetime format
# - None: ignore column

Metek_USA1 = OrderedDict(
    v=100, # units are [cm/s], i.e., 100*[m/s]
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
    qc=1, # basic quality control code: 0 - OK, 1 - sonic bad data code, 2 - broken data line, and 3 - missed line
)


def read_data(fpath, column_spec,
              height=None, multi_index=True,
              datetime_start='', datetime_start_format='',
              data_freq=None, max_data_rows=None, output_freq=None,
              datetime=None, datetime_offset=None,
              start=pd.datetime(1990,1,1), end=pd.datetime.today(),
              return_description=False,
              verbose=False,
              **kwargs):
    """Read in data (e.g., output from a sonic anemometer) at a height
    on the met mast and standardize outputs

    Inputs
    ------
    fpath : str
        Filename passed to pandas.read_csv()
    column_spec : OrderedDict
        Pairs of column names and data formats
    height : float
        Height to associate with this data
    multi_index : bool, optional
        If height is specified, then return a pandas.DataFrame with a
        MultiIndex
    datetime_start : str or callable, optional
        To specify datetime information missing from the data file; if
        callable, then a function to parse the starting datetime from
        the filename
    datetime_start_format : str, optional
        If datetime_start is specified, then the format of the provided
        datetime string
    datetime : pandas.DatetimeIndex, optional
        If no date or time information are included in datafile, use
        this specified datetime series
    data_freq : str, optional
        Assuming data were recorded at regular intervals, the time
        between samples described by a pandas offset string; used with
        datetime_start to create a DatetimeIndex when date or time
        information are not included in the datafile (or to overwrite
        the data)
    max_data_rows : int, optional
        Maximum rows to process from datafile
    datetime_offset : float, optional
        Add a time offset (in seconds) that will be converted into a
        timedelta, e.g., for standardizing data that were averaged to
        the beginning or end of an interval
    output_freq : int, optional
        For output_freq 'N', return every Nth row in datafile; for crude
        resampling of data with high sampling frequency but inconsistent
        number of data rows in output files
    start,end : str or datetime, optional
        Trim the data down to this specified time range
    **kwargs : optional
        Additional arguments to pass to pandas.read_csv()
    """
    columns = column_spec.keys()
    df = pd.read_csv(fpath,names=columns,**kwargs)

    # standardize the data
    datetime_columns = []
    description = []
    for col,fmt in column_spec.items():
        if fmt==1:
            #description.append('read column {:s}'.format(col))
            continue
        elif isinstance(fmt,(int,float)):
            # convert to standard units
            description.append('scaled column {:s} by factor of {:g}'.format(col,1./fmt))
            df[col] = df[col] / fmt
        elif callable(fmt):
            # apply function to column
            funcdesc = inspect.getsource(fmt)
            eqidx = funcdesc.index('=')
            funcdesc = funcdesc[eqidx+1:].strip()
            description.append('applied function ({:s}) to column {:s}'.format(funcdesc,col))
            df[col] = df[col].apply(fmt)
        elif isinstance(fmt,str):
            # collect datetime column names
            #description.append('read datetime-related column {:s}'.format(col))
            datetime_columns.append(col)
        elif fmt is None:
            description.append('ignored column {:s}'.format(col))
            df = df.drop(columns=col)
        else:
            raise TypeError('Unexpected column name/format:',(col,fmt))
    if (len(datetime_columns) == 0) and \
            ((datetime is None) and ((datetime_start=='') or (data_freq is None))):
        raise ValueError('No datetime data in file; need to specify datetime, or datetime_start and data_freq')
    elif (len(datetime_columns) > 0) and (datetime is not None):
        if verbose:
            print('Note: datetime specified; datetime information in datafile ignored')
    elif (len(datetime_columns) > 0) and \
            (datetime_start != '') and (data_freq is not None):
        if verbose:
            print('Note: datetime_start and data_freq specified; datetime information in datafile ignored')

    if callable(datetime_start):
        # parse datetime from file name
        fname = os.path.split(fpath)[-1]
        datetime_start = datetime_start(fname)

    # set up date/time column
    if datetime is not None:
        # use user-specified datetime
        df[datetime_name] = datetime
    elif datetime_start and data_freq:
        # use user-specified start datetime and time interval ('data_freq')
        # specified by a pandas offset string
        # ref: http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        datetime_start = pd.to_datetime(datetime_start,
                                        format=datetime_start_format)
        df[datetime_name] = pd.DatetimeIndex(start=datetime_start,
                                             periods=len(df), freq=data_freq)
    elif datetime_name in datetime_columns:
        # we have complete information
        datetime_format = column_spec[datetime_name]
        df[datetime_name] = pd.to_datetime(df[datetime_name],
                                           format=datetime_format)
    elif (date_name in datetime_columns) and (time_name in datetime_columns):
        # we have separate date and time columns
        if datetime_start != '':
            if verbose: print('Ignored specified datetime_start',datetime_start)
        date_format = column_spec[date_name]
        time_format = column_spec[time_name]
        df[datetime_name] = pd.to_datetime(df[date_name]+df[time_name],
                                           format=date_format+time_format)
        df = df.drop(columns=[date_name,time_name])
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
        if verbose:
            print('Attempting to parse datetime with format"{:s}"'.format(datetime_format))
            print(datetime)
        df[datetime_name] = pd.to_datetime(datetime, format=datetime_format)
        df = df.drop(columns=datetime_columns)

    if max_data_rows is not None:
        df = df.iloc[:max_data_rows]
    if output_freq is not None:
        N = int(output_freq)
        df = df.iloc[::N,:]

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
    df[height_name] = height
    if height and multi_index:
        df = df.set_index([datetime_name,height_name])
    else:
        df = df.set_index(datetime_name)

    # standard calculations
    if not windspeed_name in column_spec.keys():
        # assume we have u,v velocity components
        df[windspeed_name] = np.sqrt(df['u']**2 + df['v']**2)
    if not winddirection_name in column_spec.keys():
        # assume we have u,v velocity components
        df[winddirection_name] = np.degrees(np.arctan2(-df['u'],-df['v']))
        df.loc[df[winddirection_name] < 0, winddirection_name] += 360.0
    try:
        # drop "nonstandard" variables
        df = df.drop(columns=['u','v'])
    except KeyError: pass

    #print('\n'.join(description))
    if return_description:
        return df, description
    else:
        return df


def standard_output(df,output=None,**kwargs):
    """Proposed workflow for "step 1", which entails reading, combining,
    and standardizing data prior to analysis:
        df = metmast.read_data()
        # - at this point, the data should have standard columns, with standard
        #   names, in standard units
        df['calculated_data'] = some_postprocessing()
        # - we may have additional nonstandard columns at this point containing
        #   measured and/or derived quantities
        standard_output(df)
        # Output type will be dictated by the output file extension
        standard_output(df,'/path/to/data.csv')
        standard_output(df,'/path/to/data.nc')
    """
    index_names = df.index.names
    df = df.reset_index()
    column_list = list(df.columns)
    for col in standard_output_columns: 
        column_list.remove(col)
    output_columns = standard_output_columns + column_list
    df = df[output_columns].set_index(index_names)
    if output is None:
        return df
    else:
        _,ext = os.path.splitext(output)        
        if ext == '.csv':
            df.to_csv(output,**kwargs)
        elif ext == '.nc':
            df.to_xarray().to_netcdf(output,**kwargs)
        else:
            raise NotImplementedError('Output extension {:s} not supported'.format(ext))

def tilt_correction(u,v,w,
                    reg_coefs=[[]],
                    tilts=[[]]):
    """Corrects sonic velocities for tilt given regularization
    coefficients and tilt angles.

    Based on JAS' implementation of Branko's correction from EOL 
    description.
    """
    Nz,Nt = u.shape
    assert u.shape == v.shape == w.shape
    assert len(reg_coefs) == Nz
    assert len(tilts) == Nz
    for lvl in range(Nz):
        a = reg_coefs[lvl][0]
        b = reg_coefs[lvl][1]
        c = reg_coefs[lvl][2]
        tilt = tilts[lvl][0]
        tiltaz = tilts[lvl][1]
        #Wf = ( sin(tilt)*cos(tiltaz), sin(tilt)*sin(tiltaz), cos(tilt) )
        wf1 = np.sin(tilt) * np.cos(tiltaz)
        wf2 = np.sin(tilt) * np.sin(tiltaz)
        wf3 = np.cos(tilt)
        #U'f = ((cos(tilt), 0, -sin(tilt)*cos(tiltaz))
        uf1 = np.cos(tilt)
        uf2 = 0.
        uf3 = -np.sin(tilt) * np.cos(tiltaz)
        ufm = np.sqrt(uf1**2 + uf2**2 + uf3**2)
        uf1 = uf1 / ufm
        uf2 = uf2 / ufm
        uf3 = uf3 / ufm
        #vf = wf x uf
        vf1 = wf2*uf3 - wf3*uf2
        vf2 = wf3*uf1 - wf1*uf3
        vf3 = wf1*uf2 - wf2*uf1
        ug = uf1*u[lvl,:] + uf2*v[lvl,:] + uf3*(w[lvl,:] - a)
        vg = vf1*u[lvl,:] + vf2*v[lvl,:] + vf3*(w[lvl,:] - a)
        wg = wf1*u[lvl,:] + wf2*v[lvl,:] + wf3*(w[lvl,:] - a)
        u[lvl,:] = ug
        v[lvl,:] = vg
        w[lvl,:] = wg
    return u,v,w

