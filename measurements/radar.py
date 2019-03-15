"""
Data readers for remote sensing devices (e.g., 3D data)

Based on https://github.com/NWTC/datatools/blob/master/remote_sensing.py
"""
import numpy as np
import pandas as pd

def profiler(fname,modes=None,
        check_na=['SPD','DIR'],na_values=999999,
        verbose=False):
    """Wind Profiler radar with RASS

    Users:
    - Earth Sciences Research Laboratory (ESRL)
    - Texas Tech University (TTU)

    Assumed data format for consensus data format rev 5.1 based on
    provided reference for rev 4.1 from:
    https://a2e.energy.gov/data/wfip2/attach/915mhz-cns-winds-data-format.txt

    Additional data format reference:
    https://www.esrl.noaa.gov/psd/data/obs/formats/

    Usage
    =====
    modes : int or None
        Number of data blocks to read from file; set to None to read all data
    check_na : list
        Column names from file to check for n/a or nan values
    na_values : values or list of values
        Values to be considered n/a and set to nan
    """
    dataframes = []
    with open(fname,'r') as f:
        if modes is not None:
            for _ in range(modes):
                dataframes.append(_read_profiler_data_block(f))
        else:
            while True:
                try:
                    dataframes.append(_read_profiler_data_block(f))
                except (IOError,IndexError):
                    break
    df = pd.concat(dataframes)
    if na_values is not None:
        nalist = []
        for col in check_na:
            if col in df.columns:
                matches = [col]
            else:
                matches = [ c for c in df.columns if c.startswith(col+'.') ]
            if len(matches) > 0:
                nalist += matches
            else:
                if verbose:
                    print('Note: column '+col+'* not found')
        check_na = nalist
        if not hasattr(na_values,'__iter__'):
            na_values = [na_values]
        for val in na_values:
            for col in check_na:
                if verbose:
                    print('Checking',col,'for',val)
                df.loc[df[col]==val,col] = np.nan # flag bad values
    return df

def _read_profiler_data_block(f,expected_datatypes=['WINDS','RASS']):
    """Used by radar profiler"""
    # Line 1 (may not be present for subsequent blocks within the same file
    if f.readline().strip() == '':
        f.readline() # Line 2: station name
    assert(f.readline().split()[0] in expected_datatypes) # Line 3: WINDS, version
    f.readline() # Line 4: lat (N), long (W), elevation (m)
    Y,m,d,H,M,S,_ = f.readline().split() # Line 5: date
    date_time = pd.to_datetime('20{}{}{} {}{}{}'.format(Y,m,d,H,M,S))
    f.readline() # Line 6: consensus averaging time
    f.readline() # Line 7: beam info
    f.readline() # Line 8: beam info
    f.readline() # Line 9: beam info
    f.readline() # Line 10: beam info
    header = f.readline().split()
    header = [ col + '.' + str(header[:i].count(col))
               if header.count(col) > 1
               else col
               for i,col in enumerate(header) ]
    block = []
    line = f.readline()
    while not line.strip()=='$' and not line=='':
        block.append(line.split())
        line = f.readline()
    df = pd.DataFrame(data=block,columns=header,dtype=float)
    df['date_time'] = date_time
    return df
