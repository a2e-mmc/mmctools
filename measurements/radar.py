"""
Data readers for remote sensing devices (e.g., 3D data)

Based on https://github.com/NWTC/datatools/blob/master/remote_sensing.py
"""
import numpy as np
import pandas as pd

def radar_profiler(fname,
                   modes=2,
                   check_na=['SPD','DIR'],
                   na_values=999999):
    """Wind Profiler radar with RASS

    Users:
    - Earth Sciences Research Laboratory (ESRL)
    - Texas Tech University (TTU)

    Assumed data format for consensus data format rev 5.1 based on
    provided reference for rev 4.1 from:
    https://a2e.energy.gov/data/wfip2/attach/915mhz-cns-winds-data-format.txt

    Set 'modes' to None to read all blocks in the file

    Additional data format reference:
    https://www.esrl.noaa.gov/psd/data/obs/formats/
    """
    dataframes = []
    with open(fname,'r') as f:
        if modes is not None:
            for _ in range(modes):
                dataframes.append(read_profiler_data_block(f))
        else:
            while True:
                try:
                    dataframes.append(read_profiler_data_block(f))
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
                print('Note: column '+col+'* not found')
        check_na = nalist
        if not hasattr(na_values,'__iter__'):
            na_values = [na_values]
        #print('Checking',check_na,'for',na_values)
        for val in na_values:
            for col in check_na:
                df.loc[df[col]==val,col] = np.nan # flag bad values
    return df

