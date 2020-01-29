"""
Data loading routines
- depends on readers defined in remote_sensing.py, met_mast.py, ...

Based on https://github.com/NWTC/datatools/blob/master/wfip2.py
"""
import os,glob
import logging
import numpy as np
import pandas as pd
import xarray


reader_exceptions = (IOError, UnicodeDecodeError, AssertionError, ValueError)
netcdf_time_names = ['Time','time','datetime']

def _concat(datalist):
    if isinstance(datalist[0], (pd.Series, pd.DataFrame)):
        return pd.concat(datalist)
    elif isinstance(datalist[0], (xarray.Dataset, xarray.DataArray)):
        dim = None
        for timename in netcdf_time_names:
            if timename in datalist[0].coords:
                dim = timename
                break
        if dim is None:
            print('Unable to concatenate data arrays; time dimension not one of',
                  netcdf_time_names)
            return datalist
        else:
            return xarray.concat(datalist, dim=dim)


def read_files(filelist=[],
               reader=pd.read_csv,
               sort=True,
               verbose=False,
               **kwargs):
    """Wrapper around pandas read_csv() or data reader function. 
    
    Additional readers:
    - metmast
    - measurements/radar
    - measurements/lidar
    - measurements/sodar
    
    Returns concatenated dataframe made up of dataframes read from text
    files in specified list. 

    Additional keyword arguments are passed to the data reader.
    """
    dataframes = []
    if sort:
        filelist.sort()
    for fpath in filelist:
        if not os.path.isfile(fpath): continue
        if verbose:
            print('Reading '+fpath)
        try:
            df = reader(fpath,verbose=verbose,**kwargs)
        except reader_exceptions as err:
            logging.exception('Error while reading {:s}'.format(fpath))
        else:
            dataframes.append(df)
    if len(dataframes) == 0:
        print('No dataframes were read!')
        df = None
    else:
        df = _concat(dataframes)
    return df


def read_dir(dpath='.',file_filter='*',
             reader=pd.read_csv,
             sort=True,
             verbose=False,
             **kwargs):
    """Wrapper around pandas read_csv() or data reader function. 
    
    Additional readers:
    - measurements/metmast
    - measurements/radar
    - measurements/lidar
    - measurements/sodar
    
    Returns concatenated dataframe made up of dataframes read from text
    files in specified directory. Filenames may be filtered with the 
    file_filter argument, which is used to select files with globbing.

    Additional keyword arguments are passed to the data reader.
    """
    dataframes = []
    fpathlist = glob.glob(os.path.join(dpath,file_filter))
    if sort:
        fpathlist.sort()
    for fpath in fpathlist:
        if not os.path.isfile(fpath): continue
        if verbose:
            print('Reading '+fpath)
        try:
            df = reader(fpath,verbose=verbose,**kwargs)
        except reader_exceptions as err:
            logging.exception('Error while reading {:s}'.format(fpath))
        else:
            dataframes.append(df)
    if len(dataframes) == 0:
        print('No dataframes were read!')
        df = None
    else:
        df = _concat(dataframes)
    return df


def read_date_dirs(dpath='.',dir_filter='*',file_filter='*',
                   expected_date_format='%Y%m%d',
                   reader=pd.read_csv,
                   verbose=False,
                   **kwargs):
    """Wrapper around pandas read_csv() or data reader function. 

    If expected_date_format is None, then the datetime is assumed to be
    in seconds (e.g., a timedelta or simulation time).

    Additional readers:
    - measurements/metmast
    - measurements/radar
    - measurements/lidar
    - measurements/sodar
    
    Return concatenated dataframe made up of dataframes read from
    text files contained in _subdirectories with the expected date
    format_. 

    Extra keyword arguments are passed to the data reader.
    """
    dataframes = []
    dpathlist = glob.glob(os.path.join(dpath,dir_filter))
    for fullpath in sorted(dpathlist):
        Nfiles = 0
        dname = os.path.split(fullpath)[-1]
        if os.path.isdir(fullpath):
            try:
                # check that subdirectories have the expected format
                if expected_date_format is None:
                    timedelta = float(dname)
                else:
                    collection_date = pd.to_datetime(
                            dname, format=expected_date_format)
            except ValueError:
                if verbose: print('Skipping '+dname)
            else:
                if verbose: print('Processing '+fullpath)
                filelist = glob.glob(os.path.join(fullpath,file_filter))
                for fpath in sorted(filelist):
                    if verbose: print('  reading '+fpath)
                    try:
                        df = reader(fpath,verbose=verbose,**kwargs)
                    except reader_exceptions as err:
                        logging.exception('Error while reading {:s}'.format(fpath))
                    else:
                        dataframes.append(df)
                    Nfiles += 1
            if verbose: print('  {} dataframes added'.format(Nfiles))
    if len(dataframes) == 0:
        print('No dataframes read from',fullpath)
        df = None
    else:
        df = _concat(dataframes)
    return df

