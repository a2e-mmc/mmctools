"""
Data loading routines
- depends on readers defined in remote_sensing.py, met_mast.py, ...

Based on https://github.com/NWTC/datatools/blob/master/wfip2.py
"""
import os,glob
import numpy as np
import pandas as pd


reader_exceptions = (IOError, UnicodeDecodeError, AssertionError, ValueError)


def read_dir(dpath='.',
             reader=pd.read_csv,
             file_filter='*',
             ext='csv',
             sort=True,
             verbose=False,
             **kwargs):
    """Wrapper around pandas read_csv() or data reader function. 
    
    Additional readers:
    - metmast
    - measurements/radar
    - measurements/lidar
    - measurements/sodar
    
    Returns concatenated dataframe made up of dataframes read from CSV
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
        if not fpath.endswith(ext):
            continue
        if verbose:
            print('Reading '+fpath)
        try:
            df = reader(fpath,**kwargs)
        except reader_exceptions as err:
            print(err,'while reading',fpath)
        dataframes.append(df)
    if len(dataframes) == 0:
        print('No dataframes were read!')
        df = None
    else:
        df = pd.concat(dataframes)
    return df


def read_date_dirs(dpath='.',
                   expected_date_format=None,
                   reader=pd.read_csv,
                   ext='csv',
                   verbose=False,
                   **kwargs):
    """Wrapper around pandas read_csv() or data reader function. 

    Additional readers:
    - metmast
    - measurements/radar
    - measurements/lidar
    - measurements/sodar
    
    Return concatenated dataframe made up of dataframes read from
    CSV files contained in _subdirectories with the expected date
    format_. 

    Extra keyword arguments are passed to the data reader.
    """
    dataframes = []
    for dname in sorted(os.listdir(dpath)):
        Nfiles = 0
        fullpath = os.path.join(dpath,dname)
        if os.path.isdir(fullpath):
            try:
                collection_date = pd.to_datetime(dname,
                                                 format=expected_date_format)
            except ValueError:
                if verbose:
                    print('Skipping '+dname)
            else:
                print('Processing '+fullpath)
                for fname in sorted(os.listdir(fullpath)):
                    fpath = os.path.join(fullpath,fname)
                    if not fname.endswith(ext): continue
                    if verbose:
                        print('  reading '+fname)
                    try:
                        df = reader(fpath,**kwargs)
                    except reader_exceptions as err:
                        print('Reader error {:s}: {:s} while reading {:s}'.format(
                                str(type(err)),str(err),fname))
                    dataframes.append(df)
                    Nfiles += 1
            print('  {} dataframes added'.format(Nfiles))
    if len(dataframes) == 0:
        print('No dataframes were read!')
        df = None
    else:
        df = pd.concat(dataframes)
    return df

