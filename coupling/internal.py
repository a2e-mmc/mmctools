#!/usr/bin/env python

'''
Tools for MMC via internal forcing
'''

__author__ = "Dries Allaerts"
__date__   = "May 16, 2019"

import numpy as np
import pandas as pd
import os

def BCs_to_sowfa(
                dpath,
                fname,
                df,
                dateref,
                fieldname,
                datefrom=None,
                dateto=None,
                fact=1.0
                ):
    """
    Write surface boundary conditions to SOWFA readable output file

    Usage
    =====
    dpath : str
        Folder to write files to
    fname : str
        Filename
    df : pandas.DataFrame
        Data (index should be called datetime)
    dateref : str
        Reference datetime, SOWFA uses seconds since this datetime
    fieldname : str
        Name of the field to be written out
    datefrom : str
        Start date of the period that will be written out, if None
        start from the first timestamp in df
    dateto : str
        End date of the period that will be written out, if None end
        with the last timestamp in df
    fact : float
        Scale factor for the field, e.g., to scale heat flux to follow
        OpenFOAM sign convention that boundary fluxes are positive if
        directed outward
    """

    # Create folder dpath if needed
    if not os.path.isdir(dpath):
        os.mkdir(dpath)

    # Use dataframe between datefrom and dateto
    if datefrom is None:
        datefrom = df.index[0]
    if dateto is None:
        dateto = df.index[-1]
    # Make copy to avoid SettingwithcopyWarning
    df = df.loc[(df.index>=datefrom) & (df.index<=dateto)].copy()

    # calculate time in seconds since reference date
    dateref = pd.to_datetime(dateref)
    tdelta = pd.Timedelta(1,unit='s')
    df.reset_index(inplace=True)
    df['t_index'] = (df['datetime'] - dateref) / tdelta
    df.set_index('datetime',inplace=True)

    # extract time and height array
    ts = df.t_index.values
    nt = ts.size

    # assert field exists and is complete
    assert(fieldname in df.columns), 'Field '+fieldname+' not in df'
    assert(~pd.isna(df[fieldname]).any()), 'Field '+fieldname+' is not complete (contains NaNs)'

    # scale field with factor,
    # e.g., scale heat flux with fact=-1 to follow OpenFOAM sign convention
    df[fieldname] *= fact

    with open(os.path.join(dpath,fname),'w') as fid:
        fmt = ['    (%g', '%.12g)',]
        np.savetxt(fid,np.concatenate((ts.reshape((nt,1)),
                                      df[fieldname].values.reshape((nt,1))
                                      ),axis=1),fmt=fmt)

    return


def ICs_to_sowfa(
                dpath,
                fname,
                df,
                datetime,
                xmom = 'u',
                ymom = 'v',
                temp = 'theta',
                ):
    """
    Write initial conditions at specified datetime to SOWFA readable
    input file

    Usage
    =====
    dpath : str
        Folder to write files to
    fname : str
        Filename
    df : pandas.DataFrame
        Data (index should be called datetime)
    datetime : str
        Datetime of the initial conditions
    xmom : str
        Field name corresponding to the x-velocity
    ymom : str
        Field name corresponding to the y-velocity
    temp : str
        Field name corresponding to the potential temperature
    """

    # Create folder dpath if needed
    if not os.path.isdir(dpath):
        os.mkdir(dpath)

    # Make copy to avoid SettingwithcopyWarning
    df = df.loc[datetime].copy()

    # set missing fields to zero
    fieldNames = [xmom, ymom, temp]
    for field in fieldNames:
        if not field in df.columns:
            df.loc[:,field] = 0.0

    # extract time and height array
    zs = df.height.values
    nz = zs.size

    # check data is complete
    for field in fieldNames:
        assert ~pd.isna(df[field]).any()

    # write data to SOWFA readable file
    with open(os.path.join(dpath,fname),'w') as fid:
        fmt = ['    (%g',] + ['%.12g']*2 + ['%.12g)',]
        np.savetxt(fid,np.concatenate((zs.reshape((nz,1)),
                                       df[xmom].values.reshape((nz,1)),
                                       df[ymom].values.reshape((nz,1)),
                                       df[temp].values.reshape((nz,1))
                                      ),axis=1),fmt=fmt)
    return


def timeheight_to_sowfa(
                    dpath,
                    fname,
                    df,
                    dateref,
                    datefrom=None,
                    dateto=None,
                    xmom = 'u',
                    ymom = 'v',
                    zmom = 'w',
                    temp = 'theta',
                    ):
    """
    Write time-height data to SOWFA readable input file

    Usage
    =====
    dpath : str
        Folder to write files to
    fname : str
        Filename
    df : pandas.DataFrame
        Data (index should be called datetime)
    dateref : str
        Reference datetime, SOWFA uses seconds since this datetime
    datefrom : str
        Start date of the period that will be written out, if None
        start from the first timestamp in df
    dateto : str
        End date of the period that will be written out, if None end
        with the last timestamp in df
    xmom : str
        Field name corresponding to x momentum (field or tendency)
    ymom : str
        Field name corresponding to y momentum (field or tendency)
    zmom : str
        Field name corresponding to z momentum (field or tendency)
    temp : str
        Field name corresponding to potential temperature (field or tendency)
    """
    
    # Create folder dpath if needed
    if not os.path.isdir(dpath):
        os.mkdir(dpath)

    # Use dataframe between datefrom and dateto
    if datefrom is None:
        datefrom = df.index[0]
    if dateto is None:
        dateto = df.index[-1]
    # Make copy to avoid SettingwithcopyWarning
    df = df.loc[(df.index>=datefrom) & (df.index<=dateto)].copy()
    assert(len(df.index.unique())>0), 'No data for requested period of time'

    # calculate time in seconds since reference date
    dateref = pd.to_datetime(dateref)
    tdelta = pd.Timedelta(1,unit='s')
    df.reset_index(inplace=True)
    df['t_index'] = (df['datetime'] - dateref) / tdelta
    df.set_index('datetime',inplace=True)

    # extract time and height array
    zs = df.height.unique()
    ts = df.t_index.unique()
    nz = zs.size
    nt = ts.size

    # set missing fields to zero
    fieldNames = [xmom, ymom, zmom, temp]
    for field in fieldNames:
        if not field in df.columns:
            df.loc[:,field] = 0.0

    # pivot data to time-height arrays
    df_pivot = df.pivot(columns='height',values=fieldNames)
    # check data is complete
    for field in fieldNames:
        assert ~pd.isna(df_pivot[field]).any().any()

    # write data to SOWFA readable file
    with open(os.path.join(dpath,fname),'w') as fid:
        # Write the height list for the momentum fields
        fid.write('sourceHeightsMomentum\n')    
        np.savetxt(fid,zs,fmt='    %g',header='(',footer=');\n',comments='')
              
        # Write the x-velocity
        fid.write('sourceTableMomentumX\n')
        fmt = ['    (%g',] + ['%.12g']*(nz-1) + ['%.12g)',]
        np.savetxt(fid,np.concatenate((ts.reshape((nt,1)),df_pivot[xmom].values),axis=1),fmt=fmt,
            header='(',footer=');\n',comments='')

        # Write the y-velocity
        fid.write('sourceTableMomentumY\n')
        fmt = ['    (%g',] + ['%.12g']*(nz-1) + ['%.12g)',]
        np.savetxt(fid,np.concatenate((ts.reshape((nt,1)),df_pivot[ymom].values),axis=1),fmt=fmt,
            header='(',footer=');\n',comments='')

        # Write the z-velocity
        fid.write('sourceTableMomentumZ\n')
        fmt = ['    (%g',] + ['%.12g']*(nz-1) + ['%.12g)',]
        np.savetxt(fid,np.concatenate((ts.reshape((nt,1)),df_pivot[zmom].values),axis=1),fmt=fmt,
            header='(',footer=');\n',comments='')

        # Write the height list for the temperature fields
        fid.write('sourceHeightsTemperature\n') 
        np.savetxt(fid,zs,fmt='    %g',header='(',footer=');\n',comments='')

        # Write the temperature
        fid.write('sourceTableTemperature\n')
        fmt = ['    (%g',] + ['%.12g']*(nz-1) + ['%.12g)',]
        np.savetxt(fid,np.concatenate((ts.reshape((nt,1)),df_pivot[temp].values),axis=1),fmt=fmt,
            header='(',footer=');\n',comments='')

    return
