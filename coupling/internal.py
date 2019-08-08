#!/usr/bin/env python

'''
Tools for MMC via internal forcing
'''

__author__ = "Dries Allaerts"
__date__   = "May 16, 2019"

import numpy as np
import pandas as pd
import os


class SowfaInput(object):
    """
    SOWFA input class for writing data to SOWFA readable input files
    """
    def __init__(self,
                 dpath,
                 df,
                 dateref,
                 datefrom=None,
                 dateto=None):
        """
        Initialize SOWFA input object

        Usage
        =====
        dpath : str
            Folder to write files to
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
        """
        
        self.dpath = dpath
        # Create folder dpath if needed
        if not os.path.isdir(dpath):
            os.mkdir(dpath)

        # Use dataframe between datefrom and dateto
        if datefrom is None:
            datefrom = df.index[0]
        if dateto is None:
            dateto = df.index[-1]
        # Make copy to avoid SettingwithcopyWarning
        self.df = df.loc[(df.index>=datefrom) & (df.index<=dateto)].copy()
        assert(len(self.df.index.unique())>0), 'No data for requested period of time'
        
        # Store start date for ICs
        self.datefrom = datefrom

        # calculate time in seconds since reference date
        dateref = pd.to_datetime(dateref)
        tdelta = pd.Timedelta(1,unit='s')
        self.df.reset_index(inplace=True)
        self.df['t_index'] = (self.df['datetime'] - dateref) / tdelta
        self.df.set_index('datetime',inplace=True)

    def write_BCs(self,
                  fname,
                  fieldname,
                  fact=1.0
                  ):
        """
        Write surface boundary conditions to SOWFA readable input file
    
        Usage
        =====
        fname : str
            Filename
        fieldname : str
            Name of the field to be written out
        fact : float
            Scale factor for the field, e.g., to scale heat flux to follow
            OpenFOAM sign convention that boundary fluxes are positive if
            directed outward
        """
    
        # extract time and height array
        ts = self.df.t_index.values
        nt = ts.size
    
        # assert field exists and is complete
        assert(fieldname in self.df.columns), 'Field '+fieldname+' not in df'
        assert(~pd.isna(self.df[fieldname]).any()), 'Field '+fieldname+' is not complete (contains NaNs)'
    
        # scale field with factor,
        # e.g., scale heat flux with fact=-1 to follow OpenFOAM sign convention
        fieldvalues = fact * self.df[fieldname].values
    
        with open(os.path.join(self.dpath,fname),'w') as fid:
            fmt = ['    (%g', '%.12g)',]
            np.savetxt(fid,np.concatenate((ts.reshape((nt,1)),
                                          fieldvalues.reshape((nt,1))
                                          ),axis=1),fmt=fmt)
    
        return


    def write_ICs(self,
                  fname,
                  xmom = 'u',
                  ymom = 'v',
                  temp = 'theta',
                  ):
        """
        Write initial conditions to SOWFA readable
        input file
    
        Usage
        =====
        fname : str
            Filename
        xmom : str
            Field name corresponding to the x-velocity
        ymom : str
            Field name corresponding to the y-velocity
        temp : str
            Field name corresponding to the potential temperature
        """
        
        # Make copy to avoid SettingwithcopyWarning
        df = self.df.loc[self.datefrom].copy()

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
        with open(os.path.join(self.dpath,fname),'w') as fid:
            fmt = ['    (%g',] + ['%.12g']*2 + ['%.12g)',]
            np.savetxt(fid,np.concatenate((zs.reshape((nz,1)),
                                           df[xmom].values.reshape((nz,1)),
                                           df[ymom].values.reshape((nz,1)),
                                           df[temp].values.reshape((nz,1))
                                          ),axis=1),fmt=fmt)
        return


    def write_timeheight(self,
                         fname,
                         xmom = 'u',
                         ymom = 'v',
                         zmom = 'w',
                         temp = 'theta',
                         ):
        """
        Write time-height data to SOWFA readable input file
    
        Usage
        =====
        fname : str
            Filename
        xmom : str
            Field name corresponding to x momentum (field or tendency)
        ymom : str
            Field name corresponding to y momentum (field or tendency)
        zmom : str
            Field name corresponding to z momentum (field or tendency)
        temp : str
            Field name corresponding to potential temperature (field or tendency)
        """
    
        # extract time and height array
        zs = self.df.height.unique()
        ts = self.df.t_index.unique()
        nz = zs.size
        nt = ts.size
    
        # set missing fields to zero
        fieldNames = [xmom, ymom, zmom, temp]
        for field in fieldNames:
            if not field in self.df.columns:
                self.df.loc[:,field] = 0.0
    
        # pivot data to time-height arrays
        df_pivot = self.df.pivot(columns='height',values=fieldNames)
        # check data is complete
        for field in fieldNames:
            assert ~pd.isna(df_pivot[field]).any().any()
    
        # write data to SOWFA readable file
        with open(os.path.join(self.dpath,fname),'w') as fid:
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
