"""
Module to process WRF time-series output
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

from .utils import Tower


def read_tslist(fpath):
    """Read the description of sampling locations"""
    return pd.read_csv(fpath,comment='#',delim_whitespace=True,
                       names=['name','prefix','lat','lon'])


class TowerArray(object):
    """Read and store an array of Tower objects sampled from WRF using
    the tslist
    """
    def __init__(self,casedir,outdir,towersubdir='towers'):
        """Create a TowerArray object from a WRF simulation with tslist
        output

        Parameters
        ----------
        casedir : str
            Directory path to where WRF was run, which should contain
            the tslist file and a towers subdirectory.
        outdir : str
            Directory path to where data products, e.g., tower output
            converted into netcdf files, are to be stored.
        towersubdir : str, optional
            Expected name of subdirectory containing the tslist output.
        """
        self.casedir = casedir
        self.outdir = outdir
        self.towerdir = os.path.join(casedir, towersubdir)
        assert os.path.isfile(os.path.join(casedir,'tslist')), \
                'tslist not found in WRF case directory'
        assert os.path.isdir(self.towerdir), \
                'towers subdirectory not found'


