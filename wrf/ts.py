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
    varnames = ['uu','vv','ww','th','pr','ph']

    def __init__(self,outdir,casedir,domain,towersubdir='towers'):
        """Create a TowerArray object from a WRF simulation with tslist
        output

        Parameters
        ----------
        outdir : str
            Directory path to where data products, e.g., tower output
            converted into netcdf files, are to be stored.
        casedir : str
            Directory path to where WRF was run, which should contain
            the tslist file and a towers subdirectory.
        domain : int
            WRF domain to use (domain >= 1)
        towersubdir : str, optional
            Expected name of subdirectory containing the tslist output.
        """
        self.outdir = outdir
        self.casedir = casedir
        self.domain = domain
        self.tslistpath = os.path.join(casedir,'tslist')
        self.towerdir = os.path.join(casedir, towersubdir)
        assert os.path.isfile(self.tslistpath), \
                'tslist not found in WRF case directory'
        assert os.path.isdir(self.towerdir), \
                'towers subdirectory not found'
        self._load_tslist()
        #self._load_data_if_needed()

    def __repr__(self):
        return str(self.tslist)

    def _load_tslist(self):
        self.tslist = read_tslist(self.tslistpath)
        # check availability of all towers
        for prefix in self.tslist['prefix']:
            for varname in self.varnames:
                fpath = os.path.join(self.towerdir,
                                     '{:s}.d{:02d}.{:2s}'.format(prefix,
                                                                 self.domain,
                                                                 varname.upper()))
                assert os.path.isfile(fpath), '{:s} not found'.format(fpath)

    #def _load_data_if_needed(self):
        


