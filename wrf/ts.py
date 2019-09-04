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

    def __init__(self,outdir,casedir,domain,
                 starttime,timestep=10.0,
                 towersubdir='towers',
                 verbose=True):
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
        starttime : str or Timestamp
            The datetime at which the simulation was started
            (corresponding to t=0 in the sampling output), which should
            correspond to the start_* namelist parameters in the WRF
            namelist.input.
        timestep : float
            The timestep for the selected WRF domain, in seconds.
        domain : int
            WRF domain to use (domain >= 1)
        towersubdir : str, optional
            Expected name of subdirectory containing the tslist output.
        """
        self.verbose = verbose # for debugging
        self.outdir = outdir
        self.casedir = casedir
        self.starttime = pd.to_datetime(starttime)
        self.timestep = timestep
        self.domain = domain
        self.tslistpath = os.path.join(casedir,'tslist')
        self.towerdir = os.path.join(casedir, towersubdir)
        assert os.path.isfile(self.tslistpath), \
                'tslist not found in WRF case directory'
        assert os.path.isdir(self.towerdir), \
                'towers subdirectory not found'
        self._load_tslist()
        self._load_data()

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
        self.tslist.set_index('prefix',inplace=True)

    def _load_data(self):
        """Load ncfile(s) if they exist, or generate them using the
        Tower class"""
        self.data = {}
        for prefix in self.tslist.index:
            fpath = os.path.join(self.outdir, prefix+'.nc')
            if os.path.isfile(fpath):
                if self.verbose: print('Reading',fpath)
                self.data[prefix] = xr.open_dataset(fpath)
            else:
                if self.verbose: print('Creating',fpath)
                self.data[prefix] = self._process_tower(prefix,fpath)

    def _process_tower(self,prefix,outfile=None):
        """Use Tower.to_dataframe() to create a dataframe, to which we
        add tower latitude/longitude. Setting them as indices makes the
        recognizable as coordinates by xarray.
        """
        towerfile = '{:s}.d{:02d}.*'.format(prefix, self.domain)
        fpath = os.path.join(self.towerdir,towerfile)
        df = Tower(fpath,varlist=self.varnames).to_dataframe(
                start_time=self.starttime, time_step=self.timestep, heights=None)
        towerinfo = self.tslist.loc[prefix]
        df['lat'] = towerinfo['lat']
        df['lon'] = towerinfo['lon']
        df.set_index(['lat','lon'], append=True, inplace=True)
        nc = df.to_xarray()
        if outfile is not None:
            nc.to_netcdf(outfile)
        return nc

        


