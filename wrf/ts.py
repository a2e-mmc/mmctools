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
        self._check_inputs()
        self._load_tslist()
        self._load_data()

    def __repr__(self):
        return str(self.tslist)

    def _check_inputs(self):
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        assert os.path.isfile(self.tslistpath), \
                'tslist not found in WRF case directory'
        assert os.path.isdir(self.towerdir), \
                'towers subdirectory not found'

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
        self.filelist = [ os.path.join(self.outdir, prefix+'.nc')
                          for prefix in self.tslist.index ]
        for prefix,fpath in zip(self.tslist.index, self.filelist):
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

    def combine(self,cleanup=True):
        """At the moment, xr.combine_by_coords() is not generally
        available (at least not from the default conda xarray package,
        version 0.12.1). As a workaround, xr.open_mfdataset() can
        accomplish the same thing with an add I/O step.
        """
        self.ds = xr.open_mfdataset(self.filelist)
        if cleanup is True:
            import gc # garbage collector
            try:
                del self.data
            except AttributeError:
                pass
            else:
                if self.verbose:
                    print('Cleared data dict from memory')
            finally:
                gc.collect()
        return self.ds

