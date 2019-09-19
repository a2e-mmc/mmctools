"""
Module to process WRF time-series output
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import time

from .utils import Tower


def read_tslist(fpath,
                snap_to_grid=None,grid_order='F',max_shift=1e-3,
                convert_to_xy=None, latlon_ref=(0,0)):
    """Read the description of sampling locations

    Parameters
    ----------
    fpath : str
        Path to tslist file
    snap_to_grid : list or tuple, or None
        If not None, then adjust the lat/lon coordinates so that they
        lie on a regular grid with shape (Nlat, Nlon). Assume that the
        sampling locations are regularly ordered.
    grid_order : str
        Either 'F' or 'C' for Fortran (axis=0 changes fastest) and C
        ordering (axis=-1 changes fastest), respectively.
    max_shift : float
        If snap_to_grid is True, then this is the maximum amount (in
        degrees) that a tower location will change in latitude or
        longitude.
    convert_to_xy : str or None
        Mapping to use for converting from lat/lon to x/y coordinates.
        If None, x and y are not calculated
    latlon_ref : list or tuple
        Latitude and longitude to use as a reference to determine the
        zone number and relative distances x,y.
    """
    df = pd.read_csv(fpath,comment='#',delim_whitespace=True,
                     names=['name','prefix','lat','lon'])
    if snap_to_grid is not None:
        print('Attemping to adjust grid lat/lon')
        assert (len(snap_to_grid) == 2), 'snap_to_grid should be (Nlat,Nlon)'
        Nlat,Nlon = snap_to_grid
        # original center of sampling grid
        lat0 = df['lat'].mean()
        lon0 = df['lon'].mean()
        # lat/lon have correspond to the first and second dims, respectively
        lat = np.reshape(df['lat'].values, snap_to_grid, order=grid_order)
        lon = np.reshape(df['lon'].values, snap_to_grid, order=grid_order)
        # calculate 1-d lat/lon vectors from average spacing
        delta_lat = np.mean(np.diff(lat, axis=0))
        delta_lon = np.mean(np.diff(lon, axis=1))
        print('  lat/lon spacings:',delta_lat,delta_lon)
        new_lat1 = np.linspace(lat[0,0], lat[0,0]+(Nlat-1)*delta_lat, Nlat)
        new_lon1 = np.linspace(lon[0,0], lon[0,0]+(Nlon-1)*delta_lon, Nlon)
        # calculate new lat/lon grid
        new_lat, new_lon = np.meshgrid(new_lat1, new_lon1, indexing='ij')
        # calculate new center
        new_lat0 = np.mean(new_lat1)
        new_lon0 = np.mean(new_lon1)
        # shift
        lat_shift = lat0 - new_lat0
        lon_shift = lon0 - new_lon0
        if (np.abs(lat_shift) < max_shift) and (np.abs(lon_shift) < max_shift):
            print('  shifting lat/lon grid by ({:g}, {:g})'.format(lat_shift, lon_shift))
            new_lat += lat_shift
            new_lon += lon_shift
            new_lat = new_lat.ravel(order=grid_order)
            new_lon = new_lon.ravel(order=grid_order)
            # one last sanity check, to make sure we didn't screw up
            #   anything during renumbering
            assert np.all(np.abs(new_lat - df['lat']) < max_shift)
            assert np.all(np.abs(new_lon - df['lon']) < max_shift)
            # now update the df
            df['lat'] = new_lat
            df['lon'] = new_lon
        else:
            print('  grid NOT shifted, delta lat/lon ({:g}, {:g}) > {:g}'.format(
                    lat_shift, lon_shift, max_shift))
        if convert_to_xy == 'utm':
            import utm
            x0,y0,zone0,_ = utm.from_latlon(*latlon_ref)
            for prefix,row in df.iterrows():
                x,y,_,_ = utm.from_latlon(row['lat'], row['lon'],
                                          force_zone_number=zone0)
                df.loc[prefix,'x'] = x - x0
                df.loc[prefix,'y'] = y - y0
        elif convert_to_xy is not None:
            print('Unrecognized mapping:',convert_to_xy)
    return df


class TowerArray(object):
    """Read and store an array of Tower objects sampled from WRF using
    the tslist
    """
    varnames = ['uu','vv','ww','th','pr','ph','ts']

    def __init__(self,outdir,casedir,domain,
                 starttime,timestep=10.0,
                 towersubdir='towers',
                 verbose=True,
                 **tslist_args):
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
        tslist_args : optional
            Keyword arguments passed to read_tslist, e.g., `snap_to_grid`
            to enforce a regular lat/lon grid.
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
        self._load_tslist(**tslist_args)

    def _check_inputs(self):
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        assert os.path.isfile(self.tslistpath), \
                'tslist not found in WRF case directory'
        assert os.path.isdir(self.towerdir), \
                'towers subdirectory not found'

    def _load_tslist(self,**kwargs):
        if not os.path.isfile(self.tslistpath):
            print('tslist not found')
            self.tslist = None
            return
        self.tslist = read_tslist(self.tslistpath, **kwargs)
        # check availability of all towers
        for prefix in self.tslist['prefix']:
            for varname in self.varnames:
                fpath = os.path.join(self.towerdir,
                                     '{:s}.d{:02d}.{:2s}'.format(prefix,
                                                                 self.domain,
                                                                 varname.upper()))
                assert os.path.isfile(fpath), '{:s} not found'.format(fpath)
        self.tslist.set_index('prefix',inplace=True)

    def load_data(self,
                  heights=None,height_var='height',approx_height=True,
                  overwrite=False):
        """Load ncfile(s) if they exist, or generate them using the
        Tower class

        Parameters
        ----------
        heights : array-like or None
            Interpolate to these heights at all times; ignored if data
            are read from disk instead of processed from WRF output.
        height_var : str
            If `heights` is not None, this indicates how the height
            values are determined:
            - 'height': Tower.height has been loaded; no actions are 
              performed
            - 'ph': The tower elevation has been stored in the geo-
              potential variable; the height (above ground level) will
              be automatically calculated as Tower.ph - Tower.stationz.
        approx_height : bool
            If `heights` is not None, then assume that height is
            approximately constant in time. This speeds up the
            interpolation because interpolation does not need to be
            performed at each time step.
        overwrite : bool
            Generate new data (to be written as nc files).
        """
        self.data = {}
        self.filelist = [ os.path.join(self.outdir, prefix+'.nc')
                          for prefix in self.tslist.index ]
        for prefix,fpath in zip(self.tslist.index, self.filelist):
            if os.path.isfile(fpath) and not overwrite:
                if self.verbose: print('Reading',fpath)
                self.data[prefix] = xr.open_dataset(fpath)
            else:
                if self.verbose: print('Creating',fpath)
                self.data[prefix] = self._process_tower(prefix,
                                                        heights=heights,
                                                        height_var=height_var,
                                                        approx_height=approx_height,
                                                        outfile=fpath)
    
    def load_combined_data(self,fpath,chunks=None):
        """Load data generated with combine() to avoid having to reload
        the combined dataset. The `chunks` kwarg may be used to load the
        dataset with dask. Tips:
        http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance

        Parameters
        ----------
        chunks : int or dict, optional
            If chunks is provided, it used to load the new dataset into dask
            arrays. ``chunks={}`` loads the dataset with dask using a single
            chunk for all arrays.
        """
        self.ds = xr.open_dataset(fpath, chunks=chunks)
        return self.ds

    def _process_tower(self,prefix,
                       heights=None,height_var=None,approx_height=True,
                       outfile=None):
        """Use Tower.to_dataframe() to create a dataframe, to which we
        add tower latitude/longitude. Setting them as indices makes the
        recognizable as coordinates by xarray.
        """
        towerfile = '{:s}.d{:02d}.*'.format(prefix, self.domain)
        fpath = os.path.join(self.towerdir,towerfile)

        # create Tower object
        totaltime0 = time.time()
        tow = Tower(fpath,varlist=self.varnames)
        excludelist = ['ts'] # skip surface data

        # set up height variable if needed
        if heights is not None:
            assert (height_var is not None), 'height attribute unknown'
        if height_var == 'ph':
            # this creates a time-heightlevel varying height
            tow.height = getattr(tow,height_var) - tow.stationz
            excludelist += [height_var]
            if approx_height:
                mean_height = np.mean(tow.height, axis=0)
                if self.verbose:
                    # diagnostics
                    stdev0 = np.std(tow.height, axis=0)
                    kmax0 = np.argmax(stdev0)
                    print('  max stdev in height at (z~={:g}m) : {:g}'.format(
                            mean_height[kmax0], stdev0[kmax0]))
                    if heights is not None:
                        zmax = np.max(heights)
                        heights_within_micro_dom = np.ma.masked_array(
                                tow.height, tow.height > zmax)
                        stdev = np.std(heights_within_micro_dom, axis=0)
                        kmax = np.argmax(stdev)
                        print('  max stdev in height (up to z={:g} m) at (z~={:g} m) : {:g}'.format(
                                np.max(heights_within_micro_dom), mean_height[kmax], stdev[kmax]))
                tow.height = mean_height
        elif height_var != 'height':
            raise ValueError('Unexpected height_var='+height_var+'; heights not calculated')

        # now convert to a dataframe (note that height interpolation
        # will be (optionally) performed here
        time0 = time.time()
        df = tow.to_dataframe(start_time=self.starttime,
                              time_step=self.timestep,
                              heights=heights,
                              exclude=excludelist)
        time1 = time.time()
        if self.verbose: print('  to_dataframe() time = {:g}s'.format(time1-time0))

        # add additional tower data
        towerinfo = self.tslist.loc[prefix]
        df['lat'] = towerinfo['lat']
        df['lon'] = towerinfo['lon']
        if self.verbose:
            print('  tower lat/lon:',str(towerinfo[['lat','lon']].values))
        df.set_index(['lat','lon'], append=True, inplace=True)

        # convert to xarray
        time0 = time.time()
        ds = df.to_xarray()
        time1 = time.time()
        if self.verbose: print('  to_xarray() time = {:g}s'.format(time1-time0))

        # save
        time0 = time.time()
        if outfile is not None:
            ds.to_netcdf(outfile)
        totaltime1 = time.time()
        if self.verbose:
            print('  xarray output time = {:g}s'.format(totaltime1-time0))
            print('  TOTAL time = {:g}s'.format(totaltime1-totaltime0))
        return ds

    def combine(self,cleanup=False):
        """Create volume data (excluding surface data) by combining
        lat/lon coordinates across all datasets. Tested for data on a
        regular grid.

        Notes:
        - This has a _very_ large memory overhead, i.e., need enough
          memory to store and manipulate all of the tower data
          simultaneously, otherwise it may hang.
        - xarray.combine_by_coords fails with a cryptic "the supplied
          objects do not form a hypercube" message if the lat/lon values
          do not form a regular grid
        """
        datalist = [ data for key,data in self.data.items() ]
        self.ds = xr.combine_by_coords(datalist)
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

