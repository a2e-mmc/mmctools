"""
Data readers for remote sensing devices (e.g., 3D data)

Based on https://github.com/NWTC/datatools/blob/master/remote_sensing.py
"""
import numpy as np
import pandas as pd
import xarray as xr


def calc_xyz(df,range=None,azimuth=None,elevation=0.0,
             small_elevation_angles=False):
    try:
        r  = df.index.get_level_values('range')
    except KeyError:
        assert (range is not None), 'need to specify constant value for `range`'
        r = range
    try:
        az = np.radians(90 - df.index.get_level_values('azimuth'))
    except ValueError:
        assert (range is not None), 'need to specify constant value for `azimuth`'
        az = np.radians(90 - azimuth)
    try:
        el = np.radians(df.index.get_level_values('elevation'))
    except ValueError:
        el = np.radians(elevation)
    x = r * np.cos(az)
    y = r * np.sin(az)
    if not small_elevation_angles:
        x *= np.cos(el)
        y *= np.cos(el)
    z = r * np.sin(el)
    return x,y,z


class LidarData(object):
    def __init__(self,df,verbose=True):
        """Lidar data described by range, azimuth, and elevation"""
        self.verbose = verbose
        self.df = df
        self.RHI = False
        self.PPI = False
        self._check_coords()

    def _check_coords(self):
        if all([coord in self.df.index.names
                for coord in ['range','azimuth','elevation']
               ]):
            if self.verbose: print('3D volumetric scan loaded')
        elif 'range' not in self.df.index.names:
            if self.verbose: print('Vertical scan loaded')
        elif 'azimuth' not in self.df.index.names:
            if self.verbose: print('RHI scan loaded')
            self.RHI = True
        elif 'elevation' not in self.df.index.names:
            if self.verbose: print('PPI scan loaded')
            self.PPI = True
        else:
            raise IndexError('Unexpected index levels in dataframe: '+str(self.df.index.names))

        dr = self.df.index.levels[0][1] - self.df.index.levels[0][0]
        if hasattr(self, 'range_gate_size'):
            assert self.range_gate_size == dr
        else:
            self.range_gate_size = dr
        self.rmax = self.df.index.levels[0][-1] + self.range_gate_size

    @property 
    def r(self):
        return self.df.index.levels[0]

    @property
    def az(self):
        return self.df.index.levels[1]

    @property
    def el(self):
        return self.df.index.levels[2]

    # slicers
    def get(self, r=None, az=None, el=None):
        if r is not None:
            return self.get_range(r)
        elif az is not None:
            return self.get_azimuth(az)
        elif el is not None:
            return self.get_elevation(el)

    def get_range(self, r):
        rarray = self.df.index.levels[0]
        if r < 0:
            raise ValueError('Invalid range, r < 0')
        elif r >= self.rmax:
            raise ValueError(f'Invalid range, r >= {self.rmax}')
        if r not in rarray:
            try:
                idx = np.where(r < rarray)[0][0] - 1
            except IndexError:
                idx = len(rarray) - 1
                r0 = self.df.index.levels[0][idx]
                r1 = self.rmax
            else:
                r0 = self.df.index.levels[0][idx]
                r1 = self.df.index.levels[0][idx+1]
            assert (r >= r0) & (r < r1)
        else:
            idx = list(rarray).index(r)
            r0 = r
            r1 = r + self.range_gate_size
        if self.verbose:
            print(f'getting range gate {idx} between {r0} and {r1}')
        return self.df.xs(r0, level='range'), (r0+r1)/2
    
    def get_azimuth(self, az):
        azarray = self.df.index.levels[1]
        if az < azarray[0]:
            raise ValueError(f'Invalid range, az < {azarray[0]}')
        elif az > azarray[-1]:
            raise ValueError(f'Invalid range, az > {azarray[-1]}')
        if az not in azarray:
            az = azarray[np.argmin(np.abs(az - azarray))]
            if self.verbose:
                print(f'getting nearest azimuth={az} deg')
        return self.df.xs(az, level='azimuth')

    def get_elevation(self, el):
        elarray = self.df.index.levels[2]
        if el < elarray[0]:
            raise ValueError(f'Invalid range, el < {elarray[0]}')
        elif el > elarray[-1]:
            raise ValueError(f'Invalid range, el > {elarray[-1]}')
        if el not in elarray:
            el = elarray[np.argmin(np.abs(el - elarray))]
            if self.verbose:
                print(f'getting nearest elevation={el} deg')
        return self.df.xs(el, level='elevation')


class GalionCornellPerdigao(LidarData):
    """Data from Galion scanning lidar deployed by Cornell University
    at the Perdigao field campaign

    Tested on data retrieved from the UCAR Earth Observing Laboratory
    data archive (https://data.eol.ucar.edu/dataset/536.036) retrieved
    on 2021-07-24.
    """

    def __init__(self,fpath,load_opts={},**kwargs):
        df = self._load(fpath,**load_opts)
        super().__init__(df, **kwargs)

    def _load(self,
              fpath,
              minimum_range=0.,
              range_gate_size=30.,
              range_gate_name='Range gates',
              azimuth_name='Azimuth angle',
              elevation_name='Elevation angle'):
        """Process a single scan in netcdf format

        Notes:
        - Range gates are stored as integers
        - Not all (r,az,el) data points are available
        """
        ds = xr.open_dataset(fpath)
        assert (len(ds.dims)==1) and ('y' in ds.dims)
        df = ds.to_dataframe()
        df = df.rename(columns={
            'Range gates': 'range',
            'Azimuth angle': 'azimuth',
            'Elevation angle': 'elevation',
        })
        df['range'] = minimum_range + df['range']*range_gate_size
        df = df.set_index(['range','azimuth','elevation']).sort_index()
        return df

