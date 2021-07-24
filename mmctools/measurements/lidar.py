"""
Data readers for remote sensing devices (e.g., 3D data)

Based on https://github.com/NWTC/datatools/blob/master/remote_sensing.py
"""
import numpy as np
import pandas as pd
import xarray as xr

class LidarData(object):
    def __init__(self,df,range=None,azimuth=None,elevation=None,verbose=True):
        """Lidar data described by range, azimuth, and elevation"""
        self.verbose = verbose
        self.df = df
        self.specified_range = range
        self.specified_azimuth = azimuth
        self.specified_elevation = elevation
        self._check_coords()
        self._to_xyz()

    def _check_coords(self):
        if all([coord in self.df.index.names
                for coord in ['range','azimuth','elevation']
               ]):
            if self.verbose: print('3D volumetric scan loaded')
            if self.specified_range is not None:
                print('Ignoring specified range')
            if self.specified_azimuth is not None:
                print('Ignoring specified azimuth')
            if self.specified_elevation is not None:
                print('Ignoring specified elevation')
        elif 'range' not in self.df.index.names:
            if self.verbose: print('Vertical scan loaded')
            if self.specified_range is None:
                print('`range` not specified, x/y/z will be invalid')
                self.specified_range = 1
        elif 'azimuth' not in self.df.index.names:
            if self.verbose: print('RHI scan loaded')
            if self.specified_azimuth is None:
                print('`azimuth` not specified, x/y/z may be invalid')
                self.specified_azimuth = 0.0
        elif 'elevation' not in self.df.index.names:
            if self.verbose: print('PPI scan loaded')
            if self.specified_elevation is None:
                if self.verbose:
                    print('`elevation` not specified, assuming el=0')
                self.specified_elevation = 0.0
        else:
            raise IndexError('Unexpected index levels in dataframe: '+str(self.df.index.names))

        dr = self.df.index.levels[0][1] - self.df.index.levels[0][0]
        if hasattr(self, 'range_gate_size'):
            assert self.range_gate_size == dr
        else:
            self.range_gate_size = dr
        self.rmax = self.df.index.levels[0][-1] + self.range_gate_size

    def _to_xyz(self):
        try:
            r  = self.df.index.get_level_values('range')
        except KeyError:
            r = self.specified_range
        try:
            az = np.radians(270 - self.df.index.get_level_values('azimuth'))
        except KeyError:
            az = np.radians(self.specified_azimuth)
        try:
            el = np.radians(self.df.index.get_level_values('elevation'))
        except KeyError:
            el = np.radians(self.specified_elevation)
        if not hasattr(self,'x'):
            self.x = r * np.cos(az) * np.cos(el)
        if not hasattr(self,'y'):
            self.y = r * np.sin(az) * np.cos(el)
        if not hasattr(self,'z'):
            self.z = r * np.sin(el)

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
        rs = self.df.index.levels[0]
        if r < 0:
            raise ValueError('Invalid range, r < 0')
        elif r >= self.rmax:
            raise ValueError(f'Invalid range, r >= {self.rmax}')
        if r not in rs:
            try:
                idx = np.where(r < rs)[0][0] - 1
            except IndexError:
                idx = len(rs) - 1
                r0 = self.df.index.levels[0][idx]
                r1 = self.rmax
            else:
                r0 = self.df.index.levels[0][idx]
                r1 = self.df.index.levels[0][idx+1]
            assert (r >= r0) & (r < r1)
        else:
            idx = list(rs).index(r)
            r0 = r
            r1 = r + self.range_gate_size
        if self.verbose:
            print(f'getting range gate {idx} between {r0} and {r1}')
        return self.df.xs(r0, level='range')
    
    def get_azimuth(self, az):
        azs = self.df.index.levels[1]
        if az < azs[0]:
            raise ValueError(f'Invalid range, az < {azs[0]}')
        elif az > azs[-1]:
            raise ValueError(f'Invalid range, az > {azs[-1]}')
        if az not in azs:
            az = azs[np.argmin(np.abs(az - azs))]
            if self.verbose:
                print(f'getting nearest azimuth={az} deg')
        return self.df.xs(az, level='azimuth')

    def get_elevation(self, el):
        els = self.df.index.levels[2]
        if el < els[0]:
            raise ValueError(f'Invalid range, el < {els[0]}')
        elif el > els[-1]:
            raise ValueError(f'Invalid range, el > {els[-1]}')
        if el not in els:
            el = els[np.argmin(np.abs(el - els))]
            if self.verbose:
                print(f'getting nearest elevation={el} deg')
        return self.df.xs(el, level='elevation')


class Perdigao(LidarData):
    """Galion scanning lidar"""

    def __init__(self,
                 fpath,
                 range_gate_name='Range gates',
                 azimuth_name='Azimuth angle',
                 elevation_name='Elevation angle',
                 range_gate_size=30.,
                 **kwargs):
        self.range_gate_size = range_gate_size
        df = self._load(fpath,range_gate_name,azimuth_name,elevation_name)
        super().__init__(df, **kwargs)

    def _load(self,fpath,range_gate,azimuth,elevation):
        """Process a single scan in netcdf format"""
        ds = xr.open_dataset(fpath)
        assert (len(ds.dims)==1) and ('y' in ds.dims)
        r = np.unique(ds[range_gate]) * self.range_gate_size
        az = np.unique(ds[azimuth])
        el = np.unique(ds[elevation])
        df = ds.to_dataframe()
        df = df.rename(columns={
            'Range gates': 'range',
            'Azimuth angle': 'azimuth',
            'Elevation angle': 'elevation',
        })
        df['range'] *= self.range_gate_size
        df = df.set_index(['range','azimuth','elevation']).sort_index()
        return df

