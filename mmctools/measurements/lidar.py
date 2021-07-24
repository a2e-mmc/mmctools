"""
Data readers for remote sensing devices (e.g., 3D data)

Based on https://github.com/NWTC/datatools/blob/master/remote_sensing.py
"""
import numpy as np
import pandas as pd
import xarray as xr

class LidarData(object):
    def __init__(self,df,verbose=True):
        """Lidar data described by range, azimuth, and elevation"""
        self.verbose = verbose
        self.df = df
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
        elif 'elevation' not in self.df.index.names:
            if self.verbose: print('PPI scan loaded')
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


class Perdigao(LidarData):
    """Galion scanning lidar"""

    def __init__(self,
                 fpath,
                 range_gate='Range gates',
                 azimuth='Azimuth angle',
                 elevation='Elevation angle',
                 range_gate_size=30.,
                 **kwargs):
        self.range_gate_size = range_gate_size
        df = self._load(fpath,range_gate,azimuth,elevation)
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

