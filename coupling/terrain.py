"""
Tools for working with terrain

Notes
-----
For SRTM data download (in GeoTIFF (.tif) format:
- install (on system, not within python environment): `brew install gdal`
- install with `conda install -c conda-forge elevation` or `pip install elevation`
- check with `eio selfcheck`

For processing downloaded GeoTIFF data:
- install with `conda install -c conda-forge rasterio` or `pip install rasterio`
- note: like the elevation package, this also depends on gdal
"""
import numpy as np

import utm
import elevation
import rasterio
from rasterio import transform, warp


class SRTM(object):
    """Class for working with Shuttle Radar Topography Mission data"""
    data_products = ['SRTM1','SRTM3']

    def __init__(self,latlon_bounds,output='output.tif',product='SRTM3'):
        """Download SRTM data for the specified reegion

        Usage
        =====
        latlon_bounds : list or tuple
            Latitude/longitude corresponding to west, south, east, and
            north bounds.
        output : str
            Where to save downloaded GeoTIFF (*.tif) data.
        product : str'
            Data product name, SRTM1 or SRTM3 (corresponding to 30- and
            90-m DEM).
        """
        self.bounds = latlon_bounds
        self.output = output
        assert (product in self.data_products), \
                'product should be one of '+str(self.data_products)
        self.product = product


