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

    def __init__(self,latlon_bounds,fpath='output.tif',product='SRTM3'):
        """Create container for SRTM data in the specified reegion

        Usage
        =====
        latlon_bounds : list or tuple
            Latitude/longitude corresponding to west, south, east, and
            north bounds.
        fpath : str, optional
            Where to save downloaded GeoTIFF (*.tif) data.
        product : str, optional
            Data product name, SRTM1 or SRTM3 (corresponding to 30- and
            90-m DEM).
        """
        self.bounds = latlon_bounds
        self.output = fpath
        assert (product in self.data_products), \
                'product should be one of '+str(self.data_products)
        self.product = product

    def download(self,margin='0'):
        """Download the SRTM data in GeoTIFF format

        Usage
        =====
        margin : str, optional
            Decimal degree margin added to the bounds. Use '%' for
            percent margin.
        """
        elevation.clip(self.bounds, product=self.product, output=self.output,
                       margin=margin)
        elevation.clean()

