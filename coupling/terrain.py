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
import os
import numpy as np

import utm
import elevation
import rasterio
from rasterio import transform, warp
from rasterio.crs import CRS


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

    def to_terrain(self,dx,dy,resampling=warp.Resampling.bilinear):
        """Load geospatial raster data and reproject onto specified grid

        Usage
        =====
        dx,dy : float
            Grid spacings [m].
        resampling : warp.Resampling value, optional
            See `list(warp.Resampling)`.
        """
        # load raster
        if not os.path.isfile(self.output):
            raise FileNotFoundError('Need to download()')
        dem_raster = rasterio.open(self.output)

        # calculate source coordinate reference system, transform
        src_height, src_width = dem_raster.shape
        src_crs = dem_raster.crs  # coordinate reference system
        src_transform = transform.from_bounds(*self.bounds, src_width, src_height)
        src = dem_raster.read(1)

        # calculate destination coordinate reference system, transform
        # - get coordinate system from zone number associated with mean lat/lon
        west, south, east, north = self.bounds
        lat0 = (north+south) / 2
        lon0 = (west+east) / 2
        x0,y0,zonenum,_ = utm.from_latlon(lat0,lon0)
        proj = '+proj=utm +zone={:d}'.format(zonenum) \
             + '+datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0'
        dst_crs = CRS.from_proj4(proj)
        print('EPSG code:',dst_crs.to_epsg())
        # - get origin (the _upper_ left corner) from bounds
        orix,oriy,_,_ = utm.from_latlon(north,west,force_zone_number=zonenum)
        origin = (orix, oriy)
        dst_transform = transform.from_origin(*origin, dx, dy)
        # - get extents from lower right corner
        LL_x,LL_y,_,_ = utm.from_latlon(south,east,force_zone_number=zonenum)
        Lx = LL_x - orix
        Ly = oriy - LL_y
        Nx = int(Lx / dx)
        Ny = int(Ly / dy)

        # reproject to UTM grid
        dem_array = np.empty((Ny, Nx))
        warp.reproject(src, dem_array,
                       src_transform=src_transform, src_crs=src_crs,
                       dst_transform=dst_transform, dst_crs=dst_crs,
                       resampling=resampling)
        utmx = orix + np.arange(0, Nx*dx, dx)
        utmy = oriy + np.arange((-Ny+1)*dy, dy, dy)
        self.x,self.y = np.meshgrid(utmx,utmy,indexing='ij')
        self.z = np.flipud(dem_array).T
        return self.x, self.y, self.z
