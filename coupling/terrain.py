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
from scipy.interpolate import RectBivariateSpline

import elevation
import rasterio
from rasterio import transform, warp
from rasterio.crs import CRS


class Terrain(object):

    latlon_crs = CRS.from_dict(init='epsg:4326')

    def __init__(self,latlon_bounds,fpath='output.tif'):
        """Create container for manipulating GeoTIFF data in the
        specified region

        Usage
        =====
        latlon_bounds : list or tuple
            Latitude/longitude corresponding to west, south, east, and
            north bounds, used to define the source transformation.
        fpath : str, optional
            Where to save downloaded GeoTIFF (*.tif) data.
        """
        self.bounds = list(latlon_bounds)
        self._get_utm_crs() # from bounds
        self.output = fpath
        self.have_terrain = False

    def _get_utm_crs(self,datum='WGS84',ellps='WGS84'):
        """Get coordinate system from zone number associated with the
        longitude of the northwest corner

        Parameters
        ==========
        datum : str, optional
            Origin of destination coordinate system, used to describe
            PROJ.4 string; default is WGS84.
        ellps : str, optional
            Ellipsoid defining the shape of the earth in the destination
            coordinate system, used to describe PROJ.4 string; default
            is WGS84.
        """
        #west, south, east, north = self.bounds
        self.zone_number = int((self.bounds[0] + 180) / 6) + 1
        proj = '+proj=utm +zone={:d} '.format(self.zone_number) \
             + '+datum={:s} +units=m +no_defs '.format(datum) \
             + '+ellps={:s} +towgs84=0,0,0'.format(ellps)
        self.utm_crs = CRS.from_proj4(proj)

    def to_terrain(self,dx,dy=None,resampling=warp.Resampling.bilinear):
        """Load geospatial raster data and reproject onto specified grid

        Usage
        =====
        dx,dy : float
            Grid spacings [m]. If dy is not specified, then uniform
            spacing is assumed.
        resampling : warp.Resampling value, optional
            See `list(warp.Resampling)`.
        """
        if dy is None:
            dy = dx

        # load raster
        if not os.path.isfile(self.output):
            raise FileNotFoundError('Need to download()')
        dem_raster = rasterio.open(self.output)

        # get source coordinate reference system, transform
        west, south, east, north = self.bounds
        src_height, src_width = dem_raster.shape
        src_crs = dem_raster.crs
        src_transform = transform.from_bounds(*self.bounds, src_width, src_height)
        src = dem_raster.read(1)

        # calculate destination coordinate reference system, transform
        dst_crs = self.utm_crs
        print('Projecting from',src_crs,'to',dst_crs)
        # - get origin (the _upper_ left corner) from bounds
        orix,oriy = self.to_xy(north,west)
        origin = (orix, oriy)
        self.origin = origin
        dst_transform = transform.from_origin(*origin, dx, dy)
        # - get extents from lower right corner
        SE_x,SE_y = self.to_xy(south,east)
        Lx = SE_x - orix
        Ly = oriy - SE_y
        Nx = int(Lx / dx)
        Ny = int(Ly / dy)

        # reproject to uniform grid in the UTM CRS
        dem_array = np.empty((Ny, Nx))
        warp.reproject(src, dem_array,
                       src_transform=src_transform, src_crs=src_crs,
                       dst_transform=dst_transform, dst_crs=dst_crs,
                       resampling=resampling)
        utmx = orix + np.arange(0, Nx*dx, dx)
        utmy = oriy + np.arange((-Ny+1)*dy, dy, dy)
        self.x,self.y = np.meshgrid(utmx,utmy,indexing='ij')
        self.z = np.flipud(dem_array).T

        self.zfun = RectBivariateSpline(utmx,utmy,self.z)
        self.have_terrain = True

        return self.x, self.y, self.z

    def to_latlon(self,x,y):
        """Transform uniform grid to lat/lon space"""
        if not hasattr(x, '__iter__'):
            assert ~hasattr(x, '__iter__')
            x = [x]
            y = [y]
        xlon, xlat = warp.transform(self.utm_crs,
                                    self.latlon_crs,
                                    x, y)
        try:
            shape = x.shape
        except AttributeError:
            xlat = xlat[0]
            xlon = xlon[0]
        else:
            xlat = np.reshape(xlat, shape)
            xlon = np.reshape(xlon, shape)
        return xlat,xlon

    def to_xy(self,lat,lon,xref=None,yref=None):
        """Transform lat/lon to UTM space"""
        if not hasattr(lat, '__iter__'):
            assert ~hasattr(lat, '__iter__')
            lat = [lat]
            lon = [lon]
        x,y = warp.transform(self.latlon_crs,
                             self.utm_crs,
                             lon, lat)
        try:
            shape = lon.shape
        except AttributeError:
            x = x[0]
            y = y[0]
        else:
            x = np.reshape(x, shape)
            y = np.reshape(y, shape)
        if xref is not None:
            x -= xref
        if yref is not None:
            y -= yref
        return x,y

    def xtransect(self,xy=None,latlon=None,wdir=270.0,xrange=(None,None)):
        """Get terrain transect along x for a slice aligned with the
        specified wind direction and going through a specified reference
        point (defined by xy or latlon)

        Usage
        =====
        xy : list or tuple
            Reference location in the UTM coordinate reference system [m]
        latlon : list-like
            Reference location in latitude and longitude [deg]
        wdir : float
            Wind direction with which the slice is aligned [deg]
        xrange : list or tuple, optional
            Range of x values over which slice (or None to use min/max)
        """
        assert self.have_terrain, 'Need to call to_terrain()'
        assert ((xy is not None) ^ (latlon is not None)), 'Specify xy or latlon'
        if xy:
            refloc = xy
        elif latlon: 
            x,y = self.to_xy(*latlon)
            refloc = (x,y)
        ang = 270 - wdir
        print('Slice through',refloc,'at',ang,'deg')
        ang *= np.pi/180.

        # direction specific code
        imin = 0 if (xrange[0] is None) else np.where(self.x <= xrange[0])[0][-1]
        imax = None if (xrange[1] is None) else np.where(self.x > xrange[1])[0][0]
        x = self.x[imin:imax,0]
        y = np.tan(ang) * (x-refloc[0]) + refloc[1]
        z = self.zfun(x,y,grid=False)

        return x-refloc[0], z

    def ytransect(self,xy=None,latlon=None,wdir=180.0,yrange=(None,None)):
        """Get terrain transect along x for a slice aligned with the
        specified wind direction and going through a specified reference
        point (defined by xy or latlon)

        Usage
        =====
        xy : list or tuple
            Reference location in the UTM coordinate reference system [m]
        latlon : list-like
            Reference location in latitude and longitude [deg]
        wdir : float
            Wind direction with which the slice is aligned [deg]
        xrange : list or tuple, optional
            Range of x values over which slice (or None to use min/max)
        """
        assert self.have_terrain, 'Need to call to_terrain()'
        assert ((xy is not None) ^ (latlon is not None)), 'Specify xy or latlon'
        if xy:
            refloc = xy
        elif latlon: 
            x,y = self.to_xy(*latlon)
            refloc = (x,y)
        ang = 180 - wdir
        print('Slice through',refloc,'at',ang,'deg')
        ang *= np.pi/180.

        # direction specific code
        jmin = 0 if (yrange[0] is None) else np.where(self.y <= yrange[0])[1][-1]
        jmax = None if (yrange[1] is None) else np.where(self.y > yrange[1])[1][0]
        y = self.y[0,jmin:jmax]
        x = refloc[0] - np.tan(ang) * (y-refloc[1])
        z = self.zfun(x,y,grid=False)

        return y-refloc[1], z


class SRTM(Terrain):
    """Class for working with Shuttle Radar Topography Mission (SRTM) data"""
    data_products = {
        'SRTM1': 30.0,
        'SRTM3': 90.0,
    }

    def __init__(self,latlon_bounds,fpath='output.tif',product='SRTM3',
                 margin=0.05):
        """Create container for SRTM data in the specified region

        Usage
        =====
        latlon_bounds : list or tuple
            Latitude/longitude corresponding to west, south, east, and
            north bounds, used to define the source transformation.
        fpath : str, optional
            Where to save downloaded GeoTIFF (*.tif) data.
        product : str, optional
            Data product name, SRTM1 or SRTM3 (corresponding to 30- and
            90-m DEM).
        margin : float, optional
            Decimal degree margin added to the bounds (default is 3")
            when clipping the downloaded elevation data.
        """
        latlon_bounds = list(latlon_bounds)
        if margin is not None:
            latlon_bounds[0] -= margin
            latlon_bounds[1] -= margin
            latlon_bounds[2] += margin
            latlon_bounds[3] += margin
        super().__init__(latlon_bounds,fpath=fpath)
        assert (product in self.data_products.keys()), \
                'product should be one of '+str(list(self.data_products.keys()))
        self.product = product
        self.margin = margin

    def download(self,cleanup=True):
        """Download the SRTM data in GeoTIFF format"""
        dpath = os.path.dirname(self.output)
        if not os.path.isdir(dpath):
            print('Creating path',dpath)
            os.makedirs(dpath)
        elevation.clip(self.bounds, product=self.product, output=self.output)
        if cleanup:
            elevation.clean()

    def to_terrain(self,dx=None,dy=None,resampling=warp.Resampling.bilinear):
        """Load geospatial raster data and reproject onto specified grid

        Usage
        =====
        dx,dy : float
            Grid spacings [m]. If dy is not specified, then uniform
            spacing is assumed.
        resampling : warp.Resampling value, optional
            See `list(warp.Resampling)`.
        """
        if dx is None:
            dx = self.data_products[self.product]
            print('Output grid at ds=',dx)
        if dy is None:
            dy = dx
        return super().to_terrain(dx, dy=dy, resampling=resampling)


class USGS(Terrain):
    """Class for working with the US Geological Survey's 3D Elevation
    Program (3DEP). Note that there is no API so the tif data must be
    manually downloaded from the USGS.
    """

    def __init__(self,latlon_bounds,fpath='output.tif'):
        """Create container for 3DEP data in the specified region

        Usage
        =====
        latlon_bounds : list or tuple
            Latitude/longitude corresponding to west, south, east, and
            north bounds, used to define the source transformation.
        fpath : str, optional
            Location of downloaded GeoTIFF (*.tif) data.
        """
        super().__init__(latlon_bounds,fpath=fpath)

    def download(self):
        """This is just a stub"""
        print('Data must be manually downloaded!')
        print('Go to https://viewer.nationalmap.gov/basic/,')
        print('select "Data > Elevation Products (3DEP)"')
        print('and then click "Find Products"')

