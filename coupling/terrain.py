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

import utm
import elevation
import rasterio
from rasterio import transform, warp
from rasterio.crs import CRS


class SRTM(object):
    """Class for working with Shuttle Radar Topography Mission data"""
    data_products = {
        'SRTM1': 30.0,
        'SRTM3': 90.0,
    }

    def __init__(self,latlon_bounds,fpath='output.tif',product='SRTM3',
                 margin=0.05):
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
        margin : float, optional
            Decimal degree margin added to the bounds (default is 3").
        """
        self.bounds = list(latlon_bounds)
        if margin is not None:
            self.bounds[0] -= margin
            self.bounds[1] -= margin
            self.bounds[2] += margin
            self.bounds[3] += margin
        self.output = fpath
        assert (product in self.data_products.keys()), \
                'product should be one of '+str(list(self.data_products.keys()))
        self.product = product
        self.have_terrain = False

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
        x0,y0,zonenum,zonelet = utm.from_latlon(lat0,lon0)
        self.zone_number = zonenum
        self.zone_letter = zonelet
        proj = '+proj=utm +zone={:d}'.format(zonenum) \
             + '+datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0'
        dst_crs = CRS.from_proj4(proj)
        self.utm_crs = dst_crs
        print('EPSG code:',dst_crs.to_epsg())
        # - get origin (the _upper_ left corner) from bounds
        orix,oriy,_,_ = utm.from_latlon(north,west,force_zone_number=zonenum)
        origin = (orix, oriy)
        self.origin = origin
        dst_transform = transform.from_origin(*origin, dx, dy)
        # - get extents from lower right corner
        LL_x,LL_y,_,_ = utm.from_latlon(south,east,force_zone_number=zonenum)
        Lx = LL_x - orix
        Ly = oriy - LL_y
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
                                    CRS.from_dict(init='epsg:4326'),
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
            x,y,_,_ = utm.from_latlon(*latlon)
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
            x,y,_,_ = utm.from_latlon(*latlon)
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

