"""
Tools for working with terrain

Notes
-----
For SRTM data download (in GeoTIFF (.tif) format:
- install with `conda install -c conda-forge gdal` or `pip install gdal`
- install with `conda install -c conda-forge elevation` or `pip install elevation`
- check with `eio selfcheck`

For processing downloaded GeoTIFF data:
- install with `conda install -c conda-forge rasterio` or `pip install rasterio`
- note: like the elevation package, this also depends on gdal
"""
import sys,os,glob
import numpy as np
from scipy.interpolate import RectBivariateSpline

import elevation
import rasterio
from rasterio import transform, warp
from rasterio.crs import CRS

# hard-coded here because ElementTree doesn't appear to have any
# straightforward way to access the xmlns root attributes
ISO_namespace = {
    'gmd': 'http://www.isotc211.org/2005/gmd',
    'gco': 'http://www.isotc211.org/2005/gco',
}


class Terrain(object):

    latlon_crs = CRS.from_dict(init='epsg:4326')

    def __init__(self,latlon_bounds,fpath='terrain.tif'):
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
        self.tiffdata = fpath
        self.have_terrain = False
        if not hasattr(self,'have_metadata'):
            # set attribute if it hasn't been set already
            self.have_metadata = False

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

    def _get_bounds_from_metadata(self):
        """This is a stub"""
        assert self.have_metadata
        raise NotImplementedError()

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
        if not os.path.isfile(self.tiffdata):
            raise FileNotFoundError('Need to download()')
        dem_raster = rasterio.open(self.tiffdata)

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
        print(Lx,Ly,dx,dy)
        print(west,south,east,north)
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

    def __init__(self,latlon_bounds,fpath='terrain.tif',product='SRTM3',
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
        dpath = os.path.dirname(self.tiffdata)
        if not os.path.isdir(dpath):
            print('Creating path',dpath)
            os.makedirs(dpath)
        escapedpath = self.tiffdata.replace('\ ',' ').replace(' ','\ ')
        try:
            elevation.clip(self.bounds, product=self.product, output=escapedpath)
        except:
            info = sys.exc_info()
            print(info[0])
            print(info[1])
            print('')
            print('Note: Have elevation and gdal been installed properly?')
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

    def __init__(self,latlon_bounds=None,fpath='terrain.tif'):
        """Create container for 3DEP data in the specified region

        Usage
        =====
        latlon_bounds : list or tuple, optional
            Latitude/longitude corresponding to west, south, east, and
            north bounds, used to define the source transformation. If
            not specified, then it will be read from a metadata file
            with the same name.
        fpath : str
            Location of downloaded GeoTIFF (*.tif) data.
        """
        self._read_metadata(fpath)
        if latlon_bounds is None:
            latlon_bounds = self._get_bounds_from_metadata()
            print('Bounds:',latlon_bounds)
        super().__init__(latlon_bounds,fpath=fpath)

    def _read_metadata(self,fpath):
        from xml.etree import ElementTree
        xmlfile = os.path.splitext(fpath)[0] + '.xml'
        try:
            metadata = ElementTree.parse(xmlfile).getroot()
        except IOError:
            self.have_metadata = False
        else:
            if not metadata.tag.endswith('MD_Metadata'):
                assert metadata.tag in ['metadata','gmd:MD_Metadata','modsCollection']
            if metadata.tag == 'metadata':
                # legacy metadata
                print('Source CRS datum:',metadata.find('./spref/horizsys/geodetic/horizdn').text)
            elif metadata.tag == 'modsCollection':
                # MODS XML
                print(metadata.find('./mods/titleInfo/title').text)
                raise NotImplementedError('MODS XML detected -- use ISO XML instead')
            else:
                # ISO XML
                title = metadata.find(
                    '/'.join([
                        'gmd:identificationInfo',
                        'gmd:MD_DataIdentification',
                        'gmd:citation',
                        'gmd:CI_Citation',
                        'gmd:title',
                        'gco:CharacterString',
                    ]),
                    ISO_namespace
                ).text
                print(title)
            self.have_metadata = True
            self.metadata = metadata

    def _get_bounds_from_metadata(self):
        assert self.have_metadata
        if self.metadata.tag == 'metadata':
            # legacy metadata
            bounding = self.metadata.find('./idinfo/spdom/bounding')
            bounds = [
                float(bounding.find(bcdir+'bc').text)
                for bcdir in ['west','south','east','north']
            ]
        else:
            # ISO XML
            extent = self.metadata.find(
                'gmd:identificationInfo/gmd:MD_DataIdentification/gmd:extent',
                ISO_namespace
            )
            bbox = extent.find(
                'gmd:EX_Extent/gmd:geographicElement/gmd:EX_GeographicBoundingBox',
                ISO_namespace
            )
            bounds = [
                float(bbox.find(f'gmd:{bound}/gco:Decimal',ISO_namespace).text)
                for bound in [
                    'westBoundLongitude',
                    'southBoundLatitude',
                    'eastBoundLongitude',
                    'northBoundLatitude',
                ]
            ]
        return bounds

    def download(self):
        """This is just a stub"""
        print('Data must be manually downloaded!')
        print('Go to https://apps.nationalmap.gov/downloader/#/,')
        print('select "Data > Elevation Products (3DEP)"')
        print('and then click "Find Products"')


def combine_raster_data(filelist,dtype=Terrain,latlon_bounds=None,
                        output='output.tif'):
    """Combine multiple raster datasets into a single GeoTIFF file

    Usage
    =====
    filelist : list or glob str
        List of downloaded GeoTIFF (*.tif) data.
    dtype : Terrain or derived class, optional
        Used to provide helper functions if needed.
    latlon_bounds : list of (list or tuple), optional
        Each list or tuple of latitude/longitude corresponds to west,
        south, east, and north bounds, and are used to define the bounds
        of the combined raster. If not specified, then try to read these
        bounds from metadata.
    output : str
        Location of combined GeoTIFF (*.tif) data.
    """
    if not isinstance(filelist, list):
        filelist = glob.glob(filelist)
        print('Files:',filelist)
    assert len(filelist) > 1, 'Did not find enough files to combine'
    if latlon_bounds is None:
        latlon_bounds = len(filelist) * [None]
    else:
        assert len(latlon_bounds)==len(filelist), 'Not enough specified bounds'

    terraindata = [
        dtype(bounds,fpath) for bounds,fpath in zip(latlon_bounds,filelist)
    ]

    # merge rasters
    from rasterio.merge import merge
    merged, out_transform = merge([
        rasterio.open(data.tiffdata) for data in terraindata
    ])

    # write out merged dataset
    profile = rasterio.open(filelist[0]).profile
    print('Raster profile:',profile)
    with rasterio.open(output,'w',**profile) as dst:
        dst.write(merged)

    # get global bounds
    bounds = np.empty((len(filelist),4))
    for i,data in enumerate(terraindata):
        bounds[i,:] = data.bounds
    bounds_min = bounds.min(axis=0)
    bounds_max = bounds.max(axis=0)
    return [bounds_min[0],bounds_min[1],bounds_max[2],bounds_max[3]]

def calc_slope(x,y,z):
    """Calculate local terrain slope based on project grid

    Notes:
    - Uses neighborhood method (weighted second-order difference, based on
      3x3 stencil)
    - Slopes are not calculated at edge points (i.e., locations where a 3x3
      stencil cannot be formed)

    Usage
    =====
    x,y,z : numpy array
        Equally sized 2-D arrays; if not specified, then the full terrain
        will be used
    """
    dx = x[1,0] - x[0,0]
    dy = y[0,1] - y[0,0]
    slope = np.empty_like(z)
    slope[:,:] = np.nan
    z1 = z[  :-2, 2:  ] # upper left
    z2 = z[ 1:-1, 2:  ] # upper middle
    z3 = z[ 2:  , 2:  ] # upper right
    z4 = z[  :-2, 1:-1] # center left
   #z5 = z[ 1:-1, 1:-1] # center
    z6 = z[ 2:  , 1:-1] # center right
    z7 = z[  :-2,  :-2] # lower left
    z8 = z[ 1:-1,  :-2] # lower middle
    z9 = z[ 2:  ,  :-2] # lower right
    dz_dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8*dx)
    dz_dy = ((z1 + 2*z2 + z3) - (z7 + 2*z8 + z9)) / (8*dy)
    rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
    slope[1:-1,1:-1] = np.degrees(np.arctan(rise_run))
    return slope
