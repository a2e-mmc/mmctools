import sys,os
import subprocess
from getpass import getpass
import numpy as np
import pandas as pd

def prompt(s):
    if sys.version_info[0] < 3:
        return raw_input(s)
    else:
        return input(s)


class RDADataset(object):
    """Class to help with downloading initial and boundary conditions
    from the NCAR Research Data Archive (RDA) to use with WPS.

    Users should generally import and use one of the derived classes:
    - FNL
    - ERAInterim
    """
    auth_status = 'auth_status.rda.ucar.edu'
    default_opts = ['-N']

    def __init__(self,emailaddr=None,passwd=None,opts=[],certopts=[]):
        """Setup credentials for downloading datasets from the NCAR
        Research Data Archive

        Additional opts and certopts may be optionally provided.
        """
        if emailaddr is None:
            emailaddr = prompt('RDA email address: ')
        if passwd is None:
            passwd = getpass('RDA password: ')
        self.emailaddr = emailaddr
        self.passwd = self._clean_password(passwd)
        self.opts = self.default_opts + opts
        self.certopts = certopts
        self._get_auth()

    def _clean_password(self,passwd):
        passwd = passwd.replace('&','%26')
        passwd = passwd.replace('?','%3F')
        passwd = passwd.replace('=','%3D')
        return passwd

    def _get_auth(self):
        pid = os.getpid()
        self.cookie = 'auth.rda.ucar.edu.' + str(pid)
        postdata = '--post-data="email={:s}&passwd={:s}&action=login"'.format(
                self.emailaddr,self.passwd)
        cmd = ['wget'] + self.certopts
        cmd += [
            '-O', 'auth_status.rda.ucar.edu',
            '--save-cookies', self.cookie,
            postdata,
            'https://rda.ucar.edu/cgi-bin/login',
        ]
        p = subprocess.run(cmd)
        p.check_returncode() 
        assert os.path.isfile(self.cookie)

    def __del__(self):
        #print('Cleaning up authentication files')
        if os.path.isfile(self.auth_status):
            os.remove(self.auth_status)
        if os.path.isfile(self.cookie):
            os.remove(self.cookie)

    def download(self,urlpath,datetimes,fields=[None],**kwargs):
        """Download specified data at specified datetimes

        Usage
        =====
        urlpath : str
            Path describing data to be downloaded, with valid strftime
            descriptors and optional 'field' format field. E.g.:
            'ds083.2/ei.oper.an.pl/ei.oper.an.pl.{field:s}.%Y%m%d%H'
        datetimes : timestamp or list of timestamps
            Datetime, e.g., output from
            pd.date_range(startdate,enddate,freq='21600s')
        fields : list of str, optional
            Field variable names, e.g., ['regn128sc','regn128uv']
        kwargs : optional
            Additional fields in urlpath to be updated with str.format()
        """
        if not urlpath.startswith('https://'):
            urlpath = 'https://rda.ucar.edu/data/' + urlpath.lstrip('/')
        cmd = ['wget'] + self.certopts + self.opts
        cmd += [
            '--load-cookies', self.cookie,
            'URL_placeholder'
        ]
        if not hasattr(datetimes,'__iter__'):
            datetimes = [pd.to_datetime(datetimes)]
        print('Downloading fields',fields,'at',len(datetimes),'times')
        for datetime in datetimes:
            for field in fields:
                cmd[-1] = datetime.strftime(urlpath)
                urldesc = kwargs
                if field is not None:
                    urldesc['field'] = field
                cmd[-1] = cmd[-1].format(**urldesc)
                p = subprocess.run(cmd)
                p.check_returncode() 


class FNL(RDADataset):
    """NCEP FNL Operational Analysis

    Description: https://rda.ucar.edu/datasets/ds083.2/
    """
    def download(self,datetimes):
        """Download data at specified datetimes.

        Files to download:
        - https://rda.ucar.edu/datasets/ds083.2/grib2/YY/YY.MM/fnl_YYMMDD_HH_MM.grib2

        Usage
        =====
        datetimes : timestamp or list of timestamps
            Datetime, e.g., output from
            pd.date_range(startdate,enddate,freq='21600s')
        """
        super().download(urlpath='ds083.2/grib2/%Y/%Y.%m/fnl_%Y%m%d_%H_%M.grib2',
                         datetimes=datetimes)


class ERAInterim(RDADataset):
    """ERA-Interim Reanalysis

    Note: Production stopped on August 31, 2019

    Description: https://rda.ucar.edu/datasets/ds627.0/
    """

    def download(self,datetimes):
        """Download data at specified datetimes.

        Files to download:
        - https://rda.ucar.edu/datasets/ds627.0/ei.oper.an.pl/YYMM/ei.oper.an.pl.regn128sc.YYMMDDHH
        - https://rda.ucar.edu/datasets/ds627.0/ei.oper.an.pl/YYMM/ei.oper.an.pl.regn128uv.YYMMDDHH
        - https://rda.ucar.edu/datasets/ds627.0/ei.oper.an.sfc/YYMM/ei.oper.an.sfc.regn128sc.YYMMDDHH

        Usage
        =====
        datetimes : timestamp or list of timestamps
            Datetime, e.g., output from
            pd.date_range(startdate,enddate,freq='21600s')
        """
        # pressure-level data
        super().download(urlpath='ds627.0/{prefix:s}/%Y%m/{prefix:s}.{field:s}.%Y%m%d%H',
                         prefix='ei.oper.an.pl',
                         datetimes=datetimes,
                         fields=['regn128sc','regn128uv'])
        # surface data
        super().download(urlpath='ds627.0/{prefix:s}/%Y%m/{prefix:s}.{field:s}.%Y%m%d%H',
                         prefix='ei.oper.an.sfc',
                         datetimes=datetimes,
                         fields=['regn128sc'])


class CDSDataset(object):
    """Class to help with downloading initial and boundary conditions
    from the Copernicus Climate Data Store (CDS) to use with WPS.

    Users should generally import and use one of the derived classes:
    - ERA5
    """
    api_rc = os.path.join(os.environ['HOME'], '.cdsapirc')

    def __init__(self):
        if not os.path.isfile(self.api_rc):
            print('WARNING: '+self.api_rc+' not found')
            print('Go to https://cds.climate.copernicus.eu/api-how-to for more information')
        import cdsapi
        self.client = cdsapi.Client()

    def download(self,datetimes,product,prefix=None,variables=[],
                 pressure_levels=None,area=None):
        """Download data at specified datetimes.

        Usage
        =====
        datetimes : timestamp or list of timestamps
            Datetime, e.g., output from
            pd.date_range(startdate,enddate,freq='21600s')
        product : name
            Name of data product, e.g. "reanalysis-era5-pressure-levels'
        prefix : str, optional
            Filename prefix, which may include an output path, e.g.,
            "/path/to/subset_name" to retrieve a series of files named
            "subset_name_YYYY_MM_DD_HH.grib"
        variables : list
            List of variable names
        pressure_levels : list, optional
            List of pressure levels
        area : list, optional
            North/west/south/east lat/long limits. Default retrieval
            region includes all of US and Central America, most of
            Alaska and Canada (up to 60deg latitude), and parts of
            South America that lie north of the equator.
        """
        if prefix is None:
            prefix = os.path.join('.',product)
        if area is None:
            area = [60, -169, 0, -47]
        req = {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': variables,
            'pressure_level': pressure_levels,
            'area': area, # North, West, South, East.
        }
        if pressure_levels is not None:
            req['pressure_level'] = pressure_levels
            print('Requesting',len(pressure_levels),'pressure levels')
        for datetime in datetimes:
            req['year'] = datetime.strftime('%Y')
            req['month'] = datetime.strftime('%m')
            req['day'] = datetime.strftime('%d')
            req['time'] = datetime.strftime('%H:%M')
            target = datetime.strftime('{:s}_%Y_%m_%d_%H.grib'.format(prefix))
            #print(datetime,req,target)
            self.client.retrieve(product, req, target)

    
class ERA5(CDSDataset):
    """Fifth-generation global atmospheric reanalysis from the European
    Centre for Medium-range Weather Forecasts (ECMWF)

    Improvements over ERA-Interim include:
    - Much higher spatial and temporal resolution
    - Information on variation in quality over space and time
    - Much improved troposphere
    - Improved representation of tropical cyclones
    - Better global balance of precipitation and evaporation
    - Better precipitation over land in the deep tropics
    - Better soil moisture
    - More consistent sea surface temperature and sea ice

    Ref: https://confluence.ecmwf.int/pages/viewpage.action?pageId=74764925
    """

    def download(self,datetimes,path=None,area=None):
        """Download data at specified datetimes.

        Descriptions:
        - https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels
        - https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels

        Usage
        =====
        datetimes : timestamp or list of timestamps
            Datetime, e.g., output from
            pd.date_range(startdate,enddate,freq='21600s')
        path : str, optional
            Path to directory in which to save grib files
        area : list, optional
            North/west/south/east lat/long limits. Default retrieval
            region includes all of US and Central America, most of
            Alaska and Canada (up to 60deg latitude), and parts of
            South America that lie north of the equator.
        """
        if path is None:
            path = '.'
        else:
            os.makedirs(path,exist_ok=True)
        super().download(
            datetimes,
            'reanalysis-era5-pressure-levels',
            prefix=os.path.join(path,'era5_pressure'),
            variables=[
                'divergence','fraction_of_cloud_cover','geopotential',
                'ozone_mass_mixing_ratio','potential_vorticity',
                'relative_humidity','specific_cloud_ice_water_content',
                'specific_cloud_liquid_water_content','specific_humidity',
                'specific_rain_water_content','specific_snow_water_content',
                'temperature','u_component_of_wind','v_component_of_wind',
                'vertical_velocity','vorticity'
            ],
            pressure_levels=[
                '1','2','3','5','7','10','20','30','50','70','100','125','150',
                '175','200','225','250','300','350','400','450','500','550',
                '600','650','700','750','775','800','825','850','875','900',
                '925','950','975','1000'
            ],
            area=area
        )
        super().download(
            datetimes,
            'reanalysis-era5-single-levels',
            prefix=os.path.join(path,'era5_surface'),
            variables=[
                '10m_u_component_of_wind','10m_v_component_of_wind',
                '2m_dewpoint_temperature','2m_temperature',
                'convective_snowfall','convective_snowfall_rate_water_equivalent',
                'ice_temperature_layer_1','ice_temperature_layer_2',
                'ice_temperature_layer_3','ice_temperature_layer_4',
                'land_sea_mask','large_scale_snowfall',
                'large_scale_snowfall_rate_water_equivalent',
                'maximum_2m_temperature_since_previous_post_processing',
                'mean_sea_level_pressure',
                'minimum_2m_temperature_since_previous_post_processing',
                'sea_ice_cover','sea_surface_temperature','skin_temperature',
                'snow_albedo','snow_density','snow_depth','snow_evaporation',
                'snowfall','snowmelt','soil_temperature_level_1',
                'soil_temperature_level_2','soil_temperature_level_3',
                'soil_temperature_level_4','soil_type','surface_pressure',
                'temperature_of_snow_layer','total_column_snow_water',
                'volumetric_soil_water_layer_1','volumetric_soil_water_layer_2',
                'volumetric_soil_water_layer_3','volumetric_soil_water_layer_4'
            ],
            area=area
        )

