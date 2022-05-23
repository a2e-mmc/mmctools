import sys,os
import subprocess
from getpass import getpass
import numpy as np
import pandas as pd
import glob
import xarray as xr
from mmctools.helper_functions import get_nc_file_times
from scipy.interpolate import UnivariateSpline

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

    def download(self,urlpath,datetimes,path=None,fields=[None],**kwargs):
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
        path : str, optional
            Path to directory in which to save grib files
        fields : list of str, optional
            Field variable names, e.g., ['regn128sc','regn128uv']
        kwargs : optional
            Additional fields in urlpath to be updated with str.format()
        """
        if not urlpath.startswith('https://'):
            urlpath = 'https://rda.ucar.edu/data/' + urlpath.lstrip('/')
        cmd = ['wget'] + self.certopts + self.opts
        if path is not None:
            cmd += ['-P', path]
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
    def download(self,datetimes,path=None):
        """Download data at specified datetimes.

        Files to download:
        - https://rda.ucar.edu/datasets/ds083.2/grib2/YY/YY.MM/fnl_YYMMDD_HH_MM.grib2

        Usage
        =====
        datetimes : timestamp or list of timestamps
            Datetime, e.g., output from
            pd.date_range(startdate,enddate,freq='21600s')
        path : str, optional
            Path to directory in which to save grib files
        """
        if path is None:
            path = '.'
        else:
            os.makedirs(path,exist_ok=True)
        super().download(urlpath='ds083.2/grib2/%Y/%Y.%m/fnl_%Y%m%d_%H_%M.grib2',
                         path=path,
                         datetimes=datetimes)


class ERAInterim(RDADataset):
    """ERA-Interim Reanalysis

    Note: Production stopped on September 10, 2019

    Description: https://rda.ucar.edu/datasets/ds627.0/
    """

    def download(self,datetimes,path=None):
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
        path : str, optional
            Path to directory in which to save grib files
        """
        era_interim_end_date = '2019-09-10 12:00:00'
        dates_before_end_of_era = True
        good_dates = datetimes.copy()
        for dt in datetimes:
            if str(dt) > era_interim_end_date:
                print('WARNING: Bad date ({}) - after ERA-Interim EOL ({})'.format(str(dt),era_interim_end_date))
                good_dates = good_dates.drop(dt)
        if len(good_dates) > 0:
            datetimes = good_dates
        else:
            dates_before_end_of_era = False
            print('WARNING: All dates are after ERA-Interim EOL ({})'.format(era_interim_end_date))
            print('Not downloading anything... Need to change reanalysis for these dates!')
        
        if dates_before_end_of_era:
            if path is None:
                path = '.'
            else:
                os.makedirs(path,exist_ok=True)
            # pressure-level data

            super().download(urlpath='ds627.0/{prefix:s}/%Y%m/{prefix:s}.{field:s}.%Y%m%d%H',
                             path=path,
                             prefix='ei.oper.an.pl',
                             datetimes=datetimes,
                             fields=['regn128sc','regn128uv'])
            # surface data
            super().download(urlpath='ds627.0/{prefix:s}/%Y%m/{prefix:s}.{field:s}.%Y%m%d%H',
                             path=path,
                             prefix='ei.oper.an.sfc',
                             datetimes=datetimes,
                             fields=['regn128sc'])

        
class MERRA2(RDADataset):
    """MERRA2 Global Atmosphere Forcing Data

    Description:https://rda.ucar.edu/datasets/ds313.3/
    """
   
    def download(self,datetimes,path=None,resolution_deg=0):
        """Download data at specified datetimes.

        Files to download:
        - https://rda.ucar.edu/data/ds313.3/0.9x1.25/YYYY/MERRA2_0.9x1.25_YYYYMMDD.nc

        Usage
        =====
        datetimes : timestamp or list of timestamps
            Datetime, e.g., output from
            pd.date_range(startdate,enddate,freq='21600s')
        path : str, optional
            Path to directory in which to save grib files
        """

        if resolution_deg == 2:
            self.res_str = '1.9x2.5'
        elif resolution_deg == 1:
            self.res_str = '0.9x1.25'
        elif resolution_deg == 0.5:
            self.res_str = '0.5x0.63'
        else:
            self.res_str = 'orig_res'

        if path is None:
            path = '.'
        else:
            os.makedirs(path,exist_ok=True)

        super().download(urlpath='ds313.3/'+self.res_str+'/%Y/MERRA2_'+self.res_str+'_%Y%m%d.nc',
                         path=path,
                         datetimes=datetimes)

        
        
class CDSDataset(object):
    """Class to help with downloading initial and boundary conditions
    from the Copernicus Climate Data Store (CDS) to use with WPS.

    Users should generally import and use one of the derived classes:
    - ERA5
    """
    api_rc = os.path.join(os.environ['HOME'], '.cdsapirc')

    def __init__(self):
        if not os.path.isfile(self.api_rc):
            raise FileNotFoundError(f"""Expected CDS API key in {self.api_rc}
Go to https://cds.climate.copernicus.eu/api-how-to for more information""")
        try:
            import cdsapi
        except ImportError:
            raise ModuleNotFoundError("""Need CDS API client
Run `conda install -c conda-forge cdsapi`""")
        else:
            self.client = cdsapi.Client()

    def download(self,datetimes,product,prefix=None,
                 variables=[],
                 area=[],
                 pressure_levels=None,
                 combine_request=False):
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
        area : list
            North/west/south/east lat/long limits
        pressure_levels : list, optional
            List of pressure levels
        combine_request : bool, optional
            Aggregate requested dates into lists of years, months, days,
            and hours--note that this may return additional time steps
            because the request selects all permutations of
            year/month/day/hour; should be False for WRF WPS
        """
        if prefix is None:
            prefix = os.path.join('.',product)
        
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
        if combine_request:
            print('Combining all datetimes into a single request')
            req['year'] = sorted(list(set([datetime.strftime('%Y') for datetime in datetimes])))
            req['month'] = sorted(list(set([datetime.strftime('%m') for datetime in datetimes])))
            req['day'] = sorted(list(set([datetime.strftime('%d') for datetime in datetimes])))
            req['time'] = sorted(list(set([datetime.strftime('%H:%M') for datetime in datetimes])))
            target = datetimes[0].strftime('{:s}_from_%Y_%m_%d_%H.grib'.format(prefix))
            self.client.retrieve(product, req, target)
        else:
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

    default_single_level_vars = [
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
        'volumetric_soil_water_layer_3','volumetric_soil_water_layer_4',
        'mean_wave_direction','mean_wave_period',
        'significant_height_of_combined_wind_waves_and_swell',
        'peak_wave_period',
    ]

    default_pressure_level_vars = [
        'divergence','fraction_of_cloud_cover','geopotential',
        'ozone_mass_mixing_ratio','potential_vorticity',
        'relative_humidity','specific_cloud_ice_water_content',
        'specific_cloud_liquid_water_content','specific_humidity',
        'specific_rain_water_content','specific_snow_water_content',
        'temperature','u_component_of_wind','v_component_of_wind',
        'vertical_velocity','vorticity'
    ]

    default_pressure_levels = [
        '1','2','3','5','7','10','20','30','50','70','100','125','150',
        '175','200','225','250','300','350','400','450','500','550',
        '600','650','700','750','775','800','825','850','875','900',
        '925','950','975','1000'
    ]

    def download(self,datetimes,path=None,
                 pressure_level_vars='default', pressure_levels='default',
                 single_level_vars='default',
                 bounds={},
                 combine_request=False):
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
        bounds : dict, optional
            Dictionary with keys N=North, S=South, W=West, and E=East
            for optional lat/long limits. Default retrieval region 
            includes all of US and Central America, most of Alaska 
            and Canada (up to 60deg latitude), and parts of South 
            America that lie north of the equator.
        pressure_level_vars : list, optional
            Variables to retrieve at the specified pressure levels; if
            set to 'default', then use `default_pressure_level_vars`
        pressure_levels : list, optional
            Pressure levels from which 4D data are constructed; if set
            to 'default', then use `default_pressure_levels`
        single_level_vars : list, optional
            Variables to retrieve at the specified pressure levels; if
            set to 'default', then use `default_single_level_vars`
        combine_request : bool, optional
            Aggregate requested dates into lists of years, months, days,
            and hours--note that this may return additional time steps
            because the request selects all permutations of
            year/month/day/hour; should be False for WRF WPS
        """
        if path is None:
            path = '.'
        else:
            os.makedirs(path,exist_ok=True)
            
        N_bound = bounds.get('N', 60)
        S_bound = bounds.get('S', 0)
        W_bound = bounds.get('W', -169)
        E_bound = bounds.get('E', -47)

        if single_level_vars == 'default':
            single_level_vars = self.default_single_level_vars
        if pressure_level_vars == 'default':
            pressure_level_vars = self.default_pressure_level_vars
        if pressure_levels == 'default':
            pressure_levels = self.default_pressure_levels
            
        area = [N_bound, W_bound, S_bound, E_bound]
            
        if pressure_level_vars:
            super().download(
                datetimes,
                'reanalysis-era5-pressure-levels',
                prefix=os.path.join(path,'era5_pressure'),
                variables=pressure_level_vars,
                pressure_levels=pressure_levels,
                area=area,
                combine_request=combine_request,
            )
        if single_level_vars:
            super().download(
                datetimes,
                'reanalysis-era5-single-levels',
                prefix=os.path.join(path,'era5_surface'),
                variables=single_level_vars,
                area=area,
                combine_request=combine_request,
            )


class SetupWRF_old():
    '''
    Set up run directory for WRF / WPS
    '''

    def __init__(self,run_directory,icbc_directory,executables_dict,setup_dict):
        self.setup_dict    = setup_dict
        self.run_dir       = run_directory
        self.wrf_exe_dir   = executables_dict['wrf']
        self.wps_exe_dir   = executables_dict['wps']
        self.icbc_dir      = icbc_directory
        self.icbc_dict     = self._get_icbc_info()
        missing_namelist_options = self._check_namelist_opts()
        if missing_namelist_options != []:
            print('missing these namelist options:',missing_namelist_options)
            return
    
    def _get_icbc_info(self):
        icbc_type = self.setup_dict['icbc_type']
        interval_seconds = 21600
        met_lvls = 38
        soil_lvls = 4
        download_freq = '6h'

        if icbc_type == 'ERA5':
            interval_seconds = 3600
            met_lvls  = 38
            download_freq = '1h'
        elif 'FNL' in icbc_type:
            met_lvls  = 34
        elif icbc_type == 'NARR':
            met_lvls  = 30
        elif icbc_type == 'MERRA2':
            met_lvls  = 73

        icbc_dict = {'type' : icbc_type,
                    'interval_seconds' : interval_seconds,
                    'met_levels' : met_lvls,
                    'soil_levels' : soil_lvls,
                    'download_freq' : download_freq}

        return icbc_dict

    def _check_namelist_opts(self):
        required_fields = ['start_date','end_date','number_of_domains','dxy',
                           'time_step','num_eta_levels','parent_grid_ratio',
                           'parent_time_ratio','istart','jstart','nx','ny',
                           'ref_lat','ref_lon','true_lat1','true_lat2','stand_lon']

        missing_keys = []
        missing_options = False
        setup_keys = list(self.setup_dict.keys())
        for key in required_fields:
            if key not in setup_keys:
                missing_keys.append(key)
                missing_options = True
                
        if not missing_options:
            if 'usgs+' in self.setup_dict['geogrid_args']:
                land_cat = 24
            elif 'usgs_lakes+' in self.setup_dict['geogrid_args']:
                land_cat = 28
            else:
                print('here')
                land_cat = 21
            namelist_defaults = {
                       'geogrid_args' : '30s',
                   'history_interval' : [60,60],
                   'interval_seconds' : self.icbc_dict['interval_seconds'],
                 'num_metgrid_levels' : self.icbc_dict['met_levels'],
            'num_metgrid_soil_levels' : self.icbc_dict['soil_levels'],
                    'input_from_file' : '.true.',
                   'restart_interval' : 360,            
                    'frames_per_file' : 1,            
                              'debug' : 0,            
                        'max_ts_locs' : 20,            
                       'max_ts_level' : self.setup_dict['num_eta_levels'],            
                         'mp_physics' : 10,            
                              'ra_lw' : 4,            
                              'ra_sw' : 4,
                               'radt' : int(self.setup_dict['dxy']/1000.0),
                  'sf_sfclay_physics' : 1,
                 'sf_surface_physics' : 2,
                     'bl_pbl_physics' : 1,
                         'cu_physics' : 1,
                             'isfflx' : 1,
                             'ifsnow' : 0,
                             'icloud' : 0,
               'surface_input_source' : 1,
                    'num_soil_layers' : self.icbc_dict['soil_levels'],
                       'num_land_cat' : land_cat,
                   'sf_urban_physics' : 0,
                          'w_damping' : 1,
                           'diff_opt' : 1,
                             'km_opt' : 4,
                       'diff_6th_opt' : 2,
                    'diff_6th_factor' : 0.12,
                          'base_temp' : 290.0,
                           'damp_opt' : 3,
                              'zdamp' : 5000.0,
                           'dampcoef' : 0.2,
                              'khdif' : 0,
                              'kvdif' : 0,
                              'smdiv' : 0.1,
                    'non_hydrostatic' : '.true.',
                      'moist_adv_opt' : 1,
                     'scalar_adv_opt' : 1,
                        'tke_adv_opt' : 1,
                    'h_mom_adv_order' : 5,
                    'v_mom_adv_order' : 3,
                    'h_sca_adv_order' : 5,
                    'v_sca_adv_order' : 3,
                            'gwd_opt' : 0,
                     'spec_bdy_width' : 5,
                          'spec_zone' : 1,
                         'relax_zone' : 4,
                'nio_tasks_per_group' : 0,
                         'nio_groups' : 1,
                         'sst_update' : 1,
                           'sst_skin' : 0,
                   'sf_ocean_physics' : 0,
            }

            namelist_opts = namelist_defaults
            for key in setup_keys:
                namelist_opts[key] = self.setup_dict[key]
            self.namelist_opts = namelist_opts
        else: 
            raise Exception("The following fields are missing from the setup dictionary: ",missing_keys)

        return missing_keys
    
    def _link_files(self,file_list,destination_dir):
        for filen in file_list:
            file_name = filen.split('/')[-1]
            try:
                os.symlink(filen,'{}{}'.format(destination_dir,file_name))
            except FileExistsError:
                print('file already linked')
            
    def link_executables(self):
        # Create run dir:
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        # Link WPS and WRF files / executables
        wps_files = glob.glob('{}[!n]*'.format(self.wps_exe_dir))
        self._link_files(wps_files,self.run_dir)
        wrf_files = glob.glob('{}[!n]*'.format(self.wrf_exe_dir))
        self._link_files(wrf_files,self.run_dir)
        
    def _get_nl_str(self,num_doms,phys_opt):
        phys_str = ''
        for pp in range(0,num_doms):
            if type(phys_opt) is list:
                phys_str += '{0:>5},'.format(str(phys_opt[pp]))
            else:
                phys_str += '{0:>5},'.format(str(phys_opt))
        return(phys_str)
    
    def write_wps_namelist(self):
        num_doms = self.namelist_opts['number_of_domains']
        start_date_str = "'{}',".format(self.namelist_opts['start_date'].replace(' ','_'))*num_doms
        end_date_str   = "'{}',".format(self.namelist_opts['end_date'].replace(' ','_'))*num_doms
        geog_data_res = "'{}',".format(self.namelist_opts['geogrid_args'])*num_doms
        parent_ids,parent_grid_ratios,dx_str = '','',''
        istart_str,jstart_str,nx_str,ny_str = '','','',''
        for pp,pgr in enumerate(self.namelist_opts['parent_grid_ratio']):
            if pp == 0:
                pid = 1
            else:
                pid = pp
            parent_ids += '{0:>5},'.format(str(pid))
            parent_grid_ratios += '{0:>5},'.format(str(pgr))
            istart_str += '{0:>5},'.format(str(self.namelist_opts['istart'][pp]))
            jstart_str += '{0:>5},'.format(str(self.namelist_opts['jstart'][pp]))
            nx_str += '{0:>5},'.format(str(self.namelist_opts['nx'][pp]))
            ny_str += '{0:>5},'.format(str(self.namelist_opts['ny'][pp]))
        f = open('{}namelist.wps'.format(self.run_dir),'w')
        f.write("&share\n")
        f.write(" wrf_core = 'ARW',\n")
        f.write(" max_dom = {},\n".format(num_doms))
        f.write(" start_date = {}\n".format(start_date_str))
        f.write(" end_date   = {}\n".format(end_date_str))
        f.write(" interval_seconds = {},\n".format(self.namelist_opts['interval_seconds']))
        f.write(" io_form_geogrid = 2,\n")
        f.write("/\n")
        f.write("\n")
        f.write("&geogrid\n")
        f.write(" parent_id         = {}\n".format(parent_ids))
        f.write(" parent_grid_ratio = {}\n".format(parent_grid_ratios))
        f.write(" i_parent_start    = {}\n".format(istart_str))
        f.write(" j_parent_start    = {}\n".format(jstart_str))
        f.write(" e_we              = {}\n".format(nx_str))
        f.write(" e_sn              = {}\n".format(ny_str))
        f.write(" geog_data_res     = {}\n".format(geog_data_res))
        f.write(" dx = {}\n".format(self.namelist_opts['dxy']))
        f.write(" dy = {}\n".format(self.namelist_opts['dxy']))
        f.write(" map_proj = 'lambert',\n")
        f.write(" ref_lat   = {},\n".format(self.namelist_opts['ref_lat']))
        f.write(" ref_lon   = {},\n".format(self.namelist_opts['ref_lon']))
        f.write(" truelat1  = {},\n".format(self.namelist_opts['true_lat1']))
        f.write(" truelat2  = {},\n".format(self.namelist_opts['true_lat2']))
        f.write(" stand_lon = {},\n".format(self.namelist_opts['stand_lon'])) 
        f.write(" geog_data_path = '{}',\n".format(self.namelist_opts['geog_data_path']))
        f.write("/\n")
        f.write("\n")
        f.write("&ungrib\n")
        f.write(" out_format = 'WPS',\n")
        f.write(" prefix = '{}',\n".format(self.namelist_opts['icbc_type'].upper()))
        f.write("/\n")
        f.write("\n")
        f.write("&metgrid\n")
        f.write(" fg_name = '{}',\n".format(self.namelist_opts['icbc_type'].upper()))
        f.write(" io_form_metgrid = 2,\n")
        f.write("! constants_name = 'SST:DATE', \n")
        f.write("/\n")
        f.close()
        

    def write_namelist_input(self):
        
        dt = self.namelist_opts['time_step']
        if (type(dt) is int) or (dt.is_integer()):
            ts_0 = int(dt)
            ts_num = 0
            ts_den = 1
        else:
            ts_0 = 0
            ts_num = dt.as_integer_ratio()[0]
            ts_den = dt.as_integer_ratio()[1]
        
        num_doms = self.namelist_opts['number_of_domains']
        history_interval = self.namelist_opts['history_interval']
        if type(history_interval) is int:
            history_interval_str = '{0:>4},'.format(history_interval)*num_doms
        elif (type(history_interval) is list) or (type(history_interval) is np.ndarray):
            history_interval_str = ''
            for hi in history_interval:
                history_interval_str += '{0:>4},'.format(hi) 

        start_date = pd.to_datetime(self.namelist_opts['start_date'])
        end_date   = pd.to_datetime(self.namelist_opts['end_date'])
        run_hours  = int((end_date - start_date).total_seconds()/3600.0)
        
        start_year_str   = "{0:>5},".format(start_date.year)*num_doms
        start_month_str  = "{0:>5},".format(start_date.month)*num_doms
        start_day_str    = "{0:>5},".format(start_date.day)*num_doms
        start_hour_str   = "{0:>5},".format(start_date.hour)*num_doms
        start_minute_str = "{0:>5},".format(start_date.minute)*num_doms
        start_second_str = "{0:>5},".format(start_date.second)*num_doms

        end_year_str   = "{0:>5},".format(end_date.year)*num_doms
        end_month_str  = "{0:>5},".format(end_date.month)*num_doms
        end_day_str    = "{0:>5},".format(end_date.day)*num_doms
        end_hour_str   = "{0:>5},".format(end_date.hour)*num_doms
        end_minute_str = "{0:>5},".format(end_date.minute)*num_doms
        end_second_str = "{0:>5},".format(end_date.second)*num_doms
        
        parent_ids,grid_ids,dx_str,radt_str = '','','',''
        full_pgr = self.namelist_opts['parent_grid_ratio'].copy()
        for pp,pgr in enumerate(self.namelist_opts['parent_grid_ratio']):
            full_pgr[pp] = np.prod(self.namelist_opts['parent_grid_ratio'][:pp+1])
            grid_ids += '{0:>5},'.format(str(pp+1))
            if pp == 0:
                pid = 1
            else:
                pid = pp
            parent_ids += '{0:>5},'.format(str(pid))
            dx_str += '{0:>5},'.format(str(int(self.namelist_opts['dxy']/full_pgr[pp])))
            radt = self.namelist_opts['radt']/full_pgr[pp]
            if radt < 1: radt = 1
            radt_str += '{0:>5},'.format(str(int(radt)))

        grid_ids           = self._get_nl_str(num_doms,list(range(1,num_doms+1)))
        parent_grid_ratios = self._get_nl_str(num_doms,self.namelist_opts['parent_grid_ratio'])
        parent_time_ratios = self._get_nl_str(num_doms,self.namelist_opts['parent_time_ratio'])
        istart_str         = self._get_nl_str(num_doms,self.namelist_opts['istart'])
        jstart_str         = self._get_nl_str(num_doms,self.namelist_opts['jstart'])
        nx_str             = self._get_nl_str(num_doms,self.namelist_opts['nx'])
        ny_str             = self._get_nl_str(num_doms,self.namelist_opts['ny'])
        
        if 'iofields_filename' in self.namelist_opts.keys():
            include_io = True
            io_str     = self._get_nl_str(num_doms,self.namelist_opts['iofields_filename'])
        else:
            include_io = False
        if 'auxinput4_inname' in self.namelist_opts.keys():
            include_aux4  = True
            aux4_str_name = self.namelist_opts['auxinput4_inname']
            aux4_str_int  = self._get_nl_str(num_doms,self.namelist_opts['auxinput4_interval'])
            if 'io_form_auxinput4' in self.namelist_opts.keys():
                aux4_str_form = self.namelist_opts['io_form_auxinput4']
            else:
                aux4_str_form = '2' 
        else:
            include_aux4 = False

        aux_list = []
        for key in self.namelist_opts.keys():
            if ('auxhist' in key) and ('outname' in key):
                aux_list.append(key.replace('auxhist','').replace('_outname',''))
        include_auxout = False
        if aux_list != []:
            include_auxout = True
            n_aux = len(aux_list)
            aux_block = ''
            for aa,aux in enumerate(aux_list):
                aux_out_str = '{},'.format(self.namelist_opts['auxhist{}_outname'.format(aux)])
                aux_int_str = self._get_nl_str(num_doms,self.namelist_opts['auxhist{}_interval'.format(aux)])
                aux_per_str = self._get_nl_str(num_doms,self.namelist_opts['frames_per_auxhist{}'.format(aux)])
                aux_frm_str = '{},'.format(self.namelist_opts['io_form_auxhist{}'.format(aux)])
                line1 = ' auxhist{}_outname         = {}\n'.format(aux,aux_out_str)
                line2 = ' auxhist{}_interval        = {}\n'.format(aux,aux_int_str)
                line3 = ' frames_per_auxhist{}      = {}\n'.format(aux,aux_per_str)
                line4 = ' io_form_auxhist{}         = {}\n'.format(aux,aux_frm_str)
                aux_block += line1 + line2 + line3 + line4
            
        mp_str      = self._get_nl_str(num_doms,self.namelist_opts['mp_physics'])
        sfclay_str  = self._get_nl_str(num_doms,self.namelist_opts['sf_sfclay_physics'])
        surface_str = self._get_nl_str(num_doms,self.namelist_opts['sf_surface_physics'])
        pbl_str     = self._get_nl_str(num_doms,self.namelist_opts['bl_pbl_physics'])
        cu_str      = self._get_nl_str(num_doms,self.namelist_opts['cu_physics'])
        urb_str     = self._get_nl_str(num_doms,self.namelist_opts['sf_urban_physics'])
        diff_str    = self._get_nl_str(num_doms,self.namelist_opts['diff_opt'])
        km_str      = self._get_nl_str(num_doms,self.namelist_opts['km_opt'])
        diff6o_str  = self._get_nl_str(num_doms,self.namelist_opts['diff_6th_opt'])
        diff6f_str  = self._get_nl_str(num_doms,self.namelist_opts['diff_6th_factor'])
        diff6s_str  = self._get_nl_str(num_doms,self.namelist_opts['diff_6th_slopeopt'])
        zdamp_str   = self._get_nl_str(num_doms,self.namelist_opts['zdamp'])
        damp_str    = self._get_nl_str(num_doms,self.namelist_opts['dampcoef'])
        khdif_str   = self._get_nl_str(num_doms,self.namelist_opts['khdif'])
        kvdif_str   = self._get_nl_str(num_doms,self.namelist_opts['kvdif'])
        smdiv_str   = self._get_nl_str(num_doms,self.namelist_opts['smdiv'])
        nonhyd_str  = self._get_nl_str(num_doms,self.namelist_opts['non_hydrostatic'])
        moist_str   = self._get_nl_str(num_doms,self.namelist_opts['moist_adv_opt'])
        scalar_str  = self._get_nl_str(num_doms,self.namelist_opts['scalar_adv_opt'])
        tke_str     = self._get_nl_str(num_doms,self.namelist_opts['tke_adv_opt'])
        hmom_str    = self._get_nl_str(num_doms,self.namelist_opts['h_mom_adv_order'])
        vmom_str    = self._get_nl_str(num_doms,self.namelist_opts['v_mom_adv_order'])
        hsca_str    = self._get_nl_str(num_doms,self.namelist_opts['h_sca_adv_order'])
        vsca_str    = self._get_nl_str(num_doms,self.namelist_opts['v_sca_adv_order'])
        gwd_str     = self._get_nl_str(num_doms,self.namelist_opts['gwd_opt'])
        if 'shalwater_z0' in self.namelist_opts:
            shalwater_z0_str    = self._get_nl_str(num_doms,self.namelist_opts['shalwater_z0'])
        
        specified = ['.false.']*num_doms
        nested    = ['.true.']*num_doms
        specified[0] = '.true.'
        nested[0]    = '.false.'

        f = open('{}namelist.input'.format(self.run_dir),'w')
        f.write("&time_control\n")
        f.write(" run_days                  =    0,\n")
                
        f.write(" run_hours                 = {0:>5},\n".format(run_hours))
        f.write(" run_minutes               =    0,\n")
        f.write(" run_seconds               =    0,\n")
        f.write(" start_year                =   {}\n".format(start_year_str))
        f.write(" start_month               =   {}\n".format(start_month_str))
        f.write(" start_day                 =   {}\n".format(start_day_str))
        f.write(" start_hour                =   {}\n".format(start_hour_str))
        f.write(" start_minute              =   {}\n".format(start_minute_str))
        f.write(" start_second              =   {}\n".format(start_second_str))
        f.write(" end_year                  =   {}\n".format(end_year_str))
        f.write(" end_month                 =   {}\n".format(end_month_str))
        f.write(" end_day                   =   {}\n".format(end_day_str))
        f.write(" end_hour                  =   {}\n".format(end_hour_str))
        f.write(" end_minute                =   {}\n".format(end_minute_str))
        f.write(" end_second                =   {}\n".format(end_second_str))
        f.write(" interval_seconds          = {},\n".format(self.namelist_opts['interval_seconds']))
        f.write(" input_from_file           = {}\n".format("{},".format(self.namelist_opts['input_from_file'])*num_doms))
        f.write(" restart                   = .false.,\n")
        f.write(" restart_interval          = {},\n".format(self.namelist_opts['restart_interval']))
        f.write(" io_form_history           = 2\n")
        f.write(" io_form_restart           = 2\n")
        f.write(" io_form_input             = 2\n")
        f.write(" io_form_boundary          = 2\n")
        f.write(" history_interval          = {}\n".format(history_interval_str))
        f.write(" frames_per_outfile        = {}\n".format("{0:>5},".format(self.namelist_opts['frames_per_file'])*num_doms))
        if include_io:
            f.write(" iofields_filename         = {}\n".format(io_str))
            f.write(" ignore_iofields_warning   = .true.,\n")
        if include_aux4:
            f.write(" auxinput4_inname          = {},\n".format(aux4_str_name))
            f.write(" auxinput4_interval        = {}\n".format(aux4_str_int))
            f.write(" io_form_auxinput4         = {},\n".format(aux4_str_form))
        if include_auxout:
            f.write(aux_block)
        f.write(" debug_level               = {} \n".format(self.namelist_opts['debug']))
        f.write("/\n")
        f.write("\n")
        f.write("&domains\n")
        f.write(" time_step                 =  {},\n".format(ts_0))
        f.write(" time_step_fract_num       =  {},\n".format(ts_num))
        f.write(" time_step_fract_den       =  {},\n".format(ts_den))
        f.write(" max_dom                   =  {},\n".format(num_doms))
        f.write(" max_ts_locs               =  {},\n".format(self.namelist_opts['max_ts_locs']))
        f.write(" max_ts_level              =  {},\n".format(self.namelist_opts['max_ts_level']))
        f.write(" tslist_unstagger_winds    = .true., \n")
        f.write(" s_we                      =  {}\n".format("{0:>5},".format(1)*num_doms))
        f.write(" e_we                      =  {}\n".format(nx_str))
        f.write(" s_sn                      =  {}\n".format("{0:>5},".format(1)*num_doms))
        f.write(" e_sn                      =  {}\n".format(ny_str))
        f.write(" s_vert                    =  {}\n".format("{0:>5},".format(1)*num_doms))
        f.write(" e_vert                    =  {}\n".format("{0:>5},".format(self.namelist_opts['num_eta_levels'])*num_doms))
        if 'eta_levels' in self.namelist_opts.keys():
            f.write(" eta_levels  = {},\n".format(self.namelist_opts['eta_levels']))
        if 'dzbot' in self.namelist_opts.keys():
            f.write(" dzbot                     = {},\n".format(self.namelist_opts['dzbot']))
        if 'dzstretch_s' in self.namelist_opts.keys():
            f.write(" dzstretch_s               = {},\n".format(self.namelist_opts['dzstretch_s']))
        f.write(" p_top_requested           = {},\n".format(self.namelist_opts['p_top_requested']))
        f.write(" num_metgrid_levels        = {},\n".format(self.namelist_opts['num_metgrid_levels']))
        f.write(" num_metgrid_soil_levels   = {},\n".format(self.namelist_opts['num_metgrid_soil_levels']))
        f.write(" dx                        = {}\n".format(dx_str))
        f.write(" dy                        = {}\n".format(dx_str))
        f.write(" grid_id                   = {}\n".format(grid_ids))
        f.write(" parent_id                 = {}\n".format(parent_ids))
        f.write(" i_parent_start            = {}\n".format(istart_str))
        f.write(" j_parent_start            = {}\n".format(jstart_str))
        f.write(" parent_grid_ratio         = {}\n".format(parent_grid_ratios))
        f.write(" parent_time_step_ratio    = {}\n".format(parent_time_ratios))
        f.write(" feedback                  = {},\n".format(self.namelist_opts['feedback']))
        f.write(" smooth_option             = {},\n".format(self.namelist_opts['smooth_option']))
        if ('nproc_x' in self.namelist_opts.keys()) and ('nproc_y' in self.namelist_opts.keys()):
            f.write(" nproc_x                   = {},\n".format(self.namelist_opts['nproc_x']))
            f.write(" nproc_y                   = {},\n".format(self.namelist_opts['nproc_y']))
        f.write(" /\n")
        f.write("\n")
        f.write("&physics\n")
        f.write(" mp_physics                = {}\n".format(mp_str))
        f.write(" ra_lw_physics             = {}\n".format("{0:>5},".format(self.namelist_opts['ra_lw'])*num_doms))
        f.write(" ra_sw_physics             = {}\n".format("{0:>5},".format(self.namelist_opts['ra_sw'])*num_doms))
        f.write(" radt                      = {}\n".format(radt_str))
        f.write(" sf_sfclay_physics         = {}\n".format(sfclay_str))
        f.write(" sf_surface_physics        = {}\n".format(surface_str))
        f.write(" bl_pbl_physics            = {}\n".format(pbl_str))
        f.write(" bldt                      = {}\n".format("{0:>5},".format(0)*num_doms))
        f.write(" cu_physics                = {}\n".format(cu_str))
        f.write(" cudt                      = {}\n".format("{0:>5},".format(5)*num_doms))
        f.write(" isfflx                    = {}, \n".format(self.namelist_opts['isfflx']))
        f.write(" ifsnow                    = {}, \n".format(self.namelist_opts['ifsnow']))
        f.write(" icloud                    = {}, \n".format(self.namelist_opts['icloud']))
        f.write(" surface_input_source      = {}, \n".format(self.namelist_opts['surface_input_source']))
        f.write(" num_soil_layers           = {}, \n".format(self.namelist_opts['num_soil_layers']))
        f.write(" num_land_cat              = {}, \n".format(self.namelist_opts['num_land_cat']))
        f.write(" sf_urban_physics          = {}\n".format(urb_str))
        f.write(" sst_update                = {}, \n".format(self.namelist_opts['sst_update']))
        f.write(" sst_skin                  = {}, \n".format(self.namelist_opts['sst_skin']))
        f.write(" sf_ocean_physics          = {}, \n".format(self.namelist_opts['sf_ocean_physics']))
        
        if 'shalwater_z0' in self.namelist_opts:
            f.write(" shalwater_z0            = {} \n".format(shalwater_z0_str))
        if 'shalwater_depth' in self.namelist_opts:
            f.write(" shalwater_depth            = {}, \n".format(self.namelist_opts['shalwater_depth']))
        f.write(" /\n")
        
        f.write("\n")
        f.write("&fdda\n")
        if 'fdda_dict' in self.namelist_opts:
            f.write("grid_fdda                           = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['grid_fdda'])))
            f.write('gfdda_inname                        = "{}",\n'.format(self.namelist_opts['fdda_dict']['gfdda_inname']))
            f.write("gfdda_interval_m                    = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['gfdda_interval_m'])))
            f.write("io_form_gfdda                       = {},\n".format(self.namelist_opts['fdda_dict']['io_form_gfdda']))
            f.write("if_no_pbl_nudging_uv                = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['if_no_pbl_nudging_uv'])))
            f.write("if_no_pbl_nudging_t                 = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['if_no_pbl_nudging_t'])))
            f.write("if_no_pbl_nudging_ph                = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['if_no_pbl_nudging_ph'])))
            f.write("if_zfac_uv                          = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['if_zfac_uv'])))
            f.write("k_zfac_uv                           = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['k_zfac_uv'])))
            f.write("if_zfac_t                           = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['if_zfac_t'])))
            f.write("k_zfac_t                            = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['k_zfac_t'])))
            f.write("if_zfac_ph                          = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['if_zfac_ph'])))
            f.write("k_zfac_ph                           = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['k_zfac_ph'])))
            f.write("guv                                 = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['guv'])))
            f.write("gt                                  = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['gt'])))
            f.write("gph                                 = {}\n".format(self._get_nl_str(num_doms,self.namelist_opts['fdda_dict']['gph'])))
            f.write("if_ramping                          = {},\n".format(self.namelist_opts['fdda_dict']['if_ramping']))
            f.write("dtramp_min                          = {},\n".format(self.namelist_opts['fdda_dict']['dtramp_min']))
            f.write("xwavenum                            = {},\n".format(self.namelist_opts['fdda_dict']['xwavenum']))
            f.write("ywavenum                            = {},\n".format(self.namelist_opts['fdda_dict']['ywavenum']))
        f.write("/\n")

        f.write("\n")
        f.write("&dynamics\n")
        if 'hybrid_opt' in self.namelist_opts:
            f.write(" hybrid_opt                = {}, \n".format(self.namelist_opts['hybrid_opt']))
        if 'use_theta_m' in self.namelist_opts:
            f.write(" use_theta_m               = {}, \n".format(self.namelist_opts['use_theta_m']))

        f.write(" w_damping                 = {}, \n".format(self.namelist_opts['w_damping']))
        f.write(" diff_opt                  = {}\n".format(diff_str))
        f.write(" km_opt                    = {}\n".format(km_str))
        f.write(" diff_6th_opt              = {}\n".format(diff6o_str))
        f.write(" diff_6th_factor           = {}\n".format(diff6f_str))
        if 'diff_6th_slopeopt' in self.namelist_opts:
            f.write(" diff_6th_slopeopt         = {}\n".format(diff6s_str))

        f.write(" base_temp                 = {}, \n".format(self.namelist_opts['base_temp']))
        f.write(" damp_opt                  = {}, \n".format(self.namelist_opts['damp_opt']))
        f.write(" zdamp                     = {}\n".format(zdamp_str))
        f.write(" dampcoef                  = {}\n".format(damp_str))
        f.write(" khdif                     = {}\n".format(khdif_str))
        f.write(" kvdif                     = {}\n".format(kvdif_str))
        if 'smdiv' in self.namelist_opts:
            f.write(" smdiv                     = {}\n".format(smdiv_str))
        f.write(" non_hydrostatic           = {}\n".format(nonhyd_str))
        f.write(" moist_adv_opt             = {}\n".format(moist_str))
        f.write(" scalar_adv_opt            = {}\n".format(scalar_str))
        f.write(" tke_adv_opt               = {}\n".format(tke_str))
        f.write(" h_mom_adv_order           = {}\n".format(hmom_str))
        f.write(" v_mom_adv_order           = {}\n".format(vmom_str))
        f.write(" h_sca_adv_order           = {}\n".format(hsca_str))
        f.write(" v_sca_adv_order           = {}\n".format(vsca_str))
        f.write(" gwd_opt                   = {}\n".format(gwd_str))
        f.write(" /\n")
        f.write(" \n")
        f.write("&bdy_control\n")
        f.write(" spec_bdy_width            = {}, \n".format(self.namelist_opts['spec_bdy_width']))
        f.write(" spec_zone                 = {}, \n".format(self.namelist_opts['spec_zone']))
        f.write(" relax_zone                = {}, \n".format(self.namelist_opts['relax_zone']))
        f.write(" specified                 = {}, \n".format(','.join(specified)))
        f.write(" nested                    = {}, \n".format(','.join(nested)))
        f.write("/\n")
        f.write("\n")
        f.write(" &namelist_quilt\n")
        f.write(" nio_tasks_per_group       = {}, \n".format(self.namelist_opts['nio_tasks_per_group']))
        f.write(" nio_groups                = {}, \n".format(self.namelist_opts['nio_groups']))
        f.write(" /\n")
        
    def get_icbcs(self):
        start_time = pd.to_datetime(self.setup_dict['start_date'])
        end_time   = pd.to_datetime(self.setup_dict['end_date'])
        freq       = self.icbc_dict['download_freq']

        datetimes = pd.date_range(start=start_time,
                                  end=end_time,
                                  freq=freq)
        optional_args = {}
        icbc_type = self.namelist_opts['icbc_type'].upper()
        if icbc_type == 'ERAI':
            icbc = ERAInterim()
        elif 'FNL' in icbc_type:
            icbc = FNL()
        elif icbc_type == 'MERRA2':
            print('Cannot download MERRA2 yet... please download manually and put in the IC/BC dir:')
            print(self.icbc_dir)
            #icbc = MERRA2()
            #if 'resolution_deg' not in self.setup_keys():
            #    res_drag = 0
            #else:
            #    res_drag = self.setup_dict['resolution_deg']
            #optional_args['resolution_deg'] = res_drag
        elif icbc_type == 'ERA5':
            icbc = ERA5()
            if 'bounds' not in self.setup_dict.keys():
                bounds = { 'N':60,
                           'S':30,
                           'W':-120,
                           'E':-90}
            else:
                bounds = self.setup_dict['bounds']
            optional_args['bounds'] = bounds
        else:
            print('We currently do not support ',icbc_type)
        if icbc_type != 'MERRA2':
            icbc.download(datetimes,path=self.icbc_dir, **optional_args)
        
        
    def write_submission_scripts(self,submission_dict,hpc='cheyenne'):
        executables = ['wps','real','wrf']
        for executable in executables:
            if hpc == 'cheyenne':
                f = open('{}submit_{}.sh'.format(self.run_dir,executable),'w')
                f.write("#!/bin/bash\n")
                case_str = self.run_dir.split('/')[-3].split('_')[0]
                run_str = '{0}{1}'.format(self.icbc_dict['type'],
                                         (self.setup_dict['start_date'].split(' ')[0]).replace('-',''))
                run_str = '{0}{1}'.format(case_str,
                                         (self.setup_dict['start_date'].split(' ')[0]).replace('-',''))
                f.write("#PBS -N {} \n".format(run_str))
                f.write("#PBS -A {}\n".format(submission_dict['account_key']))
                f.write("#PBS -l walltime={0:02d}:00:00\n".format(submission_dict['walltime_hours'][executable]))
                f.write("#PBS -q economy\n")
                f.write("#PBS -j oe\n")
                f.write("#PBS -m abe\n")
                f.write("#PBS -M {}\n".format(submission_dict['user_email']))
                f.write("### Select 2 nodes with 36 CPUs each for a total of 72 MPI processes\n")
                if executable == 'wps':
                    f.write("#PBS -l select=1:ncpus=1:mpiprocs=1\n")
                else:
                    f.write("#PBS -l select={0:02d}:ncpus=36:mpiprocs=36\n".format(submission_dict['nodes'][executable]))
                f.write("date_start=`date`\n")
                f.write("echo $date_start\n")
                f.write("module list\n")
                if executable == 'wps':
                    icbc_type = self.icbc_dict['type'].upper()
                    if icbc_type == 'ERA5':
                        icbc_head = 'era5_*'
                        icbc_vtable = 'ERA-interim.pl'
                    elif icbc_type == 'ERAI':
                        icbc_head = 'ei.oper*'
                        icbc_vtable = 'ERA-interim.pl'
                    elif 'FNL' in icbc_type:
                        icbc_head = 'fnl_*'
                        icbc_vtable = 'GFS'
                    elif icbc_type == 'MERRA2':
                        icbc_head = 'MERRA2*'
                        icbc_vtable = 'GFS'
                    else:
                        print('We do not support this ICBC yet...')

                    f.write("./geogrid.exe\n".format(executable))
                    icbc_files = '{}{}'.format(self.icbc_dir,icbc_head)
                    if icbc_type == 'MERRA2':
                        f.write('ln -sf {} .\n'.format(icbc_files))
                    else:
                        f.write("ln -sf ungrib/Variable_Tables/Vtable.{} Vtable\n".format(icbc_vtable))
                        f.write("./link_grib.csh {}\n".format(icbc_files))
                        f.write("./ungrib.exe\n".format(executable))
                    f.write("./metgrid.exe\n".format(executable))
                    if icbc_type != 'MERRA2':
                        f.write("for i in GRIBFILE.*; do unlink $i; done\n")
                else:
                    f.write("mpiexec_mpt ./{}.exe\n".format(executable))
                f.write("date_end=`date`\n")
                f.write("echo $date_end\n")
                f.close()

            else:
                print('The hpc requested, {}, is not currently supported... please add it!'.format(hpc))


            
    def write_io_fieldnames(self,vars_to_remove=None,vars_to_add=None):
        if 'iofields_filename' not in self.setup_dict.keys():
            print('iofields_filename not found in setup dict... add a name to allow for creating the file')
            return
        io_names = self.setup_dict['iofields_filename']

        if type(io_names) is str:
            io_names = [io_names]
        if type(io_names) is not list:
            io_names = list(io_names)

        if vars_to_remove is not None:
            assert len(vars_to_remove) ==  len(np.unique(io_names)), \
            'expecting number of io field names ({}) and remove lists {} to be same shape'.format(
                                                    len(np.unique(io_names)),len(vars_to_remove))

        if vars_to_add is not None:
            assert len(vars_to_add) ==  len(np.unique(io_names)), \
            'expecting number of io field names ({}) and add lists {} to be same shape'.format(
                                                    len(np.unique(io_names)),len(vars_to_add))
        rem_str_start = '-:h:0:'
        add_str_start = '+:h:0:'

        for ii,io_name in enumerate(np.unique(io_names)):
            if '"' in io_name:
                io_name = io_name.replace('"','')
            if "'" in io_name:
                io_name = io_name.replace("'",'')
            f = open('{}{}'.format(self.run_dir,io_name),'w')
            line = ''
            if vars_to_remove is not None:
                rem_vars = vars_to_remove[ii]
                var_count = 0
                for rv in rem_vars:
                    line += '{},'.format(rv)
                    if var_count == 7:
                        f.write('{}{}\n'.format(rem_str_start,line))
                        var_count = 0
                        line = ''
                    else:
                        var_count += 1

            if vars_to_add is not None:
                add_vars = vars_to_add[ii]
                var_count = 0
                for av in add_vars:
                    line += '{},'.format(av)
                    if var_count == 7:
                        f.write('{}{}\n'.format(add_str_start,line))
                        var_count = 0
                        line = ''
                    else:
                        var_count += 1

            f.close()
            
            
    def create_submitAll_scripts(self,main_directory,list_of_cases,executables):
        str_of_dirs = ' '.join(list_of_cases)    
        for exe in executables:
            fname = '{}submit_all_{}.sh'.format(main_directory,exe)
            f = open(fname,'w')
            f.write("#!/bin/bash\n")
            f.write("for value in {}\n".format(str_of_dirs))
            f.write("do\n")
            f.write("    cd $value/\n")
            f.write("    pwd\n")
            f.write("    qsub submit_{}.sh\n".format(exe))
            f.write("    cd ..\n")
            f.write("done\n")
            f.close()
            os.chmod(fname,0o755)
            
    def create_tslist_file(self,lat=None,lon=None,i=None,j=None,twr_names=None,twr_abbr=None):
        fname = '{}tslist'.format(self.run_dir)
        write_tslist_file(fname,lat=lat,lon=lon,i=i,j=j,twr_names=twr_names,twr_abbr=twr_abbr)
        
    def link_metem_files(self,met_em_dir):
        # Link WPS and WRF files / executables
        met_files = glob.glob('{}/*'.format(met_em_dir))
        self._link_files(met_files,self.run_dir)


        
        

class SetupWRF():
    '''
    Set up run directory for WRF / WPS
    
    *** For reading in a namelist, have the namelist turn into the namelist_control_dict...
        ... the setup_dict can then be empty and just filled with the "defaults" that are
        ... all the values from the input namelist.
    '''

    def __init__(self,
                 run_directory=None,
                 icbc_directory=None,
                 executables_dict={'wrf':{},'wps':{}},
                 #setup_dict={}
                 ):
            
        #self.setup_dict    = setup_dict
        self.run_dir       = run_directory
        self.wrf_exe_dir   = executables_dict['wrf']
        self.wps_exe_dir   = executables_dict['wps']
        self.icbc_dir      = icbc_directory


    def SetupNamelist(self,setup_dict={},load_namelist_path=None,namelist_type=None):
        self.setup_dict    = setup_dict
        self.namelist_control_dict = self._GetNamelistControlDict()
        if load_namelist_path is not None:
            if type(load_namelist_path) is not list:
                load_namelist_path = [load_namelist_path]
            for namelist_path in load_namelist_path:

                if namelist_type is None:
                    if 'input' in namelist_path.split('/')[-1]:
                        namelist_type = 'input'
                    elif 'wps' in namelist_path.split('/')[-1]:
                        namelist_type = 'wps'
                    else:
                        raise ValueError('Please specify namelist_type as "wps" or "input"')

                self._LoadNamelist(namelist_path,self.namelist_control_dict,self.setup_dict)
        if self.icbc_dir is not None:
            self.namelist_dict = self._CreateNamelistDict()
        return(self.namelist_control_dict)

        
    def _LoadNamelist(self,namelist_path,namelist_control_dict,setup_dict,namelist_type=None):
        if namelist_type is None:
            if 'input' in namelist_path.split('/')[-1]:
                namelist_type = 'input'
            elif 'wps' in namelist_path.split('/')[-1]:
                namelist_type = 'wps'
            else:
                raise ValueError('Please specify namelist_type as "wps" or "input"')
        
        if namelist_type == 'input':
            namelist_sections = ['time_control', 'domains', 'physics', 'fdda', 'dynamics', 'bdy_control', 'namelist_quilt']
        elif namelist_type == 'wps':
            namelist_sections = ['share', 'geogrid', 'ungrib', 'metgrid']
            
        f = open(namelist_path,'r')
        count = 0
        get_eta_levels = False
        for line in f: 
            if '!' in line:
                line = line.split('!')[0]
            if '&' in line:
                section = line.replace('&','').strip()
                value = None
            elif (line.strip() != '/') and (line.strip() != ''):
                if ('=' not in line) and get_eta_levels:
                    line = line.strip()
                    eta_values = line.strip().split(',')
                    if '' in eta_values: eta_values.remove('')
                    for eta in eta_values:
                        eta = ''.join(set(eta.strip().replace('.','')))
                        if eta == '0':
                            get_eta_levels = False

                    eta_level_str += eta_values
                    if get_eta_levels:
                        value = None
                    else:
                        value = ', '.join(eta_level_str)

                else:
                    line = line.strip()
                    line = line.split('=')
                    namelist_var = line[0].strip()
                    value = line[1].strip().split(',')
                    if '' in value: value.remove('')

                    if namelist_var == 'eta_levels':
                        eta_level_str = value
                        value = None
                        get_eta_levels = True

                    else:
                        if len(value) > 1:
                            for vv,val in enumerate(value):
                                val = val.strip()
                                if '.' in val:
                                    try:
                                        val = float(val)
                                    except:
                                        val = val
                                else:
                                    try:
                                        val = int(val)
                                    except:
                                        val = val
                                if (type(val) is not int) and (type(val) is not float):
                                    if '"' in val: val = val.replace('"','')
                                    if "'" in val: val = val.replace("'",'')
                                    if val == '.true.': val = True
                                    if val == '.false.': val = False
                                value[vv] = val
                        else:
                            value = value[0]
                            if '.' in value:
                                try:
                                    value = float(value)
                                except:
                                    value = value
                            else:
                                try:
                                    value = int(value)
                                except:
                                    value = value
                            if (type(value) is not int) and (type(value) is not float):
                                if '"' in value: value = value.replace('"','')
                                if "'" in value: value = value.replace("'",'')
                                if value == '.true.': value = True
                                if value == '.false.': value = False

                if value is not None:
                    namelist_control_dict[section]['required'][namelist_var] = value
            count += 1
        f.close()
        
        for section in list(namelist_sections):
            for namelist_var in list(namelist_control_dict[section]['required'].keys()):
                setup_dict[namelist_var] = namelist_control_dict[section]['required'][namelist_var]
            
        self.setup_dict = setup_dict
        self.namelist_control_dict = namelist_control_dict


    def _GetNamelistControlDict(self):
        aux_in_dict = {'_inname' : '{}_d<domain>', 
                       '_interval' : [None], 
                       'io_form_' : 2,  
                       },
        aux_out_dict = {'_outname' : '{}_d<domain>', 
                        '_interval' : [None], 
                        '_begin' : [None], 
                        '_end' : [None], 
                        'io_form_' : 2,  
                        'frames_per_' : [1],
                        },

        namelist_control_dict = {
                'time_control' : {
                            'required' : {
                                                'run_days' : 0,
                                               'run_hours' : 0,
                                             'run_minutes' : 0,
                                             'run_seconds' : 0,
                                              'start_year' : [None],
                                             'start_month' : [None],
                                               'start_day' : [None],
                                              'start_hour' : [None],
                                            'start_minute' : [None],
                                            'start_second' : [None],
                                                'end_year' : [None],
                                               'end_month' : [None],
                                                 'end_day' : [None],
                                                'end_hour' : [None],
                                              'end_minute' : [None],
                                              'end_second' : [None],
                                        'interval_seconds' : None,
                                        'history_interval' : [60],
                                      'frames_per_outfile' : [1],
                                         'io_form_history' : 2,
                                                 'restart' : False,
                                        'restart_interval' : 0,
                                         'io_form_restart' : 2,
                                           'io_form_input' : 2,
                                        'io_form_boundary' : 2,
                                          },
                             'optional' : {
                                         'input_from_file' : [True],
                                       'fine_input_stream' : [0],
                                                 'cycling' : False,
                                  'reset_simulation_start' : False,
                                              'ncd_nofill' : True,
                                    'frames_per_emissfile' : 12,
                                          'io_style_emiss' : 1,
                                              'diag_print' : 0,
                                            'all_ic_times' : False,
                                     'adjust_output_times' : False,
                                 'override_restart_timers' : False,
                                    'write_hist_at_0h_rst' : False,
                                       'output_ready_flag' : True,
                                      'force_use_old_data' : False,
                                                'auxinput' : aux_in_dict,
                                                 'auxhist' : aux_out_dict,
                                         'nwp_diagnostics' : 0,
                                      'output_diagnostics' : 0,
                                       'iofields_filename' : [None],
                                 'ignore_iofields_warning' : True,
                                             'debug_level' : 0,
                                          },

                            },

                'domains' : {
                            'required' : {
                                            'time_step' : None, 
                                  'time_step_fract_num' : 0,  
                                  'time_step_fract_den' : 1,  
                                              'max_dom' : None,  
                                                 's_we' : [1], 
                                                 'e_we' : [None],
                                                 's_sn' : [1],
                                                 'e_sn' : [None],
                                               's_vert' : [1], 
                                               'e_vert' : [None],
                                      'p_top_requested' : 10000,
                                   'num_metgrid_levels' : 73, 
                              'num_metgrid_soil_levels' : 4,
                                                   'dx' : [None],
                                                   'dy' : [None],
                                              'grid_id' : [1,2,3,4,5,6,7,8,9,10],
                                            'parent_id' : [1,1,2,3,4,5,6,7,8,9],
                                    'parent_grid_ratio' : None,
                               'parent_time_step_ratio' : [None],
                                       'i_parent_start' : [1],
                                       'j_parent_start' : [1],
                                                 'e_we' : [None],
                                                 'e_sn' : [None],
                                            'feedback'  : 0,
                                        'smooth_option' : 0,
                               'tslist_unstagger_winds' : True, 
                                         },
                            'optional' : {
                              'tslist_turbulent_output' : 0,
                                           'eta_levels' : None,
                                          'max_ts_locs' :  20, 
                                         'max_ts_level' :  20, 
                                          'ts_buf_size' : 200,
                                          'dzstretch_s' : 1.3,
                                                'dzbot' : 50.0,
                                              'nproc_x' : None,
                                              'nproc_y' : None,
                                         },
                            },

                'physics' : {
                            'required' : { 
                                           'mp_physics' : [5],
                                        'ra_lw_physics' : [4],
                                        'ra_sw_physics' : [4], 
                                                 'radt' : [None], ### Calculate from ∆x
                                    'sf_sfclay_physics' : [1],
                                   'sf_surface_physics' : [1],
                                       'bl_pbl_physics' : [1],
                                                 'bldt' : [0],
                                           'cu_physics' : [1],
                                                 'cudt' : [5],
                                               'isfflx' : 1,  
                                               'ifsnow' : 0,  
                                               'icloud' : 1,  
                                 'surface_input_source' : 1,  
                                      'num_soil_layers' : 4, 
                                         'num_land_cat' : 21, ### From ICBC

                                        },
                             'optional' : {
                                     'sf_urban_physics' : [0],
                                           'sst_update' : 1,  
                                             'sst_skin' : 0,  
                                     'sf_ocean_physics' : 0,  
                                         'shalwater_z0' : [0], 
                                      'shalwater_depth' : 40, 
                                        },
                            },

                'fdda' : {
                            'required' : {},
                            'optional' : {
                                            'grid_fdda' : [2],
                                         'gfdda_inname' : "wrffdda_d<domain>",
                                       'gfdda_interval' : [6],
                                        'io_form_gfdda' : 2,
                                 'if_no_pbl_nudging_uv' : [1],
                                  'if_no_pbl_nudging_t' : [1], 
                                 'if_no_pbl_nudging_ph' : [1], 
                                           'if_zfac_uv' : [1], 
                                            'k_zfac_uv' : [None], 
                                            'if_zfac_t' : [1], 
                                             'k_zfac_t' : [None],
                                           'if_zfac_ph' : [1],
                                            'k_zfac_ph' : [None], 
                                                  'guv' : [0.0003],
                                                   'gt' : [0.0003],
                                                  'gph' : [0.0003],
                                           'if_ramping' : 1,
                                           'dtramp_min' : 60.0,
                                             'xwavenum' : 7,
                                             'ywavenum' : 7,
                                         },
                            },

                'dynamics' : {
                            'required' : {
                                           'hybrid_opt' : 0,
                                          'use_theta_m' : 0,
                                             'diff_opt' : [1],
                                               'km_opt' : [4],
                                         'diff_6th_opt' : [2],
                                      'diff_6th_factor' : [0.12],
                                            'base_temp' : 290.0,
                                             'damp_opt' : 3,
                                               'zdamp'  : [5000.0],
                                             'dampcoef' : [0.2],
                                                'khdif' : [0],
                                                'kvdif' : [0],
                                                'smdiv' : [0.1],
                                      'non_hydrostatic' : [True],
                                        'moist_adv_opt' : [1],
                                       'scalar_adv_opt' : [1],
                                          'tke_adv_opt' : [1],
                                      'h_mom_adv_order' : [5],
                                      'v_mom_adv_order' : [3],
                                      'h_sca_adv_order' : [5],
                                      'v_sca_adv_order' : [3],
                                         },
                             'optional' : {
                                            'w_damping' : 1,
                                              'gwd_opt' : [1],
                                    'diff_6th_slopeopt' : [0],
                                            'cell_pert' : [False],
                                            'cell_tvcp' : [False],
                                            'pert_tsec' : [100.],
                                         'cell_kbottom' : [3],  
                                                'm_opt' : [0],
                                                  'c_k' : [0.1],
                                           'm_pblh_opt' : [0],
                                              'cpm_opt' : [0],
                                            'cpm_lim_z' : 0.0,
                                               'cpm_eb' : [0],
                                               'cpm_wb' : [0],
                                               'cpm_nb' : [0],
                                               'cpm_sb' : [0],
                                        },
                            },

                'bdy_control' : {
                            'required' : {
                                       'spec_bdy_width' : 5,  
                                            'spec_zone' : 1,  
                                           'relax_zone' : 4,  
                                            'specified' : [True,False,False,False,False,False,False,False,False,False],
                                               'nested' : [False,True,True,True,True,True,True,True,True,True],
                                         },
                            },

                'namelist_quilt' : {
                            'required' : {
                                  'nio_tasks_per_group' : 0,  
                                           'nio_groups' : 1,
                                         },
                            },

                'share' : {
                        'required' : {
                            'wrf_core' : 'ARW',
                            'max_dom' : 1,
                            'start_date' : [None],
                            'end_date' : [None],
                            'interval_seconds' : None,
                            'io_form_geogrid' : 2,
                                     },
                            },
                'geogrid' : {
                        'required' : {
                            'parent_id' : [1,1,2,3,4,5,6,7,8,9],
                            'parent_grid_ratio' : [1],
                            'i_parent_start' : [1],
                            'j_parent_start' : [1],
                            'e_we' : [None],
                            'e_sn' : [None],
                            'geog_data_res' : ['30s'],
                            'dx' : None,
                            'dy' : None,
                            'map_proj' : 'lambert',
                            'ref_lat' : None,
                            'ref_lon' : None,
                            'truelat1' : None,
                            'truelat2' : None,
                            'stand_lon' : None,
                            'geog_data_path' : None,
                                     },
                            },
                'ungrib' : {
                        'required' : {
                            'out_format' : 'WPS',
                            'prefix' : None,   ### From IC/BC               
                                     },
                            },
                'metgrid' : {            
                        'required' : {
                            'fg_name' : None,  ### From IC/BC
                            'io_form_metgrid' : 2,
                                     },
                            },
        }
        return(namelist_control_dict)
        
        
    def _CreateNamelistDict(self):
        namelist_control_dict = self.namelist_control_dict
        namelist_dict = self.setup_dict.copy()
        namelist_dict = self._GetICBCinfo(namelist_dict,namelist_dict['icbc_type'])
        namelist_dict = self._CheckForRequiredFields(namelist_dict,namelist_control_dict)
        namelist_dict = self._MaxDomainAdjustment(namelist_dict,namelist_control_dict)
        return(namelist_dict)

    def _link_files(self,file_list,destination_dir):
        for filen in file_list:
            file_name = filen.split('/')[-1]
            try:
                os.symlink(filen,'{}{}'.format(destination_dir,file_name))
            except FileExistsError:
                print('file already linked')
                
    def CreateRunDirectory(self,auxdir=None):
        # Create run dir:
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        if auxdir is not None:
            os.makedirs('{}{}'.format(self.run_dir,auxdir),exist_ok=True)
            
        # Link WPS and WRF files / executables
        wps_files = glob.glob('{}[!n]*'.format(self.wps_exe_dir))
        self._link_files(wps_files,self.run_dir)
        wrf_files = glob.glob('{}[!n]*'.format(self.wrf_exe_dir))
        self._link_files(wrf_files,self.run_dir)
        
    def _GetICBCinfo(self,namelist_dict,icbc_type):
        interval_seconds = 21600
        met_lvls = 38
        soil_lvls = 4
        download_freq = '6h'

        if icbc_type == 'ERA5':
            interval_seconds = 3600
            met_lvls  = 38
            download_freq = '1h'
        elif 'FNL' in icbc_type:
            met_lvls  = 34
        elif icbc_type == 'NARR':
            met_lvls  = 30
        elif icbc_type == 'MERRA2':
            met_lvls  = 73
        else:
            if icbc_type != 'ERAI':
                raise ValueError('{} is not yet supported. Try ERA5, FNL, NARR, or MERRA2'.format(icbc_type))

        icbc_dict = {'type' : icbc_type,
                    'interval_seconds' : interval_seconds,
                    'met_levels' : met_lvls,
                    'soil_levels' : soil_lvls,
                    'download_freq' : download_freq}
        self.icbc_dict = icbc_dict

        namelist_dict['interval_seconds']        = icbc_dict['interval_seconds']
        namelist_dict['num_metgrid_levels']      = icbc_dict['met_levels']
        namelist_dict['num_metgrid_soil_levels'] = icbc_dict['soil_levels']
        namelist_dict['num_soil_layers']         = int(icbc_dict['soil_levels'])
        namelist_dict['prefix']  = icbc_type
        namelist_dict['fg_name'] = icbc_type

        return namelist_dict
    
    def get_icbcs(self):
        start_time = pd.to_datetime(self.namelist_dict['start_date'])
        end_time   = pd.to_datetime(self.namelist_dict['end_date'])
        freq       = self.icbc_dict['download_freq']

        datetimes = pd.date_range(start=start_time,
                                  end=end_time,
                                  freq=freq)
        optional_args = {}
        icbc_type = self.setup_dict['icbc_type'].upper()
        if icbc_type == 'ERAI':
            icbc = ERAInterim()
        elif 'FNL' in icbc_type:
            icbc = FNL()
        elif icbc_type == 'MERRA2':
            print('Cannot download MERRA2 yet... please download manually and put in the IC/BC dir:')
            print(self.icbc_dir)
            #icbc = MERRA2()
            #if 'resolution_deg' not in self.setup_keys():
            #    res_drag = 0
            #else:
            #    res_drag = self.setup_dict['resolution_deg']
            #optional_args['resolution_deg'] = res_drag
        elif icbc_type == 'ERA5':
            icbc = ERA5()
            if 'bounds' not in self.setup_dict.keys():
                bounds = { 'N':60,
                           'S':30,
                           'W':-120,
                           'E':-90}
            else:
                bounds = self.setup_dict['bounds']
            optional_args['bounds'] = bounds
        else:
            print('We currently do not support ',icbc_type)
        if icbc_type != 'MERRA2':
            icbc.download(datetimes,path=self.icbc_dir, **optional_args)

    def write_submission_scripts(self,submission_dict,hpc='cheyenne',restart_args=False):
        executables = ['wps','real','wrf']
        for executable in executables:
            if hpc == 'cheyenne':
                f = open('{}submit_{}.sh'.format(self.run_dir,executable),'w')
                f.write("#!/bin/bash\n")
                case_str = self.run_dir.split('/')[-3].split('_')[0]
                if (type(self.setup_dict['start_date']) is list): 
                    start_date = self.setup_dict['start_date'][0]
                else:
                    start_date = self.setup_dict['start_date']
                

                run_str = '{0}{1}'.format(case_str,
                                         (start_date.split(' ')[0]).replace('-',''))
                f.write("#PBS -N {} \n".format(run_str))
                f.write("#PBS -A {}\n".format(submission_dict['account_key']))
                f.write("#PBS -l walltime={0:02d}:00:00\n".format(submission_dict['walltime_hours'][executable]))
                f.write("#PBS -q economy\n")
                f.write("#PBS -j oe\n")
                f.write("#PBS -m abe\n")
                f.write("#PBS -M {}\n".format(submission_dict['user_email']))
                f.write("### Select 2 nodes with 36 CPUs each for a total of 72 MPI processes\n")
                if executable == 'wps':
                    args_line = "#PBS -l select=1:ncpus=1:mpiprocs=1\n"
                else:
                    args_line = "#PBS -l select={0:02d}:ncpus=36:mpiprocs=36".format(submission_dict['nodes'][executable])
                    if 'optional_args' in list(submission_dict.keys()):
                        if submission_dict['optional_args'][executable] is not None:
                            args_line += ':{}'.format(submission_dict['optional_args'][executable])
                    args_line += '\n'
                f.write(args_line)
                f.write("date_start=`date`\n")
                f.write("echo $date_start\n")
                f.write("module list\n")
                if executable == 'wps':
                    icbc_type = self.icbc_dict['type'].upper()
                    if icbc_type == 'ERA5':
                        icbc_head = 'era5_*'
                        icbc_vtable = 'ERA-interim.pl'
                    elif icbc_type == 'ERAI':
                        icbc_head = 'ei.oper*'
                        icbc_vtable = 'ERA-interim.pl'
                    elif 'FNL' in icbc_type:
                        icbc_head = 'fnl_*'
                        icbc_vtable = 'GFS'
                    elif icbc_type == 'MERRA2':
                        icbc_head = 'MERRA2*'
                        icbc_vtable = 'GFS'
                    else:
                        print('We do not support this ICBC yet...')

                    f.write("./geogrid.exe\n".format(executable))
                    icbc_files = '{}{}'.format(self.icbc_dir,icbc_head)
                    if icbc_type == 'MERRA2':
                        f.write('ln -sf {} .\n'.format(icbc_files))
                    else:
                        f.write("ln -sf ungrib/Variable_Tables/Vtable.{} Vtable\n".format(icbc_vtable))
                        f.write("./link_grib.csh {}\n".format(icbc_files))
                        f.write("./ungrib.exe\n".format(executable))
                    f.write("./metgrid.exe\n".format(executable))
                    if icbc_type != 'MERRA2':
                        f.write("for i in GRIBFILE.*; do unlink $i; done\n")
                else:
                    if restart_args:
                        f.write('\nRESTART="A"\n\n')
                        f.write('RESTARTDIR="RESTART_$RESTART"\n')
                        f.write('mkdir $RESTARTDIR\n')
                        f.write('cp namelist.input_$RESTART namelist.input\n')
                    f.write("mpiexec_mpt ./{}.exe\n".format(executable))
                    if restart_args:
                        f.write('mv *.d0?.[!n][!c] $RESTARTDIR/.\n')
                f.write("date_end=`date`\n")
                f.write("echo $date_end\n")
                f.close()

            else:
                print('The hpc requested, {}, is not currently supported... please add it!'.format(hpc))

    def write_io_fieldnames(self,io_fields):
        
        if 'iofields_filename' not in self.setup_dict.keys():
            print('iofields_filename not found in setup dict... add a name to allow for creating the file')
            return
        
        if type(io_fields) is dict:
            # Only one io fields dict... needs to have remove or add:
            if sorted(list(io_fields.keys())) != (sorted(['add','remove'])):
           
                io_names = list(io_fields.keys()) #
                
                if sorted(io_names) != sorted(np.unique(self.setup_dict['iofields_filename'])):
                    print('io_fields keys do not match the iofields_filename from the setup dict')
                    raise ValueError ('{} must match {} from setup dict'.format(
                                        io_names,np.unique(self.setup_dict['iofields_filename'])))
                else:
                    io_fields_dict = io_fields
            else:
                io_names = np.unique(self.setup_dict['iofields_filename'])
                if len(io_names) != 1:
                    raise ValueError ('Number of iofields_filename in setup dict is greater than 1. Please specify multiple io_fields.')
                else:
                    io_fields_dict = {io_names:io_fields}
                    
        elif type(io_fields) is list:
            for item in io_fields:
                if type(item) is not dict:
                    raise ValueError ('Specified list for io_fields must be a list of dictionaries.')
            io_names = np.unique(self.setup_dict['iofields_filename'])
            if len(io_fields) != len(io_names):
                raise ValueError ('Number of io_fields specified does not match the number of iofields_filename in setup dict')
            else:
                io_fields_dict = {}
                for ii,item in enumerate(io_fields):
                    io_fields_dict[io_names[ii]] = item
                    
        else:
            raise ValueError ('io_fields must be a list or dictionary')
            
        rem_str_start = '-:h:{}:'
        add_str_start = '+:h:{}:'

        io_names = list(io_fields_dict.keys())

        max_vars_on_line = 8
        for ii,io_name in enumerate(np.unique(io_names)):
            
            if '"' in io_name:
                io_name = io_name.replace('"','')
            if "'" in io_name:
                io_name = io_name.replace("'",'')
            f = open('{}{}'.format(self.run_dir,io_name),'w')
            
            available_keys = list(io_fields_dict[io_name].keys())
            line = ''
            var_count = 0
            if 'remove' in available_keys:
               
                streams_to_remove = list(io_fields_dict[io_name]['remove'].keys())
                for stream in streams_to_remove:
                    vars_to_remove = io_fields_dict[io_name]['remove'][stream]
                    for rv in vars_to_remove:
                        line += '{},'.format(rv)
                        if var_count == max_vars_on_line:
                            f.write('{}{}\n'.format(rem_str_start.format(stream),line))
                            var_count = 0
                            line = ''
                        else:
                            var_count += 1
                    if line != '':
                        f.write('{}{}\n'.format(rem_str_start.format(stream),line))
                        line = ''
                        var_count = 0
                
            if 'add' in available_keys:
               
                streams_to_add = list(io_fields_dict[io_name]['add'].keys())
                for stream in streams_to_add:
                    vars_to_add = io_fields_dict[io_name]['add'][stream]
                    for av in vars_to_add:
                        line += '{},'.format(av)
                        if var_count == max_vars_on_line:
                            f.write('{}{}\n'.format(add_str_start.format(stream),line))
                            var_count = 0
                            line = ''
                        else:
                            var_count += 1
                    if line != '':
                        f.write('{}{}\n'.format(add_str_start.format(stream),line))
                        line = ''
                        var_count = 0


            f.close()

    def write_io_fieldnames_old(self,vars_to_remove=None,vars_to_add=None):
        if 'iofields_filename' not in self.setup_dict.keys():
            print('iofields_filename not found in setup dict... add a name to allow for creating the file')
            return
        io_names = self.setup_dict['iofields_filename']

        if type(io_names) is str:
            io_names = [io_names]
        if type(io_names) is not list:
            io_names = list(io_names)

        if vars_to_remove is not None:
            assert len(vars_to_remove) ==  len(np.unique(io_names)), \
            'expecting number of io field names ({}) and remove lists {} to be same shape'.format(
                                                    len(np.unique(io_names)),len(vars_to_remove))

        if vars_to_add is not None:
            assert len(vars_to_add) ==  len(np.unique(io_names)), \
            'expecting number of io field names ({}) and add lists {} to be same shape'.format(
                                                    len(np.unique(io_names)),len(vars_to_add))
        rem_str_start = '-:h:0:'
        add_str_start = '+:h:0:'

        for ii,io_name in enumerate(np.unique(io_names)):
            if '"' in io_name:
                io_name = io_name.replace('"','')
            if "'" in io_name:
                io_name = io_name.replace("'",'')
            f = open('{}{}'.format(self.run_dir,io_name),'w')
            line = ''
            if vars_to_remove is not None:
                rem_vars = vars_to_remove[ii]
                var_count = 0
                for rv in rem_vars:
                    line += '{},'.format(rv)
                    if var_count == 7:
                        f.write('{}{}\n'.format(rem_str_start,line))
                        var_count = 0
                        line = ''
                    else:
                        var_count += 1

            if vars_to_add is not None:
                add_vars = vars_to_add[ii]
                var_count = 0
                for av in add_vars:
                    line += '{},'.format(av)
                    if var_count == 7:
                        f.write('{}{}\n'.format(add_str_start,line))
                        var_count = 0
                        line = ''
                    else:
                        var_count += 1

            f.close()
            
    def create_submitAll_scripts(self,main_directory,list_of_cases,executables):
        str_of_dirs = ' '.join(list_of_cases)    
        for exe in executables:
            fname = '{}submit_all_{}.sh'.format(main_directory,exe)
            f = open(fname,'w')
            f.write("#!/bin/bash\n")
            f.write("for value in {}\n".format(str_of_dirs))
            f.write("do\n")
            f.write("    cd $value/\n")
            f.write("    pwd\n")
            f.write("    qsub submit_{}.sh\n".format(exe))
            f.write("    cd ..\n")
            f.write("done\n")
            f.close()
            os.chmod(fname,0o755)

    def create_tslist_file(self,lat=None,lon=None,i=None,j=None,twr_names=None,twr_abbr=None,preV4p3=False):
        fname = '{}tslist'.format(self.run_dir)
        write_tslist_file(fname,lat=lat,lon=lon,i=i,j=j,twr_names=twr_names,twr_abbr=twr_abbr,preV4p3=preV4p3)
        
    def link_metem_files(self,met_em_dir):
        # Link WPS and WRF files / executables
        met_files = glob.glob('{}/*'.format(met_em_dir))
        if met_files == []:
            raise ValueError('No met_em files found in {}. Please check that this is where the met_em files are stored'.format(met_em_dir))
        self._link_files(met_files,self.run_dir)

    def _InvertControlDict(self,namelist_control_dict):
        namelist_var_dict = {}
        for section in list(namelist_control_dict.keys()):
            for opt in list(namelist_control_dict[section].keys()):
                for namelist_var in list(namelist_control_dict[section][opt].keys()):
                    if type(namelist_control_dict[section][opt][namelist_var]) is list:
                        max_dom = True
                    else:
                        max_dom = False
                    namelist_var_dict[namelist_var] = {'section':section,'required':(opt=='required'),'max_dom':max_dom}
        return(namelist_var_dict)
        
    def _CheckMissing(self,have,need,namelist_dict):
        missing = []

        for key in need:
            if key not in have:
                missing.append(key)
            found_key = True
            try: 
                namelist_dict[key]
            except:
                found_key = False

            if found_key:
                if type(namelist_dict[key]) is list:
                    if None in namelist_dict[key]: 
                        missing.append(key)
                else:
                    if namelist_dict[key] is None: missing.append(key)
        
        return(missing)

    def _CheckForRequiredFields(self,namelist_dict,namelist_control_dict,namelist_type='input'):
        
        if namelist_type == 'input':
            namelist_sections = ['time_control', 'domains', 'physics', 'fdda', 'dynamics', 'bdy_control', 'namelist_quilt']
        elif namelist_type == 'wps':
            namelist_sections = ['share', 'geogrid', 'ungrib', 'metgrid']
            
        setup_keys = list(self.setup_dict.keys())
        missing_keys = {}
        missing_key_flag = False

        try:
            max_dom = self.setup_dict['max_dom']
        except:
            raise ValueError('max_dom must be specified in the setup_dict')

        try:
            max_dom = int(max_dom)
        except:
            raise ValueError('max_dom must be an integer')
            
        namelist_dict['max_dom'] = max_dom

        got_start_and_end_dates = (('start_date' in setup_keys) and ('end_date' in setup_keys))

        got_run_time = (('run_days' in setup_keys) and ('run_hours' in setup_keys))

        got_start_and_end_times = (('start_year' in setup_keys) and ('end_year' in setup_keys))
        
        got_enough_time_information = got_start_and_end_dates or got_start_and_end_times
        
        if not got_enough_time_information:
            raise ValueError('Not enough time information to create a run. Please add start_date/end_date')

        if got_start_and_end_dates:
            sim_start = self.setup_dict['start_date']
            sim_end   = self.setup_dict['end_date']
        elif got_start_and_end_times:
            date_fmt = '{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}'
            if type(self.setup_dict['start_year']) is not list:
                self.setup_dict['start_year'] = [self.setup_dict['start_year']]
                self.setup_dict['start_month'] = [self.setup_dict['start_month']]
                self.setup_dict['start_day'] = [self.setup_dict['start_day']]
                self.setup_dict['start_hour'] = [self.setup_dict['start_hour']]
                self.setup_dict['start_minute'] = [self.setup_dict['start_minute']]
                self.setup_dict['start_second'] = [self.setup_dict['start_second']]
                self.setup_dict['end_year'] = [self.setup_dict['end_year']]
                self.setup_dict['end_month'] = [self.setup_dict['end_month']]
                self.setup_dict['end_day'] = [self.setup_dict['end_day']]
                self.setup_dict['end_hour'] = [self.setup_dict['end_hour']]
                self.setup_dict['end_minute'] = [self.setup_dict['end_minute']]
                self.setup_dict['end_second'] = [self.setup_dict['end_second']]
                
            sim_start,sim_end = [],[]
            for dd in range(0,len(self.setup_dict['start_year'])):
                sim_start.append(date_fmt.format(int(self.setup_dict['start_year'][dd]),
                                                 int(self.setup_dict['start_month'][dd]),
                                                 int(self.setup_dict['start_day'][dd]),
                                                 int(self.setup_dict['start_hour'][dd]),
                                                 int(self.setup_dict['start_minute'][dd]),
                                                 int(self.setup_dict['start_second'][dd])))
                sim_end.append(date_fmt.format(int(self.setup_dict['end_year'][dd]),
                                               int(self.setup_dict['end_month'][dd]),
                                               int(self.setup_dict['end_day'][dd]),
                                               int(self.setup_dict['end_hour'][dd]),
                                               int(self.setup_dict['end_minute'][dd]),
                                               int(self.setup_dict['end_second'][dd])))

            self.setup_dict['start_date'] = sim_start
            self.setup_dict['end_date'] = sim_end

        if (type(sim_start) is list) and (type(sim_end) is not list):
            sim_end = [sim_end]*len(sim_start)
        if (type(sim_start) is str) or (type(sim_start) is list): 
            if type(sim_start) is list:
                for ss,sim_s in enumerate(sim_start):
                    sim_start[ss] = sim_s.replace('_',' ')
            else:
                sim_start = sim_start.replace('_',' ')
            sim_start=pd.to_datetime(sim_start)
        if (type(sim_end) is str) or (type(sim_end) is list): 
            if type(sim_end) is list:
                for se,sim_e in enumerate(sim_end):
                    sim_end[se] = sim_e.replace('_',' ')
            else:
                sim_end = sim_end.replace('_',' ')
            sim_end=pd.to_datetime(sim_end)
            
        if not got_run_time:
            run_time = sim_end - sim_start
            if np.shape(run_time) != ():
                run_time = max(run_time)
            namelist_dict['run_days'] = int(run_time.components.days)
            namelist_dict['run_hours'] = int(run_time.components.hours)
            namelist_dict['run_minutes'] = int(run_time.components.minutes)
            namelist_dict['run_seconds'] = int(run_time.components.seconds)
            
        if not got_start_and_end_times:
            if np.shape(sim_start) != ():
                sim_start_y = list(sim_start.year)
                sim_start_m = list(sim_start.month)
                sim_start_d = list(sim_start.day)
                sim_start_H = list(sim_start.hour)
                sim_start_M = list(sim_start.minute)
                sim_start_S = list(sim_start.second)
                sim_end_y = list(sim_end.year)
                sim_end_m = list(sim_end.month)
                sim_end_d = list(sim_end.day)
                sim_end_H = list(sim_end.hour)
                sim_end_M = list(sim_end.minute)
                sim_end_S = list(sim_end.second)
            else:
                sim_start_y = sim_start.year
                sim_start_m = sim_start.month
                sim_start_d = sim_start.day
                sim_start_H = sim_start.hour
                sim_start_M = sim_start.minute
                sim_start_S = sim_start.second
                sim_end_y = sim_end.year
                sim_end_m = sim_end.month
                sim_end_d = sim_end.day
                sim_end_H = sim_end.hour
                sim_end_M = sim_end.minute
                sim_end_S = sim_end.second

            namelist_dict['start_year'] = sim_start_y
            namelist_dict['start_month'] = sim_start_m
            namelist_dict['start_day'] = sim_start_d
            namelist_dict['start_hour'] = sim_start_H
            namelist_dict['start_minute'] = sim_start_M
            namelist_dict['start_second'] = sim_start_S
            
            namelist_dict['end_year'] = sim_end_y
            namelist_dict['end_month'] = sim_end_m
            namelist_dict['end_day'] = sim_end_d
            namelist_dict['end_hour'] = sim_end_H
            namelist_dict['end_minute'] = sim_end_M
            namelist_dict['end_second'] = sim_end_S


        if 'dxy' in setup_keys:
            dx,dy = self.setup_dict['dxy'],self.setup_dict['dxy']
            if ((type(dx) is not list) and (max_dom > 1)) and ('parent_grid_ratio' in setup_keys):
                ratio_total = np.ones(max_dom)
                for rr in range(0,max_dom):
                    ratio_total[rr] = np.product(self.setup_dict['parent_grid_ratio'][:rr+1])
                dx = list(dx/ratio_total)
                dy = list(dy/ratio_total)
            else:
                raise ValueError('Please specify parent_grid_ratio for nested run, or limit to 1 domain')
            namelist_dict['dx'],namelist_dict['dy'] = dx,dy
        if 'nx' in setup_keys: namelist_dict['e_we'] = self.setup_dict['nx']
        if 'ny' in setup_keys: namelist_dict['e_sn'] = self.setup_dict['ny']
        if 'icbc_type' in setup_keys:
            namelist_dict['prefix'],namelist_dict['fg_name'] = self.setup_dict['icbc_type'],self.setup_dict['icbc_type']

        namelist_dict = self._aux_inout_check(namelist_dict,
                                              inout='in',
                                              aux_dict=namelist_control_dict['time_control']['optional']['auxinput'][0])
        namelist_dict = self._aux_inout_check(namelist_dict,
                                              inout='out',
                                              aux_dict=namelist_control_dict['time_control']['optional']['auxhist'][0])
        
        if ('e_vert' not in setup_keys) and (self.setup_dict['eta_levels'] is not None):
            eta_levels = self.setup_dict['eta_levels']
            if (type(eta_levels) is list):
                namelist_dict['e_vert'] = len(eta_levels)
            elif (type(eta_levels) is str):
                namelist_dict['e_vert'] = len(eta_levels.split(','))
            else:
                raise ValueError('Must specify either eta_levels or e_vert')

        if 'radt' not in setup_keys:
            if (max_dom > 1): 
                radt = np.round(np.asarray(namelist_dict['dx'])/1000)
                radt = [max(int(rad),1) for rad in radt]
                namelist_dict['radt'] = radt
            else:
                namelist_dict['radt'] = max(int(namelist_dict['dx']/1000),1)


    ##### FOR THE REST... IF THE VALUE IN NAMELIST_CONTROL_DICT IS NOT NONE, FILL WITH DEFAULT! #####

        for section in list(namelist_control_dict.keys()):
        #for section in namelist_sections:
            required_fields = list(namelist_control_dict[section]['required'].keys())
            assigned_fields = list(namelist_dict.keys())

            vars_to_add = []
            
            
            for req_field in required_fields:
                if req_field not in assigned_fields:
                    fill_val = namelist_control_dict[section]['required'][req_field]
                    if '_interval' in req_field:
                        for asg_field in assigned_fields:
                            if req_field in asg_field:
                                vars_to_add.append(req_field)
                                fill_val = None
                    if fill_val is not None:
                        namelist_dict[req_field] = fill_val
                    #else:
                        

            namelist_keys = list(namelist_dict.keys()) + vars_to_add

            missing_keys[section] = self._CheckMissing(namelist_keys,required_fields,namelist_dict)

        #for section in list(namelist_control_dict.keys()):
        for section in namelist_sections:
            if (missing_keys[section] != []):
                f_str = '{} section is missing: '.format(section)
                for val in missing_keys[section]:
                    f_str += '{}, '.format(val)
                print(f_str)
                missing_key_flag = True
        if missing_key_flag: raise ValueError('Certain required namelist entries are missing from the setup_dict.')
            
        return(namelist_dict)

    def _TimeIntervalCheck(self,key,full_key_name):
        interval = ''
        if ('interval' in key) or ('begin' in key) or ('end' in key):
            if key[-2] == '_':
                if key[-1] in ['d','h','m','s']:
                    interval = key[-1]
                    key = key[:-2]
                else:
                    raise ValueError('{} is not supported'.format(full_key_name))
        return(key,interval)
    
    def _aux_inout_check(self,namelist_dict,inout=None,aux_dict=None):
        if inout is None:
            raise ValueError('specify "in" or "out"')
        elif inout == 'in':
            aux_key = 'auxinput'
        elif inout == 'out':
            aux_key = 'auxhist'
            
        setup_keys = list(self.setup_dict.keys())
        max_dom = namelist_dict['max_dom']
        got_aux = {}
        
        aux_names = []
        aux_keys_all = []
        for key in setup_keys:
            if aux_key in key:
                key = key.split('_')
                for kk in key:
                    if 'aux' in kk: key_name = kk
                aux_names.append(key_name)
                aux_keys_all.append('_'.join(key))
        aux_names = np.unique(aux_names)

        if aux_names != []:
            for aux_name in aux_names:
                aux_keys = []
                for key in aux_keys_all:
                    if aux_name in key: 
                        aux_key = key.replace(aux_name,'')
                        aux_key,interval = self._TimeIntervalCheck(aux_key,key)
                        aux_keys.append(aux_key)
                        if aux_key in list(aux_dict.keys()): 
                            key_val = self.setup_dict[key]
                            fill_val = aux_dict[aux_key]
                            
                            if type(fill_val) is list:
                                if type(key_val) is not list:
                                    if (fill_val[0] is not None) and (type(key_val) is not type(fill_val[0])):
                                        raise ValueError('Type disagreement for {}'.format(key))
                                    key_val = [key_val]
                                    if len(key_val) < max_dom:
                                        key_val *= max_dom
                                    elif len(key_val) > max_dom:
                                        key_val = key_val[:max_dom]
                                        
                            namelist_dict[key] = key_val
                            
                            if type(fill_val) is not type(namelist_dict[key]):
                                switch_type = True
                                try: 
                                    namelist_dict[key] = np.asarray(namelist_dict[key]).astype(type(fill_val))[0]
                                except:
                                    switch_type = False

                                if switch_type:
                                    if type(fill_val) is not type(namelist_dict[key]):
                                        raise ValueError('Type disagreement for {}'.format(key))
                        else:
                            print('{} not recognized... skipping.'.format(key))
                            
                necessary_aux_in = sorted(list(aux_dict.keys()))
                have_aux_in = sorted(aux_keys)

                for key in necessary_aux_in:
                    if key not in have_aux_in:
                        fill_val = aux_dict[key]
                        if fill_val == [None]: raise ValueError('{} for {} must be specified'.format(key,aux_name))
                            
                        if key[0] == '_':
                            key = aux_name + key
                        elif key[-1] == '_':
                            key += aux_name
                        print('Filling {} to {}'.format(key,fill_val))
                        namelist_dict[key] = fill_val

        return(namelist_dict)

    
    def _get_maxdom_str(self,num_doms,namelist_opt):
        namelist_str = ''
        for dd in range(0,num_doms):
            if type(namelist_opt) is list:
                namelist_str += '{0:>5},'.format(str(namelist_opt[dd]))
            else:
                namelist_str += '{0:>5},'.format(str(namelist_opt))
        return(phys_str)

    def _MaxDomainAdjustment(self,namelist_dict,namelist_control_dict):
        vars_that_do_not_go_in_namelist = ['nx','ny','dxy','icbc_type']
        namelist_var_dict = self._InvertControlDict(namelist_control_dict)
        all_namelist_vars = list(namelist_var_dict.keys())

        for key in list(namelist_dict.keys()):
            interval = ''
            if '_interval' in key:
                key_fill,interval = self._TimeIntervalCheck(key,key)
            else:
                key_fill = key
            if key_fill in all_namelist_vars:
                if namelist_var_dict[key_fill]['max_dom']:
                    if type(namelist_dict[key]) is not list:
                        var_type = type(namelist_dict[key])
                        if var_type is str:
                            var_list = ['{}'.format(namelist_dict[key])]*namelist_dict['max_dom']
                        else:
                            var_list = [namelist_dict[key]]*namelist_dict['max_dom']
                        namelist_dict[key] = var_list
                    else: # Vars that are 'list'
                        if len(namelist_dict[key]) < namelist_dict['max_dom']:
                            namelist_dict[key] = namelist_dict[key]*namelist_dict['max_dom']
                        elif len(namelist_dict[key]) > namelist_dict['max_dom']:
                            namelist_dict[key] = namelist_dict[key][:namelist_dict['max_dom']]

            else:
                if ('auxinput' not in key) and ('auxhist' not in key):    
                    if key in vars_that_do_not_go_in_namelist:
                        #print('{} is not needed in the namelist... removing'.format(key))
                        namelist_dict.pop(key)
                    else:
                        msg1 = 'Cannot find {} in namelist_control_dict. '.format(key)
                        msg2 = 'Please add so it can be added in the correct section of the namelist'
                        print(msg1+msg2)
        
        # Nest Checks:
        nest_error_str = '{} improperly set. Currently: {}'
        if namelist_dict['max_dom'] > 1:
            vars_to_check = ['parent_grid_ratio','i_parent_start','j_parent_start']
            for nest_var in vars_to_check:
                if max(namelist_dict[nest_var]) == 1: raise ValueError(nest_error_str.format(nest_var,namelist_dict[nest_var]))

        return(namelist_dict)
           

    def _FormatVariableForNamelist(self,value,key):
        if type(value) is str:
            if key == 'eta_levels':
                value_str = value
            else:
                value_str = "'{}'".format(value)
        elif type(value) is bool:
            if value is True:
                value_str = '.true.'
            else:
                value_str = '.false.'
        else:
            value_str = str(value)
        return(value_str)

    def write_namelist(self,namelist_type,namelist_name=None):
        namelist_dict = self.namelist_dict
        if namelist_type == 'input':
            namelist_sections = ['time_control', 'domains', 'physics', 'fdda', 'dynamics', 'bdy_control', 'namelist_quilt']
            opt_fmt = ' {0: <24} ='
        elif namelist_type == 'wps':
            opt_fmt = ' {0: <17} ='
            namelist_sections = ['share', 'geogrid', 'ungrib', 'metgrid']
            
        if namelist_name is None:
            f_name = '{}namelist.{}'.format(self.run_dir,namelist_type)
        else:
            f_name = '{}{}'.format(self.run_dir,namelist_name)
        f = open(f_name,'w')
        section_fmt = '&{}\n'
        for section in namelist_sections:
            f.write(section_fmt.format(section))
            section_keys = []
            for opt in list(self.namelist_control_dict[section].keys()):
                section_keys += (list(self.namelist_control_dict[section][opt].keys()))
            ordered_section_keys = []
            [ordered_section_keys.append(key) for key in section_keys if key not in ordered_section_keys]
            section_keys = ordered_section_keys

            ordered_section = []
            # These are variables that can have different structures...
            # ... e.g., history_interval_h or history_interval_m or history_interval, etc.
            weird_keys = ['history_interval','restart_interval','auxhist','auxinput','gfdda_interval']
            for fill_key in section_keys:
                if (fill_key in weird_keys):
                    order_option = 2
                else:
                    order_option = 1
                for key in list(namelist_dict.keys()):
                    if order_option == 2:
                        if fill_key in key:
                            ordered_section.append(key)
                    else:
                        if fill_key == key:
                            ordered_section.append(key)

            if 'eta_levels' in ordered_section:
                evert_ind = int(np.where(np.asarray(ordered_section) == 'e_vert')[0]) + 1
                eta_ind = int(np.where(np.asarray(ordered_section) == 'eta_levels')[0][0])
                eta_levels = ordered_section[eta_ind]
                ordered_section.remove(ordered_section[eta_ind])
                ordered_section.insert(evert_ind, eta_levels)

            for key in ordered_section:
                if ('_interval' in key) and (key[-2] == '_'):
                    fill_key = key[:-2]
                elif ('auxhist' in key):
                    fill_key = 'auxhist'
                elif ('auxinput' in key):
                    fill_key = 'auxinput'
                else:
                    fill_key = key

                if fill_key in section_keys:
                    key_val = namelist_dict[key]
                    if type(key_val) is list:
                        if (namelist_type == 'wps') and ((key == 'dx') or (key == 'dy')):
                            key_val = [key_val[0]]
                        if 'eta_levels' in key:
                            value_str = self._FormatEtaLevels(key_val)
                        else:
                            value_str = ''
                            for kk in key_val:
                                value_str += '{0: <6}'.format(self._FormatVariableForNamelist(kk,fill_key)+',')
                    else:
                        if 'eta_levels' in key:
                            value_str = self._FormatEtaLevels(key_val)
                        else:
                            value_str = '{0: <2}'.format(self._FormatVariableForNamelist(key_val,fill_key)+',')
                    f.write('{} {}\n'.format(opt_fmt.format(key), value_str ))
            f.write('/\n')
        f.close()

    def _FormatEtaLevels(self,eta_levels,ncols=4):
        eta_level_str = ''
        if type(eta_levels) is str:
            eta_levels = eta_levels.split(',')

        count = 1

        for eta in eta_levels:
            eta = float(eta)
            if count < ncols:
                eta_level_str += '{0:8.7f}, '.format(eta)
                count += 1
            else:
                eta_level_str += '{0:8.7f},\n  '.format(eta)
                count = 1

        return(eta_level_str[:-2])
        
        
        
def write_tslist_file(fname,
                      lat=None,
                      lon=None,
                      i=None,
                      j=None,
                      twr_names=None,
                      twr_abbr=None,
                      preV4p3=False):
    """
    Write a list of lat/lon or i/j locations to a tslist file that is
    readable by WRF.

    Usage
    ====
    fname : string 
        The path to and filename of the file to be created
    lat,lon,i,j : list or 1-D array
        Locations of the towers. 
        If using lat/lon - locx = lon, locy = lat
        If using i/j     - locx = i,   locy = j
    twr_names : list of strings, optional
        List of names for each tower location. Names should not be
        longer than 25 characters, each. If None, default names will
        be given.
    twr_abbr : list of strings, optional
        List of abbreviations for each tower location. Names should not be
        longer than 5 characters, each. If None, default abbreviations
        will be given.
    """
    if (lat is not None) and (lon is not None) and (i is None) and (j is None):
        header_keys = '# 24 characters for name | pfx |  LAT  |   LON  |'
        twr_locx = lon
        twr_locy = lat
        ij_or_ll = 'll'
    elif (i is not None) and (j is not None) and (lat is None) and (lon is None):
        header_keys = '# 24 characters for name | pfx |   I   |    J   |'
        twr_locx = i
        twr_locy = j
        ij_or_ll = 'ij'
    else:
        print('Please specify either lat&lon or i&j')
        return
    
    header_line = '#-----------------------------------------------#'
    header = '{}\n{}\n{}\n'.format(header_line,header_keys,header_line)
    
    if len(twr_locy) == len(twr_locx):
        ntowers = len(twr_locy)  
    else:
        print('Error - tower_x: {}, tower_y: {}'.format(len(twr_locx),len(twr_locy)))
        return
    
    if not isinstance(twr_names,list):
        twr_names = list(twr_names)    
    if twr_names != None:
        if len(twr_names) != ntowers:
            print('Error - Tower names: {}, tower_x: {}, tower_y: {}'.format(len(twr_names),len(twr_locx),len(twr_locy)))
            return
    else:
        twr_names = []
        for twr in np.arange(0,ntowers):
            twr_names.append('Tower{0:04d}'.format(twr+1))
            
    if not isinstance(twr_abbr,list):
        twr_abbr = list(twr_abbr)                
    if twr_abbr != None:
        if len(twr_abbr) != ntowers:
            print('Error - Tower abbr: {}, tower_x: {}, tower_y: {}'.format(len(twr_abbr),len(twr_locx),len(twr_locy)))
            return
        if len(max(twr_abbr,key=len)) > 5:
            print('Tower abbreviations are too large... setting to default names')
            twr_abbr = None
    if twr_abbr==None:
        twr_abbr = []
        for twr in np.arange(0,ntowers):
            twr_abbr.append('T{0:04d}'.format(twr+1))
            
    f = open(fname,'w')
    f.write(header)
            
    for tt in range(0,ntowers):
        if ij_or_ll == 'ij':
            twr_line = '{0:<26.25}{1: <6}{2: <8d} {3: <8d}\n'.format(
                twr_names[tt], twr_abbr[tt], int(twr_locx[tt]), int(twr_locy[tt]))
        else:
            if preV4p3:
                twr_line = '{0:<26.25}{1: <6}{2:.7s}  {3:<.8s}\n'.format(
                    twr_names[tt], twr_abbr[tt], '{0:8.7f}'.format(float(twr_locy[tt])), 
                                                 '{0:8.7f}'.format(float(twr_locx[tt])))
            else:
                twr_line = '{0:<26.25}{1: <6}{2:.9s} {3:<.10s}\n'.format(
                    twr_names[tt], twr_abbr[tt], '{0:9.7f}'.format(float(twr_locy[tt])), 
                                                 '{0:10.7f}'.format(float(twr_locx[tt])))
        f.write(twr_line)
    f.close()
    


sst_dict = {
    'OSTIA' : {
        'time_dim' : 'time',
         'lat_dim' : 'lat',
         'lon_dim' : 'lon',
        'sst_name' : 'analysed_sst',
          'sst_dx' : 5.5, # km
    },
    
    'NAVO' : {
        'time_dim' : 'time',
         'lat_dim' : 'lat',
         'lon_dim' : 'lon',
        'sst_name' : 'analysed_sst',
          'sst_dx' : 10.0, # km
    },
    
    'OSPO' : {
        'time_dim' : 'time',
         'lat_dim' : 'lat',
         'lon_dim' : 'lon',
        'sst_name' : 'analysed_sst',
          'sst_dx' : 5.5, # km
    },
    
    'NCEI' : {
        'time_dim' : 'time',
         'lat_dim' : 'lat',
         'lon_dim' : 'lon',
        'sst_name' : 'analysed_sst',
          'sst_dx' : 27.75, # km
    },
    
    'CMC' : {
        'time_dim' : 'time',
         'lat_dim' : 'lat',
         'lon_dim' : 'lon',
        'sst_name' : 'analysed_sst',
          'sst_dx' : 11.1, # km
    },
    
    'G1SST' : {
        'time_dim' : 'time',
         'lat_dim' : 'lat',
         'lon_dim' : 'lon',
        'sst_name' : 'analysed_sst',
          'sst_dx' : 1.0, # km
    },
    
    'MUR' : {
        'time_dim' : 'time',
         'lat_dim' : 'lat',
         'lon_dim' : 'lon',
        'sst_name' : 'analysed_sst',
          'sst_dx' : 1.1, # km
    },
    
    'MODIS' : {
        'time_dim' : 'time',
         'lat_dim' : 'latitude',
         'lon_dim' : 'longitude',
        'sst_name' : 'sst_data',
          'sst_dx' : 4.625, # km
    },
    
    'GOES16' : {
        'time_dim' : 'time',
         'lat_dim' : 'lats',
         'lon_dim' : 'lons',
        'sst_name' : 'sea_surface_temperature',
          'sst_dx' : 2.0, # km
    },
}

icbc_dict = {
    'ERAI' : {
        'sst_name' : 'SST',
    },
    
    'FNL' : {
        'sst_name' : 'SKINTEMP',
    },
    
    'ERA5' : {
        'sst_name' : 'SST',
    },
    
    'MERRA2' : {
        'sst_name' : 'SKINTEMP',
    },
}


class OverwriteSST():
    '''
    Given WRF met_em files and auxiliary SST data, overwrite the SST data with 
    the new SST data.
    
    Inputs:
    met_type       = string; Initial / boundary conditions used (currently only ERAI,
                             ERA5, and FNL are supported)
    overwrite_type = string; Name of SST data (OSTIA, MUT, MODIS) or FILL (replace
                             only missing values with SKINTEMP), or TSKIN (replace
                             all SST values with SKINTEMP)
    met_directory  = string; location of met_em files
    sst_directory  = string; location of sst files
    out_directory  = string; location to save new met_em files
    smooth_opt     = boolean; use smoothing over new SST data 
    fill_missing   = boolean; fill missing values in SST data with SKINTEMP
    
    '''
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    def __init__(self,
                 met_type,
                 overwrite_type,
                 met_directory,
                 sst_directory,
                 out_directory,
                 smooth_opt=False,
                 smooth_domains=None,
                 fill_missing=False,
                 skip_finished=True):
        
        self.met_type       = met_type
        self.overwrite      = overwrite_type
        self.met_dir        = met_directory
        self.sst_dir        = sst_directory
        self.out_dir        = out_directory
        self.smooth_opt     = smooth_opt
        self.smooth_domains = smooth_domains
        self.fill_opt       = fill_missing
        self.skip_finished  = skip_finished
        
        if overwrite_type == 'FILL': fill_missing=True
        
        if smooth_opt:
            self.smooth_str = 'smooth'
        else:
            self.smooth_str = 'raw'
        
        if fill_missing:
            self.smooth_str += '-filled'
            
        self.out_dir += '{}/'.format(self.smooth_str)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            
        # Get met_em_files 
        self.met_em_files = sorted(glob.glob('{}met_em.d0*'.format(self.met_dir)))        
        
        # Get SST data info (if not doing fill or tskin)
        if (overwrite_type.upper() != 'FILL') and (overwrite_type.upper() != 'TSKIN'):
            self._get_sst_info()

        # Overwrite the met_em SST data:
        for mm,met_file in enumerate(self.met_em_files[:]):
            self._check_file_exists(met_file)
            if self.exists and self.skip_finished:
                    print('{} exists... skipping'.format(self.new_file))
            else:
                print(self.new_file)
                self._get_new_sst(met_file)
                # If filling missing values with SKINTEMP:
                if fill_missing:
                    self._fill_missing(met_file)
                self.new_sst = np.nan_to_num(self.new_sst)
                # Write to new file:
                self._write_new_file(met_file)

    def _check_file_exists(self,met_file):
        f_name = met_file.split('/')[-1]
        self.new_file = self.out_dir + f_name
        self.exists = os.path.exists(self.new_file)
            
            
    def _get_sst_info(self):
        
        if self.overwrite == 'MODIS':
            get_time_from_fname = True
        else:
            get_time_from_fname = False
            
        sst_file_times = get_nc_file_times(f_dir='{}'.format(self.sst_dir),
                                           f_grep_str='*.nc',
                                           decode_times=True,
                                           time_dim=sst_dict[self.overwrite]['time_dim'],
                                           get_time_from_fname=get_time_from_fname,
                                           f_split=['.'],
                                           time_pos=[1])

        
        '''
        self.sst_files = sorted(glob.glob('{}*.nc'.format(self.sst_dir)))
        num_sst_files = len(self.sst_files)
        sst_file_times = {}
        for ff,fname in enumerate(self.sst_files): 
            sst = xr.open_dataset(fname)
            
            if ff == 0:
                self.sst_lat = sst[sst_dict[self.overwrite]['lat_dim']]
                self.sst_lon = sst[sst_dict[self.overwrite]['lon_dim']]
            
            if self.overwrite == 'MODIS':
                f_time = [datetime.strptime(fname.split('/')[-1].split('.')[1],'%Y%m%d')]
            else:
                f_time = sst[sst_dict[self.overwrite]['time_dim']].data
                
            for ft in f_time:
                ft = pd.to_datetime(ft)
                sst_file_times[ft] = fname
        '''
        self.sst_file_times = sst_file_times
        sst = xr.open_dataset(sst_file_times[list(sst_file_times.keys())[0]])
        sst_lat = sst[sst_dict[self.overwrite]['lat_dim']]
        sst_lon = sst[sst_dict[self.overwrite]['lon_dim']]
        
        self.sst_lat = sst_lat
        self.sst_lon = sst_lon
        
        


    
    def _get_new_sst(self,met_file):
        import matplotlib.pyplot as plt

        met = xr.open_dataset(met_file)
        met_domain = int(met_file.split('.')[1].replace('d',''))
        met_time = pd.to_datetime(met.Times.data[0].decode().replace('_',' '))
        met_lat = np.squeeze(met.XLAT_M)
        met_lon = np.squeeze(met.XLONG_M)
        met_landmask = np.squeeze(met.LANDMASK)
        met_sst = np.squeeze(met[icbc_dict[self.met_type]['sst_name']])
        
        met_sst.assign_coords()

        new_sst = met_sst.data.copy()
        
        sst_lat = self.sst_lat.data
        sst_lon = self.sst_lon.data
            
        if (self.overwrite.upper() != 'FILL') and (self.overwrite.upper() != 'TSKIN'):

            # Find closest SST files:
            sst_neighbors = self._get_closest_files(met_time)
            sst_weights   = self._get_time_weights(met_time,sst_neighbors)

            before_ds = xr.open_dataset(self.sst_file_times[sst_neighbors[0]])
            after_ds  = xr.open_dataset(self.sst_file_times[sst_neighbors[1]])

            if (self.overwrite == 'MODIS') or (self.overwrite == 'GOES16'):
                min_lat = np.min([np.nanmax(met_lat)+1,90])
                max_lat = np.max([np.nanmin(met_lat)-1,-90])
            else:
                before_ds = before_ds.sortby('lat')
                after_ds = after_ds.sortby('lat')
                min_lat = np.max([np.nanmin(met_lat)-1,-90])
                max_lat = np.min([np.nanmax(met_lat)+1,90])

            min_lon = np.max([np.nanmin(met_lon)-1,-180])
            max_lon = np.min([np.nanmax(met_lon)+1,180])
            
            
            # Select the time from the dataset:
            if self.overwrite == 'MODIS':
                # MODIS doesn't have time, so just squeeze:
                before_sst = np.squeeze(before_ds[sst_dict[self.overwrite]['sst_name']]) + 273.15
                after_sst  = np.squeeze(after_ds[sst_dict[self.overwrite]['sst_name']]) + 273.15
            else:
                # Grab time and SST data:
                before_sst = before_ds.sel({sst_dict[self.overwrite]['time_dim'] : sst_neighbors[0]})[sst_dict[self.overwrite]['sst_name']]
                after_sst  = after_ds.sel({sst_dict[self.overwrite]['time_dim'] : sst_neighbors[1]})[sst_dict[self.overwrite]['sst_name']]
                if self.overwrite == 'GOES16':
                    before_sst += 273.15
                    after_sst  += 273.15

            
            # Get window length for smoothing
            if self.smooth_opt and (met_domain in self.smooth_domains):
                #met_dx = met.DX/1000.0
                #met_dy = met.DY/1000.0     
                #met_delta = min([met_dx,met_dy])
                #sst_delta = sst_dict[self.overwrite]['sst_dx']
                #print(met_delta,sst_delta)
                #if met_delta > sst_delta:
                #    window = int((met_delta/sst_delta)/2.0)
                #elif met_delta < sst_delta:
                #    window = int((sst_delta/met_delta)/2.0)
                
                met_dlat = np.mean(met_lat.data[1:,:] - met_lat.data[:-1,:])
                met_dlon = np.mean(met_lon.data[:,1:] - met_lon.data[:,:-1])

                sst_dlat = np.round(np.mean(sst_lat[1:] - sst_lat[:-1]),decimals=5)
                sst_dlon = np.round(np.mean(sst_lon[1:] - sst_lon[:-1]),decimals=5)

                
                if min_lat > max_lat:
                    met_dlat *= -1
                interp_lat = np.arange(min_lat,max_lat,met_dlat)
                interp_lon = np.arange(min_lon,max_lon,met_dlon)

                before_sst = before_sst.interp({sst_dict[self.overwrite]['lat_dim']:interp_lat,
                                                sst_dict[self.overwrite]['lon_dim']:interp_lon})
                after_sst  = after_sst.interp({sst_dict[self.overwrite]['lat_dim']:interp_lat,
                                               sst_dict[self.overwrite]['lon_dim']:interp_lon})
                sst_lat = before_sst[sst_dict[self.overwrite]['lat_dim']]
                sst_lon = before_sst[sst_dict[self.overwrite]['lon_dim']]
                
            else:
                print('not smoothing')
            
            before_sst = before_sst.sel({sst_dict[self.overwrite]['lat_dim']:slice(min_lat,max_lat),
                                      sst_dict[self.overwrite]['lon_dim']:slice(min_lon,max_lon)})
            after_sst  = after_sst.sel({sst_dict[self.overwrite]['lat_dim']:slice(min_lat,max_lat),
                                     sst_dict[self.overwrite]['lon_dim']:slice(min_lon,max_lon)})

            #from matplotlib.colors import Normalize
            #fig,ax = plt.subplots(nrows=2,figsize=(18,18))
            #before_sst.plot(ax=ax[0])
            #after_sst.plot(ax=ax[1])
            #plt.show()
            
            window = 0
            for jj in met.south_north:
                for ii in met.west_east:
                    if met_landmask[jj,ii] == 0.0:
                        
                        within_lat = (np.nanmin(sst_lat) <= met_lat[jj,ii] <= np.nanmax(sst_lat))
                        within_lon = (np.nanmin(sst_lon) <= met_lon[jj,ii] <= np.nanmax(sst_lon))

                        if within_lat and within_lon:
                            dist_lat = abs(sst_lat - float(met_lat[jj,ii]))
                            dist_lon = abs(sst_lon - float(met_lon[jj,ii]))
                            
                            lat_ind = np.where(dist_lat==np.min(dist_lat))[0]
                            lon_ind = np.where(dist_lon==np.min(dist_lon))[0]

                            if (len(lat_ind) > 1) and (len(lon_ind) > 1):
                                lat_s = sst_lat[lat_ind[0] - window]
                                lat_e = sst_lat[lat_ind[1] + window]
                                lon_s = sst_lon[lon_ind[0] - window]
                                lon_e = sst_lon[lon_ind[1] + window]

                            elif (len(lat_ind) > 1) and (len(lon_ind) == 1):
                                lat_s = sst_lat[lat_ind[0] - window]
                                lat_e = sst_lat[lat_ind[1] + window]
                                lon_s = sst_lon[lon_ind[0] - window]
                                lon_e = sst_lon[lon_ind[0] + window]

                            elif (len(lat_ind) == 1) and (len(lon_ind) > 1):
                                lat_s = sst_lat[lat_ind[0] - window]
                                lat_e = sst_lat[lat_ind[0] + window]
                                lon_s = sst_lon[lon_ind[0] - window]
                                lon_e = sst_lon[lon_ind[1] + window]

                            else:
                                lat_s = sst_lat[lat_ind[0] - window]
                                lat_e = sst_lat[lat_ind[0] + window]
                                lon_s = sst_lon[lon_ind[0] - window]
                                try:
                                    lon_e = sst_lon[lon_ind[0] + window]
                                except IndexError:
                                    lon_e = len(sst_lon)

                            sst_before_val = before_sst.sel({
                                                sst_dict[self.overwrite]['lat_dim']:slice(lat_s,lat_e),
                                                sst_dict[self.overwrite]['lon_dim']:slice(lon_s,lon_e)}).mean(skipna=True)

                            sst_after_val  = after_sst.sel({
                                                sst_dict[self.overwrite]['lat_dim']:slice(lat_s,lat_e),
                                                sst_dict[self.overwrite]['lon_dim']:slice(lon_s,lon_e)}).mean(skipna=True)

                            new_sst[jj,ii] = sst_before_val*sst_weights[0] + sst_after_val*sst_weights[1]

        else:
            if (self.overwrite.upper() == 'TSKIN'):
                new_sst = met_sst.data.copy()
                tsk = np.squeeze(met.SKINTEMP).data
                new_sst[np.where(met_landmask==0.0)] = tsk[np.where(met_landmask==0.0)]
            elif (self.overwrite.upper() == 'FILL'):
                new_sst = np.squeeze(met[icbc_dict[self.met_type]['sst_name']]).data

        self.new_sst = new_sst

    def _get_closest_files(self,met_time):
        sst_times = np.asarray(list(self.sst_file_times.keys()))

        if (met_time > max(sst_times)) or (met_time < min(sst_times)):
            raise Exception("met_em time out of range of SST times:\nSST min,max = {}, {}\nmet_time = {}".format(
                                min(sst_times),max(sst_times),met_time))

        time_dist = sst_times.copy()
        for dt,stime in enumerate(sst_times):
            time_dist[dt] = (stime - met_time).total_seconds()

        before_time = np.max(time_dist[np.where(time_dist <= 0)])
        after_time  = np.min(time_dist[np.where(time_dist >= 0)])
        
        sst_t_before = sst_times[np.where(time_dist==before_time)][0]
        sst_t_after = sst_times[np.where(time_dist==after_time)][0]

        return([sst_t_before,sst_t_after])
    
    
    def _get_time_weights(self,met_time,times):
        before = times[0]
        after  = times[1]
        
        d1 = (met_time - before).seconds
        d2 = (after - met_time).seconds
        if d1 == 0 & d2 == 0:
            w1,w2 = 1.0,0.0
        else:
            w1 = d2/(d1+d2)
            w2 = d1/(d1+d2)
            
        return([w1,w2])
            
        
    def _fill_missing(self,met_file):
        met = xr.open_dataset(met_file)
        tsk = np.squeeze(met.SKINTEMP)
        met_landmask = np.squeeze(met.LANDMASK)
        bad_inds = np.where(((met_landmask == 0) & (self.new_sst == 0.0)))
        self.new_sst[bad_inds] = tsk.data[bad_inds]
        bad_inds = np.where(np.isnan(self.new_sst))
        self.new_sst[bad_inds] = tsk.data[bad_inds]
        
        
    def _write_new_file(self,met_file):
        f_name = met_file.split('/')[-1]
        new = xr.open_dataset(met_file)
        new[icbc_dict[self.met_type]['sst_name']].data = np.expand_dims(self.new_sst,axis=0)
        new.attrs['source']   = '{}'.format(self.overwrite)
        new.attrs['smoothed'] = '{}'.format(self.smooth_opt)
        new.attrs['filled']   = '{}'.format(self.fill_opt)
        if os.path.exists(self.new_file):
            print('File exists... replacing')
            os.remove(self.new_file)
        new.to_netcdf(self.new_file)

        

class CreateEtaLevels_old():
    '''
    Generate a list of eta levels for WRF simulations. Core of the eta level code
    comes from Tim Juliano of NCAR. Alternatively, the user can provide a list of
    eta levels and use the smooth_eta_levels feature.
    
    Usage:
    =====
    levels : list or array
        The list of levels you want to convert to eta levels
    surface_temp : float
        average temperature at the surface
    pres_top : float
        pressure at the top of the domain
    height_top : float
        height of the top of the domain
    p0 :float
        reference pressure / pressure at the surface
    n_total_levels : int
        number of total levels. If levels does not reach the model
        top, then this needs to be specified so that the code 
        knows how many more levels to generate
    fill_to_top : boolean
        if len(levels) != n_total_levels then this tells the code
        to fill to the top or not
    smooth_eta : boolean
        If true, this will use a spline interpolation on the d(eta)
        levels and re-normalize between 1.0 and 0.0 so that there
        is a smooth transition between the specified levels and the
        filled levels.
                             
    Examples:
    1. Levels specified to the model top:

    eta_levels = generate_eta_levels(levels=np.arange(0,4000.1,20.0),
                                     pres_top=62500.0,
                                     surface_temp=290.0,
                                     height_top=4000,
                                     n_total_levels=201
                                     )
                                     
    
    2. Only specify lower levels and let the program fill the rest (no smoothing):

    eta_levels = generate_eta_levels(levels=np.linspace(0,1000,51),
                                     pres_top=10000.0,
                                     surface_temp=282.72,
                                     height_top=16229.028,
                                     n_total_levels=88
                                     )

    3. Smooth the eta levels so there are no harsh jumps in d(eta):
    
    eta_levels = generate_eta_levels(levels=np.linspace(0,1000,50),
                                     pres_top=10000.0,
                                     surface_temp=282.72,
                                     height_top=16229.028,
                                     n_total_levels=88
                                     ).smooth_eta_levels(smooth_fact=9e-4)

    '''
    import matplotlib.pyplot as plt

    gas_constant_dry_air = 287.06
    gravity = 9.80665
    M = 0.0289644
    universal_gas_constant = 8.3144598
    
    deta_limit = -0.035

    def __init__(self, levels=None,
                 eta_levels=None,
                 surface_temp=None,
                 pres_top=None,
                 height_top=None,
                 p0=100000.0,
                 fill_to_top=True,
                 n_total_levels=None,
                 transition_zone=None,
                 min_transition_deta=None,
                 cos_squeeze=None,
                 cos_tail=None,
                 overlaying_slope=None
                 ):
    
        self.p0 = p0
        self.pres_top = pres_top
        self.surface_temp = surface_temp
        
        if eta_levels is not None:
            self.eta_levels = eta_levels
        else:
            if (levels is None) or ((type(levels) is not list) and (type(levels) is not np.array) and (type(levels) is not np.ndarray)):
                print('Please specify levels in list or array')
                return
            else:        
                if not (all((isinstance(z, float) or isinstance(z, int) or isinstance(z, np.int64)) for z in levels)):
                    print('Levels must be of type float or integer')
                    return
                if type(levels) is list:
                    levels = np.asarray(levels)

            self.levels = levels

            if n_total_levels is not None:
                if n_total_levels < len(self.levels):
                    print('Setting n_total_levels to be len(levels).')
                    n_total_levels = len(levels)
            elif transition_zone is not None:
                n_total_levels = len(levels) + transition_zone
            else:
                n_total_levels = len(levels)

            pressure = self._pressure_calc()
            
            if pres_top is None:
                self.pres_top = pressure[-1]
            

            self.eta_levels = self._eta_level_calc(pressure,
                                                   height_top,
                                                   n_total_levels,
                                                   fill_to_top,
                                                   transition_zone,
                                                   min_transition_deta,
                                                   cos_squeeze,
                                                   cos_tail,
                                                   overlaying_slope)
        
            self.estimated_heights = self._estimate_heights()
            
    def _pressure_calc(self):

        return(self.p0*np.exp((-self.gravity*self.levels)/self.gas_constant_dry_air/self.surface_temp))
    
    def _eta_level_calc(self,
                        pressure,
                        height_top,
                        n_total_levels,
                        fill_to_top,
                        transition_zone,
                        min_transition_deta,
                        cos_squeeze,
                        cos_tail,
                        overlaying_slope):
        
        import matplotlib.pyplot as plt
        if height_top is None:
            height_top = (self.gas_constant_dry_air*self.surface_temp/self.gravity)*np.log((self.p0/self.pres_top))

        eta_levels = (pressure-self.pres_top)/(self.p0-self.pres_top)

        reached_model_top = False
        if float(np.max(self.levels)) < float(height_top):
            if transition_zone is not None:
                if (len(self.levels)+transition_zone) > n_total_levels:
                    print('Transition zone makes this larger than n_total_levels')
                    print('Setting n_total_levels to len(levels) + transition_zone')
                    n_total_levels = len(self.levels)+transition_zone
                
                transition = np.zeros(transition_zone)
                for tt,tran in enumerate(range(1,transition_zone+1)):
                    if overlaying_slope is None:
                        overlaying_slope = 0.002 # slope applied to cos curve
                    if cos_tail is None:
                        cos_tail = 0.3 # How much of the cos curve should continue (0.3 = 30%)
                    if cos_squeeze is None:
                        cos_squeeze = 0.9 # How much of the cos curve should be squeezed (1 = none, 0.9 = a little toward the beginning)
                        
                    transition[tt] = (1.0 + np.cos((tran*((tran/((1-cos_tail)*transition_zone))**cos_squeeze)/transition_zone)*np.pi)) - tt*overlaying_slope

                transition -= np.min(transition)
                transition /= np.max(transition)
                
                max_transition_deta = eta_levels[len(self.levels)-1] - eta_levels[len(self.levels)-2]

                orig_eta_levels = eta_levels
                orig_transition = transition.copy()

                top_lvl_threshold = 0.001
                if min_transition_deta is not None:
                    if min_transition_deta > 0:
                        min_transition_deta *= -1
                    eta_levels,error = self._calc_transition(eta_levels,
                                                             transition,
                                                             max_transition_deta,
                                                             min_transition_deta)
                    
                    if error > top_lvl_threshold:
                        iterate_for_transition = True
                        print('Specified min_transition_deta resulted in bad levels...')
                        eta_levels = orig_eta_levels.copy()
                    else:
                        iterate_for_transition = False
                else:
                    iterate_for_transition = True
                    min_transition_deta = -0.02
                    
                if iterate_for_transition:
                    print('Iterating to find reasonable levels.')
                    error = 1.0
                    prev_error = 1.0
                    count = 0
                    max_iterations = 100
                    
                    
                    check_transition = False
                    if check_transition:
                        fig,ax = plt.subplots(ncols=2,figsize=(12,5))
                    
                    while ((np.abs(error) > top_lvl_threshold) and (count <= max_iterations)) and (min_transition_deta > self.deta_limit):
                        eta_levels = orig_eta_levels.copy()
                        transition = orig_transition.copy()
                        
                        eta_levels,error = self._calc_transition(eta_levels,
                                                                 transition,
                                                                 max_transition_deta,
                                                                 min_transition_deta,
                                                                 top_lvl_threshold)
                        if abs(error) < abs(prev_error):
                            min_error = {'error': error,'deta':min_transition_deta}
                            
                            if error > 0:
                                min_transition_deta *= 1.001
                            else:
                                min_transition_deta *= 0.999
                        else:
                            if error < 0:
                                min_transition_deta *= 1.001
                            else:
                                min_transition_deta *= 0.999
                        count+=1
                        if check_transition:
                            ax[0].scatter(count,error)
                            ax[0].plot([count-1,count],[prev_error,error])
                            ax[0].set_title('error')
                            ax[1].scatter(count,min_transition_deta)
                            ax[1].set_title('min_transition_deta')
                        prev_error = error
                        if min_transition_deta < self.deta_limit:
                            print('min_transition_deta of {} exceeds reasonable limit for d(eta), {}'.format(
                                    min_transition_deta, self.deta_limit))
                            #raise ValueError ('Could not find a reasonable min_transition_deta - try adding more levels')
                            min_transition_deta = self.deta_limit
                            count = max_iterations + 1

                    
                    if (count > max_iterations) and len(eta_levels) >= n_total_levels:
                        print(eta_levels,error)
                        print(min_error)
                        raise ValueError ('Not enough levels to reach the top')
                    else:
                        if error <= top_lvl_threshold:
                            eta_levels -= np.min(eta_levels)
                            eta_levels /= np.max(eta_levels)

                if np.min(eta_levels) < 0.0:
                    eta_levels -= np.min(eta_levels)
                    eta_levels /= np.max(eta_levels)
                    reached_model_top = True
                
            if (n_total_levels is None) or (n_total_levels <= len(self.levels)):
                print('Insufficient number of levels to reach model top.')
                print('Height top: {}, top of specified levels: {}, number of levels: {}'.format(
                                height_top,self.levels[-1],n_total_levels))
                raise ValueError ('Must specify n_total_levels to complete eta_levels to model top')
                
            remaining_levels = n_total_levels - len(eta_levels)

            if remaining_levels > 0:
                if not reached_model_top:
                    print('Filling to top...')
                    eta_levels_top = np.zeros(remaining_levels+2)
                    z_scale = 0.4
                    for k in range(1,remaining_levels+2):
                        kind =  k - 1
                        eta_levels_top[kind] = (np.exp(-(k-1)/float(n_total_levels)/z_scale) - np.exp(-1./z_scale))/ (1.-np.exp(-1./z_scale))
                        #eta_levels_top[kind] = (np.exp(-(k-1)/float(remaining_levels)/z_scale) - np.exp(-1./z_scale))/ (1.-np.exp(-1./z_scale))
                        
                    eta_levels_top = eta_levels_top[:-1]
                    eta_levels_top -= np.min(eta_levels_top)
                    eta_levels_top /= np.max(eta_levels_top)

                    eta_levels_top *= eta_levels[-1]

                    eta_levels_top = list(eta_levels_top)
                    eta_levels = list(eta_levels)
                    
                    eta_levels += eta_levels_top[1:]
                    eta_levels = np.array(eta_levels)
                else:
                    print('Specified levels + transition zone reached model top.')
                    print('Setting n_total_levels to len(levels) + transition_zone')
                    n_total_levels = len(self.levels)+transition_zone


            if np.min(eta_levels) != 0.0:
                print('Insufficient number of levels to reach model top.')
                raise ValueError ('Lower the model top, increase number of levels, or increase the deta_lim')

        return(eta_levels)
    
    def _calc_transition(self,eta_levels,
                         transition,
                         max_transition_deta,
                         min_transition_deta,
                         top_lvl_threshold=None):        
        import matplotlib.pyplot as plt
        transition *= (max_transition_deta - min_transition_deta)
        transition += min_transition_deta

        eta_levels = list(eta_levels)
        for tt,tran in enumerate(transition):
            tind = len(self.levels) + tt
            eta_levels += list([eta_levels[tind-1] + tran])
        eta_levels = np.array(eta_levels)
        
        error = eta_levels[-1]
        return (eta_levels,error)
    
    def _estimate_heights(self):
        
        pressure = self.eta_levels*(self.p0-self.pres_top) + self.pres_top
        return((self.gas_constant_dry_air*self.surface_temp/self.gravity)*np.log((self.p0/pressure)))
        
    
    def smooth_eta_levels(self,
                          smooth_fact=7e-4,
                          smooth_degree=2):
        
        eta_levels = self.eta_levels
        deta_levels = eta_levels[1:] - eta_levels[:-1]
        
        import matplotlib.pyplot as plt
       
        start_smooth_ind = len(self.levels)-4
        eta_levels_to_smooth = eta_levels[start_smooth_ind:]
        
        deta_x = np.arange(0,len(eta_levels_to_smooth)-1)
        deta = eta_levels_to_smooth[1:] - eta_levels_to_smooth[:-1]
        
        spl = UnivariateSpline(deta_x,deta,k=smooth_degree)

        spl.set_smoothing_factor(smooth_fact)
        deta_x = np.arange(0,len(eta_levels_to_smooth))
        new_deta = spl(deta_x)

        final_eta_levels = np.ones(len(eta_levels))
        final_eta_levels[:start_smooth_ind] = eta_levels[:start_smooth_ind]
        for ee,eta in enumerate(new_deta):
            final_eta_levels[start_smooth_ind + ee] = eta_levels[start_smooth_ind -1] + sum(new_deta[:ee+1])
        final_eta_levels -= min(final_eta_levels)
        final_eta_levels /= max(final_eta_levels)
        
        buffer_deta = final_eta_levels[start_smooth_ind-2:start_smooth_ind+3] - final_eta_levels[start_smooth_ind-3:start_smooth_ind+2]
        
        buff_slope = (buffer_deta[-1]-buffer_deta[0]) / (len(buffer_deta)-1)
        new_buff_deta = buffer_deta[0] + buff_slope*np.arange(0,len(buffer_deta))
        
        for ww in range(0,len(buffer_deta)):
            final_eta_levels[start_smooth_ind - 2 + ww] = final_eta_levels[start_smooth_ind - 3 + ww] + new_buff_deta[ww]
        
        self.original_eta_levels = self.eta_levels
        self.eta_levels = final_eta_levels
        if self.pres_top is not None:
            self.estimated_heights = self._estimate_heights()
        return(self)
    
    
    def print_eta_levels(self,ncols=4):
        count = 0
        line = ''
        print('{} levels'.format(len(self.eta_levels)))
        for kk,eta in enumerate(self.eta_levels):
            line += '{0:8.7f}, '.format(eta)
            count+=1
            if count == ncols:
                #line += '\n'
                print(line)
                count = 0
                line = ''
        if line != '':
            print(line)

            
class CreateEtaLevels():
    
    def __init__(self, 
                 nz=None,
                 dz_bottom=None,
                 dz_top=None,
                 sfc_pressure=100000.0,
                 surface_z=0.0,
                 top_pressure=None,
                 top_z=None,
                 sfc_temperature=280.0,
                 transition_steepness=8.0,
                 transition_inflection_location=None,
                 stretch_lower_bound=0.0,
                 stretch_upper_bound=1.0,
                 tolerance=0.01,
                 verbose=False,
                 ):
        
        if nz is None: raise ValueError('Need to specify the number of levels, nz')
        if dz_bottom is None: raise ValueError('Need to specify the desired ∆z at the surface, dz_bottom')
        if dz_top is None: raise ValueError('Need to specify the desired ∆z at the top, dz_top')
        
        if transition_inflection_location is None:
            raise ValueError('Need to specify transition_start')
        if type(transition_inflection_location) is int:
            pinflect = (transition_inflection_location/nz)*2.0 - 1.0
        elif type(transition_inflection_location) is float:
            if transition_inflection_location <= 1.0:
                pinflect = (transition_inflection_location)*2.0 - 1.0
            else:
                raise ValueError('transition_inflection_location must be the index (int) or the percentage (divided by 100.0; float) of where the inflection point will be located')
        
        if (top_pressure is None):
            if (top_z is None):
                raise ValueError('Must specify either top_pressure or top_z')
            else:
                if top_z <= 0:
                    raise ValueError('top_z must be positive')
                else:
                    top_pressure = -999.
        else:
            if (top_z is None):
                if top_pressure <= 0:
                    raise ValueError('top_pressure must be positive')
                else:
                    top_z = -999.
            else:
                if (top_pressure > 0) and (top_z > 0):
                    print('Both pressure top and model top height are specified... default is height')
                    top_pressure = -999.
                elif (top_pressure <= 0) and (top_z <= 0):
                    raise ValueError('Either top_pressure or top_z must be positive')
                    
        
        self._CalculateEtaLevels(nz,dz_bottom,dz_top,sfc_pressure,
                                 surface_z,top_pressure,top_z,sfc_temperature,
                                 transition_steepness,pinflect,
                                 stretch_lower_bound,stretch_upper_bound,
                                 tolerance,verbose)
        
    def _standard_atm(self,z,pb,hb,Tb=290.0):
        R=8.31447  # J/(K*mol) - universal gas constant
        g0=9.80665 # m/s^2     - standard gravity
        M=0.0289644 # kg/mol   - molar mass of Earth's air
        #Tb=290.0      # K        - standard temperature
        p=pb*np.exp(g0*M*(hb-z)/(R*Tb))
        return p
    ####
    # function create_eta
    # computes an array of eta levels - eta
    # given an array of pressure levels - p
    # and base pressure and elevation pb and hb
    #
    def _create_eta(self,z,pb,hb,Tb=290.0):
        p=self._standard_atm(z,pb,hb,Tb)
        s=np.size(p)
        n=s
        eta=-(p-p[n-1])/(p[n-1]-p[0])
        return eta
    #
    ####
    # function stretch_coefficient
    #
    def _stretch_coefficient(self,amp,nz,steep,pinflect):
        fac = steep
        xc  = pinflect
        xa  =-1.
        xb  = 1.
        x   = np.arange(xa, xb, (xb-xa)/nz)
        cf  = amp*((np.exp(fac*(x-xc))-1.)/(np.exp(fac*(x-xc))+1.)+1.)
        return cf
    
    def _CalculateEtaLevels(self,nz,dz_bottom,dz_top,sfc_pressure,
                 surface_z,top_pressure,top_z,sfc_temperature,
                 transition_steepness,pinflect,
                 stretch_lower_bound,stretch_upper_bound,
                 tolerance,verbose):
        if top_pressure > 0.:
            diff=np.abs(top_pressure)

        if top_z > 0.:
            diff=np.abs(top_z)

        it=0

        while diff > tolerance:

            amp = stretch_lower_bound+0.5*(stretch_upper_bound-stretch_lower_bound)
            cf  = self._stretch_coefficient(amp,nz,transition_steepness,pinflect)
            dz  = cf*dz_top+dz_bottom
            z   = np.cumsum(dz)

            if top_z > 0.:
                if z[nz-1] > top_z:
                    stretch_upper_bound=amp
                if z[nz-1] < top_z:
                    stretch_lower_bound=amp
                diff=np.abs(z[nz-1]-top_z)
                if amp==0.:
                    print( 'no convergence - change parameters')
                    quit()

            p=self._standard_atm(z,sfc_pressure,surface_z,sfc_temperature)

            if top_pressure > 0.:
                if p[nz-1] > top_pressure:
                    stretch_lower_bound=amp
                if p[nz-1] < top_pressure:
                    stretch_upper_bound=amp
                diff=np.abs(p[nz-1]-top_pressure)
                if amp==0.:
                    print( 'no convergence - change parameters')
                    quit()

            it=it+1
            if verbose:
                print( it,diff,z[nz-1],p[nz-1],amp)

            

        s=np.cumsum(dz)
        t=np.arange(np.size(s))

        if verbose:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(t, s)
            ax.set(xlabel='Number of grid points', ylabel='Elevation [m]',
                   title='Z')
            ax.grid()

            #fig.savefig("elevation.png")
            plt.show()

            fig, ax = plt.subplots(nrows=2,figsize=(5,12))
            plt.subplots_adjust(hspace=0.3)
            ax[0].set(xlabel='Grid spacing [m]', ylabel='Elevation [m]',
                   title='∆Z')
            ax[0].grid()
            ax[0].plot(dz,z)
            ax[1].set(xlabel='Grid spacing [m]', ylabel='Elevation [m]',
                   title='∆Z Zoom')
            ax[1].grid()
            ax[1].plot(dz, z,marker='+')
            zoom_ind = int(pinflect*nz)
            ax[1].set_xlim(dz[0]-0.5,dz[zoom_ind]+0.5)
            ax[1].set_ylim(0,z[zoom_ind]+0.5)

            ax[0].plot([0,0,dz[zoom_ind],dz[zoom_ind],0],
                       [0,z[zoom_ind],z[zoom_ind],0,0],
                       c='k',ls='-',alpha=0.5)

            for axi in range(0,2):
                ax[axi].tick_params(labelsize=14)
                ax[axi].xaxis.label.set_fontsize(16)
                ax[axi].yaxis.label.set_fontsize(16)
                ax[axi].title.set_fontsize(18)

            #fig.savefig("dz.png")
            plt.show()

        if verbose:
            print (' level     height      dz     pressure')
            for k in range(nz):
                print( '{0:6d},{1:12.5f},{2:14.5f},{3:14.5f}'.format(k,z[k],dz[k],p[k]))

        
        self.eta_levels=self._create_eta(z,sfc_pressure,surface_z,sfc_temperature)

        
        
    def print_eta_levels(self,ncols=4):
        count = 0
        line = ''
        print('{} levels'.format(len(self.eta_levels)))
        for kk,eta in enumerate(self.eta_levels):
            line += '{0:8.7f}, '.format(eta)
            count+=1
            if count == ncols:
                #line += '\n'
                print(line)
                count = 0
                line = ''
        if line != '':
            print(line)
        
    '''
    def print_eta_levels(self,ncols=4,num_decimals=5):

        eta_levels = self.eta_levels
        count = 0
        line = ''
        print('{} levels'.format(len(eta_levels)))
        
        for kk,eta in enumerate(eta_levels):
            #eta = str(eta)[:num_decimals+2]
            eta = '{0:0.{1}f}'.format(eta, num_decimals)
            line += '{}, '.format(eta)
            count+=1
            if count == ncols:
                #line += '\n'
                print(line)
                count = 0
                line = ''
        if line != '':
            print(line)
    '''

