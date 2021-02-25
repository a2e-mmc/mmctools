import sys,os
import subprocess
from getpass import getpass
import numpy as np
import pandas as pd
import glob

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

    Note: Production stopped on August 31, 2019

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
            print('WARNING: '+self.api_rc+' not found')
            print('Go to https://cds.climate.copernicus.eu/api-how-to for more information')
        import cdsapi
        self.client = cdsapi.Client()

    def download(self,datetimes,product,prefix=None,
                 variables=[],
                 area=[],
                 pressure_levels=None):
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

    def download(self,datetimes,path=None,bounds={}):
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
        """
        if path is None:
            path = '.'
        else:
            os.makedirs(path,exist_ok=True)
            
        N_bound = bounds.get('N', 60)
        S_bound = bounds.get('S', 0)
        W_bound = bounds.get('W', -169)
        E_bound = bounds.get('E', -47)
            
        area = [N_bound, W_bound, S_bound, E_bound]
            
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


class setup_wrf():
    '''
    Set up run directory for WRF / WPS
    '''

    def __init__(self,run_directory,icbc_directory,executables_dict,setup_dict):
        self.setup_dict    = setup_dict
        self.run_directory = run_directory
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
        elif icbc_type == 'FNL':
            met_lvls  = 27
        elif icbc_type == 'NARR':
            met_lvls  = 30
        elif icbc_type == 'MERRA2':
            met_lvls  = 73
            download_freq = '1d'

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
            namelist_defaults = {
                       'geogrid_args' : '30s',
                   'history_interval' : [60,60],
                   'interval_seconds' : self.icbc_dict['interval_seconds'],
                 'num_metgrid_levels' : self.icbc_dict['met_levels'],
            'num_metgrid_soil_levels' : self.icbc_dict['soil_levels'],
                    'input_from_file' : '.true.',
                   'restart_interval' : 360,            
                    'frames_per_file' : 1,            
                  'iofields_filename' : 'myoutfields.txt',            
                              'debug' : 0,            
                       'ts_locations' : 20,            
                          'ts_levels' : self.setup_dict['num_eta_levels'],            
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
                    'non_hydrostatic' : '.true.',
                      'moist_adv_opt' : 1,
                     'scalar_adv_opt' : 1,
                        'tke_adv_opt' : 1,
                    'h_mom_adv_order' : 5,
                    'v_mom_adv_order' : 3,
                    'h_sca_adv_order' : 5,
                    'v_sca_adv_order' : 3,
                     'spec_bdy_width' : 5,
                          'spec_zone' : 1,
                         'relax_zone' : 4,
                'nio_tasks_per_group' : 0,
                         'nio_groups' : 1,
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
        if not os.path.exists(self.run_directory):
            os.makedirs(self.run_directory)

        # Link WPS and WRF files / executables
        wrf_files = glob.glob('{}[!n]*'.format(self.wrf_exe_dir))
        self._link_files(wrf_files,self.run_directory)
        wps_files = glob.glob('{}[!n]*'.format(self.wps_exe_dir))
        self._link_files(wps_files,self.run_directory)
        
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
        f = open('{}namelist.wps'.format(self.run_directory),'w')
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
        f.write(" geog_data_path = '/glade/work/hawbecke/geog/',\n")
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
        for pp,pgr in enumerate(self.namelist_opts['parent_grid_ratio']):
            grid_ids += '{0:>5},'.format(str(pp+1))
            if pp == 0:
                pid = 1
            else:
                pid = pp
            parent_ids += '{0:>5},'.format(str(pid))
            dx_str += '{0:>5},'.format(str(int(self.namelist_opts['dxy']/pgr)))
            radt = self.namelist_opts['radt']/pgr
            if radt < 1: radt = 1
            radt_str += '{0:>5},'.format(str(int(radt)))

        grid_ids           = self._get_nl_str(num_doms,list(range(1,num_doms+1)))
        parent_grid_ratios = self._get_nl_str(num_doms,self.namelist_opts['parent_grid_ratio'])
        parent_time_ratios = self._get_nl_str(num_doms,self.namelist_opts['parent_time_ratio'])
        istart_str         = self._get_nl_str(num_doms,self.namelist_opts['istart'])
        jstart_str         = self._get_nl_str(num_doms,self.namelist_opts['jstart'])
        nx_str             = self._get_nl_str(num_doms,self.namelist_opts['nx'])
        ny_str             = self._get_nl_str(num_doms,self.namelist_opts['ny'])
            
        io_str      = self._get_nl_str(num_doms,self.namelist_opts['iofields_filename'])
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
        zdamp_str   = self._get_nl_str(num_doms,self.namelist_opts['zdamp'])
        damp_str    = self._get_nl_str(num_doms,self.namelist_opts['dampcoef'])
        khdif_str   = self._get_nl_str(num_doms,self.namelist_opts['khdif'])
        kvdif_str   = self._get_nl_str(num_doms,self.namelist_opts['kvdif'])
        nonhyd_str  = self._get_nl_str(num_doms,self.namelist_opts['non_hydrostatic'])
        moist_str   = self._get_nl_str(num_doms,self.namelist_opts['moist_adv_opt'])
        scalar_str  = self._get_nl_str(num_doms,self.namelist_opts['scalar_adv_opt'])
        tke_str     = self._get_nl_str(num_doms,self.namelist_opts['tke_adv_opt'])
        hmom_str    = self._get_nl_str(num_doms,self.namelist_opts['h_mom_adv_order'])
        vmom_str    = self._get_nl_str(num_doms,self.namelist_opts['v_mom_adv_order'])
        hsca_str    = self._get_nl_str(num_doms,self.namelist_opts['h_sca_adv_order'])
        vsca_str    = self._get_nl_str(num_doms,self.namelist_opts['v_sca_adv_order'])
        
        specified = ['.false.']*num_doms
        nested    = ['.true.']*num_doms
        specified[0] = '.true.'
        nested[0]    = '.false.'

        f = open('{}namelist.input'.format(self.run_directory),'w')
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
        f.write(" iofields_filename         = {}\n".format(io_str))
        f.write(" ignore_iofields_warning   = .true.,\n")
        f.write(" debug_level               = {} \n".format(self.namelist_opts['debug']))
        f.write("/\n")
        f.write("\n")
        f.write("&domains\n")
        f.write(" time_step                 =  {},\n".format(ts_0))
        f.write(" time_step_fract_num       =  {},\n".format(ts_num))
        f.write(" time_step_fract_den       =  {},\n".format(ts_den))
        f.write(" max_dom                   =  {},\n".format(num_doms))
        f.write(" max_ts_locs               =  {},\n".format(self.namelist_opts['ts_locations']))
        f.write(" max_ts_level              =  {},\n".format(self.namelist_opts['ts_locations']))
        f.write(" tslist_unstagger_winds    = .true., \n")
        f.write(" s_we                      =  {}\n".format("{0:>5},".format(1)*num_doms))
        f.write(" e_we                      =  {}\n".format(nx_str))
        f.write(" s_sn                      =  {}\n".format("{0:>5},".format(1)*num_doms))
        f.write(" e_sn                      =  {}\n".format(ny_str))
        f.write(" s_vert                    =  {}\n".format("{0:>5},".format(1)*num_doms))
        f.write(" e_vert                    =  {}\n".format("{0:>5},".format(self.namelist_opts['num_eta_levels'])*num_doms))
        f.write(" eta_levels  = {},\n".format(self.namelist_opts['eta_levels']))
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
        f.write(" sf_urban_physics          = {}\n".format(urb_str))
        f.write(" /\n")
        f.write("\n")
        f.write("&fdda\n")
        f.write("/\n")
        f.write("\n")
        f.write("&dynamics\n")
        f.write(" w_damping                 = {}, \n".format(self.namelist_opts['w_damping']))
        f.write(" diff_opt                  = {}\n".format(diff_str))
        f.write(" km_opt                    = {}\n".format(km_str))
        f.write(" diff_6th_opt              = {}\n".format(diff6o_str))
        f.write(" diff_6th_factor           = {}\n".format(diff6f_str))
        f.write(" base_temp                 = {}, \n".format(self.namelist_opts['base_temp']))
        f.write(" damp_opt                  = {}, \n".format(self.namelist_opts['damp_opt']))
        f.write(" zdamp                     = {}\n".format(zdamp_str))
        f.write(" dampcoef                  = {}\n".format(damp_str))
        f.write(" khdif                     = {}\n".format(khdif_str))
        f.write(" kvdif                     = {}\n".format(kvdif_str))
        f.write(" non_hydrostatic           = {}\n".format(nonhyd_str))
        f.write(" moist_adv_opt             = {}\n".format(moist_str))
        f.write(" scalar_adv_opt            = {}\n".format(scalar_str))
        f.write(" tke_adv_opt               = {}\n".format(tke_str))
        f.write(" h_mom_adv_order           = {}\n".format(hmom_str))
        f.write(" v_mom_adv_order           = {}\n".format(vmom_str))
        f.write(" h_sca_adv_order           = {}\n".format(hsca_str))
        f.write(" v_sca_adv_order           = {}\n".format(vsca_str))
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
        elif icbc_type == 'FNL':
            icbc = FNL()
        elif icbc_type == 'MERRA2':
            icbc = MERRA2()
            if 'resolution_deg' not in self.setup_keys():
                res_drag = 0
            else:
                res_drag = self.setup_dict['resolution_deg']
            optional_args['resolution_deg'] = res_drag
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

        icbc.download(datetimes,path=self.icbc_dir, **optional_args)
        
        
    def write_submission_scripts(self,submission_dict,executable,hpc='cheyenne'):

        if hpc == 'cheyenne':
            f = open('{}submit_{}.sh'.format(self.run_directory,executable),'w')
            f.write("#!/bin/bash\n")
            run_str = '{0}{1}'.format(self.icbc_dict['type'],
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
                f.write("#PBS -l select=1:ncpus=1:mpiprocs=1\n".format(submission_dict['nodes']))
            else:
                f.write("#PBS -l select={0:02d}:ncpus=36:mpiprocs=36\n".format(submission_dict['nodes']))
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
                elif icbc_type == 'FNL':
                    icbc_head = 'fnl_*'
                    icbc_vtable = 'GFS'
                elif icbc_type == 'MERRA2':
                    icbc_head = 'MERRA2_*'
                    icbc_vtable = 'GFS'
                else:
                    print('We do not support this ICBC yet...')
                
                icbc_files = '{}{}'.format(self.icbc_dir,icbc_head)
                f.write("./link_grib.csh {}\n".format(icbc_files))
                f.write("ln -sf ungrib/Variable_Tables/Vtable.{} Vtable\n".format(icbc_vtable))
                f.write("./geogrid.exe\n".format(executable))
                f.write("./ungrib.exe\n".format(executable))
                f.write("./metgrid.exe\n".format(executable))
                f.write("for i in GRIBFILE.*; do unlink $i; done\n")
            else:
                f.write("mpiexec_mpt ./{}.exe\n".format(executable))
            f.write("date_end=`date`\n")
            f.write("echo $date_end\n")
            f.close()
            
        else:
            print('The hpc requested, {}, is not currently supported... please add it!'.format(hpc))

