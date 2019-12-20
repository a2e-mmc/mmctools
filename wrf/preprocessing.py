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
    to use with WPS.

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

