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

    def download(self,dataid,prefix,fields,datetimes):
        """Download specified data at specified datetimes

        Usage
        =====
        dataid : str
            RDA dataset identifier, e.g. 'ds083.2' for FNL
        prefix : str
            Data prefix, e.g., 'ei.oper.an.pl'
        fields : list of str
            Field names, e.g., ['regn128sc','regn128uv']
        datetimes : timestamp or list of timestamps
            Datetime, e.g., output from
            pd.date_range(startdate,enddate,freq='21600s')
        """
        url = 'https://rda.ucar.edu/data/{dataid:s}/{prefix:s}/{YY:d}{MM:02d}/'
        url += '{prefix:s}.{field:s}.{YY:d}{MM:02d}{DD:02d}{HH:02d}'
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
                cmd[-1] = url.format(dataid=dataid,
                                     prefix=prefix,
                                     field=field,
                                     YY=datetime.year,
                                     MM=datetime.month,
                                     DD=datetime.day,
                                     HH=datetime.hour)
                p = subprocess.run(cmd)
                p.check_returncode() 


class ERAInterim(RDADataset):
    """ERA-Interim Reanalysis

    https://rda.ucar.edu/datasets/ds627.0/#!description
    """

    def download(self,datetimes):
        """Download data at specified datetimes.

        Usage
        =====
        datetimes : timestamp or list of timestamps
            Datetime, e.g., output from
            pd.date_range(startdate,enddate,freq='21600s')
        """
        # pressure-level data
        super().download(dataid='ds627.0',
                         prefix='ei.oper.an.pl',
                         fields=['regn128sc','regn128uv'],
                         datetimes=datetimes)
        # surface data
        super().download(dataid='ds627.0',
                         prefix='ei.oper.an.sfc',
                         fields=['regn128sc'],
                         datetimes=datetimes)

