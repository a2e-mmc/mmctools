"""
Module to process WRF time-series output
"""
import numpy as np
import pandas as pd
import xarray as xr


def read_tslist(fpath):
    """Read the description of sampling locations"""
    return pd.read_csv(fpath,comment='#',delim_whitespace=True,
                       names=['name','prefix','lat','lon'])


