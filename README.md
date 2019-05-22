# mmctools
A repository for mesoscale-to-microscale coupling (MMC) preprocessing,
postprocessing, and analysis tools


## Overview

These tools are intended to:
1. Enable offline-coupled mesoscale-to-microscale simulation between a variety
   of mesoscale and microscale solvers
2. Standardize output from simulations in addition to observational data
3. Facilitate the analysis, assessment, and reporting of MMC results

The anticipated code structure is described in the sections below.

### Offline coupling methods

- One-way internal coupling with 2D mesoscale data f(t,z)
```python
# create internal source terms, f(t,z), from time-height series
from mmctools.coupling.internal import timeheight_to_sowfa
# create initial field, f(x,y,z), from vertical profiles
from mmctools.coupling.internal import ICs_to_sowfa
```

- One-way boundary coupling with 4D mesoscale data f(t,x,y,z)
```python
# create inflow planes, e.g., f(t,y,z) or f(t,x,z)
from mmctools.coupling.boundary import BCs_to_sowfa
# create initial field, f(x,y,z)
from mmctools.coupling.boundary import ICs_to_sowfa
```

### Data processing

- The `dataloaders` module provides tools to download and read datasets from the
  [A2e: Data Archive and Portal (DAP)](https://a2e.energy.gov/about/dap).
  `read_dir()` and `read_date_dirs()` are wrappers around file readers such as
  `pandas.read_csv()`, a measurement-specific reader provided herein, or a
  user-defined reader function. For example:
```python
from mmctools.dataloaders import read_dir
from mmctools.measurements.radar import profiler
# read selected files within a directory and concatenate into dataframe
df = read_dir(dpath, file_filter='*_w*', reader=profiler)
```

- The `measurements.*` modules provide reader functions for meteorological mast
  instruments and remote sensing devices.
  - `measurements.metmast`
  - `measurements.radar`
  - `measurements.lidar`
  - `measurements.sodar`

- The `datawriters` module provides tools to write out data for MMC analysis
  with consistent, *A2e-MMC* standard formats.

- Additional code-specific modules and scripts are also provided (e.g., `wrf.*`)
  where needed.

### Data analysis

- `helper_functions` includes atmospheric science formulae and utility functions
   for converting between quantities of interest

- `plotting` provides routines for visualization in the *A2e-MMC* style


## Code Development Principles

- All code should be usable in, and demonstrated by, Jupyter notebooks.
- All code should be written in Python 3.
- [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines should be
  followed as much as possible.


## Acknowledgements

Mesoscale-to-Microscale Coupling (MMC) is a project within the Atmosphere to
Electrons (A2e) Initiative, an effort within the Wind Energy Technologies
Office of the U.S. Department of Energy’s (DOE’s) Energy Efficiency and
Renewable Energy Office. The MMC project is a joint collaboration between six
DOE national laboratories and the National Center for Atmospheric Research.

