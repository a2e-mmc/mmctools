
#import windtools.windtools.plotting as plotting
#
# <module 'windtools.windtools.plotting' from '/home/equon/windtools/windtools/plotting.py'>
# -- the wrong module was imported!

#from .windtools.windtools import plotting
#print(plotting)
#
# <module 'mmctools.windtools.windtools.plotting' from '/home/equon/a2e-mmc/mmctools/mmctools/windtools/windtools/plotting.py'>
# ImportError: cannot import name 'plot_*' from 'mmctools.plotting' (unknown location)

#from .windtools.windtools import *
#print(plotting)
#
# NameError: name 'plotting' is not defined


# for backwards compatibility, enable `from mmctools.plotting import plot_something`
# enable `from mmctools.foo.bar import baz # to import baz from windtools.foo.bar`
import os
mmctools_module = os.path.split(__file__)[0]
__path__.append(os.path.join(mmctools_module,'windtools','windtools'))

