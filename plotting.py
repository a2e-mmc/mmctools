"""
Library of standardized plotting functions for basic plot formats
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch

# TODO:
# - Separate out calculation of spectra?

# Standard field labels
standard_fieldlabels = {'wspd': r'Wind speed [m/s]',
                        'wdir': r'Wind direction $[^\circ]$',
                        'w': r'Vertical wind speed [m/s]',
                        'theta': r'$\theta$ [K]',
                        'thetav': r'$\theta_v$ [K]',
                        'uu': r'$\langle u^\prime u^\prime \rangle \;[\mathrm{m^2/s^2}]$',
                        'vv': r'$\langle v^\prime v^\prime \rangle \;[\mathrm{m^2/s^2}]$',
                        'ww': r'$\langle w^\prime w^\prime \rangle \;[\mathrm{m^2/s^2}]$',
                        'uv': r'$\langle u^\prime v^\prime \rangle \;[\mathrm{m^2/s^2}]$',
                        'uw': r'$\langle u^\prime w^\prime \rangle \;[\mathrm{m^2/s^2}]$',
                        'vw': r'$\langle v^\prime w^\prime \rangle \;[\mathrm{m^2/s^2}]$',
                        'tw': r'$\langle w^\prime \theta^\prime \rangle \;[\mathrm{Km/s}]$',
                        'TI': r'TI $[-]$',
                        'TKE': r'TKE $[\mathrm{m^2/s^2}]$',
                        }

# Standard field labels for frequency spectra
standard_spectrumlabels = {'u': r'$E_{uu}\;[\mathrm{m^2/s}]$',
                           'v': r'$E_{vv}\;[\mathrm{m^2/s}]$',
                           'w': r'$E_{ww}\;[\mathrm{m^2/s}]$',
                           'wspd': r'$E_{UU}\;[\mathrm{m^2/s}]$',
                           }

# Default color cycle
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Expected names for times or heights
time_names = ['datetime','time','Time']
height_names = ['height','heights','z']

def plot_timeheight(datasets,
                    fields=None,
                    fig=None,ax=None,
                    colorschemes={},
                    fieldlimits={},
                    heightlimits=None,
                    timelimits=None,
                    fieldlabels={},
                    labelsubplots=False,
                    showcolorbars=True,
                    ncols=1,
                    subfigsize=(12,4),
                    plot_local_time=False,
                    local_time_offset=0,
                    datasetkwargs={},
                    **kwargs
                    ):
    """
    Plot time-height contours for different datasets and fields

    Usage
    =====
    datasets : pandas.DataFrame or dict 
        Dataset(s). If more than one set, datasets should
        be a dictionary with entries <dataset_name>: dataset
    fields : str, list, 'all' (or None)
        Fieldname(s) corresponding to particular column(s) of
        the datasets. fields can be None if input are MultiIndex Series.
        'all' means all fields will be plotted (in this case all
        datasets should have the same fields)
    fig : figure handle
        Custom figure handle. Should be specified together with ax
    ax : axes handle, or list or numpy ndarray with axes handles
        Customand axes handle(s).
        Size of ax should equal ndatasets*nfields
    colorschemes : str or dict
        Name of colorschemes. If only one field is plotted, colorschemes
        can be a string. Otherwise, it should be a dictionary with
        entries <fieldname>: name_of_colorschemes
        Missing colorschemess are set to 'viridis'
    fieldlimits : list or tuple, or dict
        Value range for the various fields. If only one field is 
        plotted, fieldlimits can be a list or tuple. Otherwise, it
        should be a dictionary with entries <fieldname>: fieldlimit.
        Missing fieldlimits are set automatically
    heightlimits : list or tuple
        Height axis limits
    timelimits : list or tuple
        Time axis limits
    fieldlabels : str or dict
        Custom field labels. If only one field is plotted, fieldlabels
        can be a string. Otherwise it should be a dictionary with
        entries <fieldname>: fieldlabel
    labelsubplots : bool
        Label subplots as (a), (b), (c), ...
    showcolorbars : bool
        Show colorbar per subplot
    ncols : int
        Number of columns in axes grid, must be a true divisor of total
        number of axes.
    subfigsize : list or tuple
        Standard size of subfigures
    plot_local_time : bool
        Plot dual x axes with both UTC time and local time
    local_time_offset : float
        Local time offset from UTC
    datasetkwargs : dict
        Dataset-specific options that are passed on to the actual
        plotting function. These options overwrite general options
        specified through **kwargs. The argument should be a dictionary
        with entries <dataset_name>: {**kwargs}
    **kwargs : other keyword arguments
        Options that are passed on to the actual plotting function.
        Note that these options should be the same for all datasets and
        fields and can not be used to set dataset or field specific
        limits, colorschemess, norms, etc.
        Example uses include setting shading, rasterized, etc.
    """

    args = PlottingInput(
        datasets=datasets,
        fields=fields,
        fieldlimits=fieldlimits,
        fieldlabels=fieldlabels,
        colorschemes=colorschemes,
    )
    args.set_missing_fieldlimits()

    nfields = len(args.fields)
    ndatasets = len(args.datasets)
    ntotal = nfields * ndatasets

    # Concatenate custom and standard field labels
    # (custom field labels overwrite standard fields labels if existent)
    args.fieldlabels = {**standard_fieldlabels, **args.fieldlabels}        

    fig, ax, nrows, ncols = _create_subplots_if_needed(
                                    ntotal,
                                    ncols,
                                    sharex=True,
                                    sharey=True,
                                    subfigsize=subfigsize,
                                    hspace=0.2,
                                    fig=fig,
                                    ax=ax
                                    )

    # Create flattened view of axes
    axv = np.asarray(ax).reshape(-1)

    # Initialise list of colorbars
    cbars = []

    # Loop over datasets, fields and times 
    for i, dfname in enumerate(args.datasets):
        df = args.datasets[dfname]

        heightvalues = _get_height_values(df)
        timevalues   = _get_time_values(df)
        assert(heightvalues is not None), 'timeheight plot needs a height axis'
        assert(timevalues is not None), 'timeheight plot needs a time axis'

        if isinstance(timevalues, pd.DatetimeIndex):
            # If plot local time, shift timevalues
            if plot_local_time:
                timevalues = timevalues + pd.to_timedelta(local_time_offset,'h')

            # Convert to days since 0001-01-01 00:00 UTC, plus one
            numerical_timevalues = mdates.date2num(timevalues.values)
        else:
            if isinstance(timevalues, pd.TimedeltaIndex):
                timevalues = timevalues.total_seconds()

            # Timevalues is already a numerical array
            numerical_timevalues = timevalues

        # Create time-height mesh grid
        Ts,Zs = np.meshgrid(numerical_timevalues,heightvalues,indexing='xy')

        # Create list with available fields only
        available_fields = _get_available_fieldnames(df,args.fields)

        # Pivot all fields in a dataset at once
        df_pivot = _get_pivot_table(df,available_fields)

        for j, field in enumerate(args.fields):
            # Skip loop if field not available
            if not field in available_fields:
                print('Warning: field "'+field+'" not available in dataset '+dfname)
                continue

            # Store plotting options in dictionary
            plotting_properties = {
                'vmin': args.fieldlimits[field][0],
                'vmax': args.fieldlimits[field][1],
                'cmap': args.cmap[field]
                }

            # Index of axis corresponding to dataset i and field j
            axi = i*nfields + j

            # Extract data from dataframe
            fieldvalues = _get_pivoted_field(df_pivot,field).values

            # Gather label, color, general options and dataset-specific options
            # (highest priority to dataset-specific options, then general options)
            try:
                plotting_properties = {**plotting_properties,**kwargs,**datasetkwargs[dfname]}
            except KeyError:
                plotting_properties = {**plotting_properties,**kwargs}

            # Plot data
            im = axv[axi].pcolormesh(Ts,Zs,fieldvalues.T,**plotting_properties)

            # Colorbar mark up
            if showcolorbars:
                cbar = fig.colorbar(im,ax=axv[axi],shrink=1.0)
                # Set field label if known
                try:
                    cbar.set_label(args.fieldlabels[field])
                except KeyError:
                    pass
                # Save colorbar
                cbars.append(cbar)

            # Set title if more than one dataset
            if ndatasets>1:
                axv[axi].set_title(dfname,fontsize=16)


    # Format time axis
    if isinstance(timevalues, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        ax2 = _format_time_axis(axv[(nrows-1)*ncols:],plot_local_time,local_time_offset,timelimits)
    else:
        # Set time limits if specified
        if not timelimits is None:
            axv[-1].set_xlim(timelimits)
        # Set time label
        for axi in axv[(nrows-1)*ncols:]:
            axi.set_xlabel('time [s]')

    if not heightlimits is None:
        axv[-1].set_ylim(heightlimits)

    # Add y labels
    for axi in axv: 
        axi.set_ylabel(r'Height [m]')
    
    # Number sub figures as a, b, c, ...
    if labelsubplots and axv.size > 1:
        for i,axi in enumerate(axv):
            axi.text(-0.14,1.0,'('+chr(i+97)+')',transform=axi.transAxes,size=16)

    if plot_local_time and  ax2 is not None:
        return fig, ax, ax2, cbars
    else:
        return fig, ax, cbars


def plot_timehistory_at_height(datasets,
                               fields=None,
                               heights=None,
                               fig=None,ax=None,
                               fieldlimits={},
                               timelimits=None,
                               fieldlabels={},
                               cmap=None,
                               stack_by_datasets=None,
                               labelsubplots=False,
                               ncols=1,
                               subfigsize=(12,3),
                               plot_local_time=False,
                               local_time_offset=0,
                               datasetkwargs={},
                               **kwargs
                               ):
    """
    Plot time history at specified height(s) for various dataset(s)
    and/or field(s).
    
    By default, data for multiple datasets or multiple heights are
    stacked in a single subplot. When multiple datasets and multiple
    heights are specified together, heights are stacked in a subplot
    per field and per dataset.

    Usage
    =====
    datasets : pandas.DataFrame or dict 
        Dataset(s). If more than one set, datasets should
        be a dictionary with entries <dataset_name>: dataset
    fields : str, list, 'all' (or None)
        Fieldname(s) corresponding to particular column(s) of
        the datasets. fields can be None if input are Series.
        'all' means all fields will be plotted (in this case all
        datasets should have the same fields)
    heights : float, list, 'all' (or None)
        Height(s) for which time history is plotted. heights can be
        None if all datasets combined have no more than one height
        value. 'all' means the time history for all heights in the
        datasets will be plotted (in this case all datasets should
        have the same heights)
    fig : figure handle
        Custom figure handle. Should be specified together with ax
    ax : axes handle, or list or numpy ndarray with axes handles
        Customand axes handle(s).
        Size of ax should equal nfields * (ndatasets or nheights)
    fieldlimits : list or tuple, or dict
        Value range for the various fields. If only one field is 
        plotted, fieldlimits can be a list or tuple. Otherwise, it
        should be a dictionary with entries <fieldname>: fieldlimit.
        Missing fieldlimits are set automatically
    timelimits : list or tuple
        Time axis limits
    fieldlabels : str or dict
        Custom field labels. If only one field is plotted, fieldlabels
        can be a string. Otherwise it should be a dictionary with
        entries <fieldname>: fieldlabel
    cmap : str
        Colormap used when stacking heights
    stack_by_datasets : bool (or None)
        Flag to specify what is plotted ("stacked") together per subfigure.
        If True, stack datasets together, otherwise stack by heights. If
        None, stack_by_datasets will be set based on the number of heights
        and datasets. 
    labelsubplots : bool
        Label subplots as (a), (b), (c), ...
    ncols : int
        Number of columns in axes grid, must be a true divisor of total
        number of axes.
    subfigsize : list or tuple
        Standard size of subfigures
    plot_local_time : bool
        Plot dual x axes with both UTC time and local time
    local_time_offset : float
        Local time offset from UTC
    datasetkwargs : dict
        Dataset-specific options that are passed on to the actual
        plotting function. These options overwrite general options
        specified through **kwargs. The argument should be a dictionary
        with entries <dataset_name>: {**kwargs}
    **kwargs : other keyword arguments
        Options that are passed on to the actual plotting function.
        Note that these options should be the same for all datasets,
        fields and heights, and they can not be used to set dataset,
        field or height specific colors, limits, etc.
        Example uses include setting linestyle/width, marker, etc.
    """
    # Avoid FutureWarning concerning the use of an implicitly registered
    # datetime converter for a matplotlib plotting method. The converter
    # was registered by pandas on import. Future versions of pandas will
    # require explicit registration of matplotlib converters, as done here.
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    args = PlottingInput(
        datasets=datasets,
        fields=fields,
        heights=heights,
        fieldlimits=fieldlimits,
        fieldlabels=fieldlabels,
    )

    nfields = len(args.fields)
    nheights = len(args.heights)
    ndatasets = len(args.datasets)

    # Concatenate custom and standard field labels
    # (custom field labels overwrite standard fields labels if existent)
    args.fieldlabels = {**standard_fieldlabels, **args.fieldlabels}

    # Set up subplot grid
    if stack_by_datasets is None:
        if nheights>1:
            stack_by_datasets = False
        else:
            stack_by_datasets = True

    if stack_by_datasets:
        ntotal = nfields*nheights
    else:
        ntotal = nfields*ndatasets

    fig, ax, nrows, ncols = _create_subplots_if_needed(
                                    ntotal,
                                    ncols,
                                    sharex=True,
                                    subfigsize=subfigsize,
                                    hspace=0.2,
                                    fig=fig,
                                    ax=ax
                                    )

    # Create flattened view of axes
    axv = np.asarray(ax).reshape(-1)

    # Loop over datasets and fields 
    for i,dfname in enumerate(args.datasets):
        df = args.datasets[dfname]
        heightvalues = _get_height_values(df)
        timevalues = _get_time_values(df)
        assert(timevalues is not None), 'timehistory plot needs a time axis'

        if isinstance(timevalues, pd.TimedeltaIndex):
            timevalues = timevalues.total_seconds()

        # If plot local time, shift timevalues
        if plot_local_time and isinstance(timevalues, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            timevalues = timevalues + pd.to_timedelta(local_time_offset,'h')

        # Create list with available fields only
        available_fields = _get_available_fieldnames(df,args.fields)

        # If any of the requested heights is not available,
        # pivot the dataframe to allow interpolation.
        # Pivot all fields in a dataset at once to reduce computation time
        if (not heightvalues is None) and (not all([h in heightvalues for h in args.heights])):
            df_pivot = _get_pivot_table(df,available_fields)
            print('Pivoting '+dfname)

        for j, field in enumerate(args.fields):
            # Skip loop if field not available
            if not field in available_fields:
                print('Warning: field "'+field+'" not available in dataset '+dfname)
                continue


            for k, height in enumerate(args.heights):
                # Store plotting options in dictionary
                # Set default linestyle to '-' and no markers
                plotting_properties = {
                    'linestyle':'-',
                    'marker':None,
                    }

                # Axis order, label and title depend on value of stack_by_datasets 
                if stack_by_datasets:
                    # Index of axis corresponding to field j and height k
                    axi = k*nfields + j

                    # Use datasetname as label
                    plotting_properties['label'] = dfname

                    # Set title if multiple heights are compared
                    if nheights>1:
                        axv[axi].set_title('z = {:.1f} m'.format(height),fontsize=16)

                    # Set colors
                    plotting_properties['color'] = default_colors[i]
                else:
                    # Index of axis corresponding to field j and dataset i 
                    axi = i*nfields + j

                    # Use height as label
                    plotting_properties['label'] = 'z = {:.1f} m'.format(height)

                    # Set title if multiple datasets are compared
                    if ndatasets>1:
                        axv[axi].set_title(dfname,fontsize=16)

                    # Set colors
                    if cmap is not None:
                        cmap = mpl.cm.get_cmap(cmap)
                        plotting_properties['color'] = cmap(k/(nheights-1))
                    else:
                        plotting_properties['color'] = default_colors[k]

                # Extract data from dataframe
                if heightvalues is None:
                    signal = df.values
                elif height in heightvalues:
                    if isinstance(df,pd.DataFrame):
                        if isinstance(df.index,pd.MultiIndex):
                            signal = df.loc[(slice(None),height),field].values
                        else:
                            signal = df.loc[df.height==height,field].values
                    elif isinstance(df,pd.Series):
                        if isinstance(df.index,pd.MultiIndex):
                            signal = df.loc[(slice(None),height)].values
                        else:
                            signal = df.values
                else:
                    signal = interp1d(heightvalues,_get_pivoted_field(df_pivot,field).values,axis=-1,fill_value="extrapolate")(height)
                
                # Gather label, color, general options and dataset-specific options
                # (highest priority to dataset-specific options, then general options)
                try:
                    plotting_properties = {**plotting_properties,**kwargs,**datasetkwargs[dfname]}
                except KeyError:
                    plotting_properties = {**plotting_properties,**kwargs}
                
                # Plot data
                axv[axi].plot(timevalues,signal,**plotting_properties)

                # Set field label if known
                try:
                    axv[axi].set_ylabel(args.fieldlabels[field])
                except KeyError:
                    pass
                # Set field limits if specified
                try:
                    axv[axi].set_ylim(args.fieldlimits[field])
                except KeyError:
                    pass
   
    # Set axis grid
    for axi in axv:
        axi.xaxis.grid(True,which='minor')
        axi.yaxis.grid()
    
    # Format time axis
    if isinstance(timevalues, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        ax2 = _format_time_axis(axv[(nrows-1)*ncols:],plot_local_time,local_time_offset,timelimits)
    else:
        # Set time limits if specified
        if not timelimits is None:
            axv[-1].set_xlim(timelimits)
        # Set time label
        for axi in axv[(nrows-1)*ncols:]:
            axi.set_xlabel('time [s]')

    # Number sub figures as a, b, c, ...
    if labelsubplots and axv.size > 1:
        for i,axi in enumerate(axv):
            axi.text(-0.14,1.0,'('+chr(i+97)+')',transform=axi.transAxes,size=16)

    # Add legend if more than one entry
    if (stack_by_datasets and ndatasets>1) or (not stack_by_datasets and nheights>1):
        leg = axv[ncols-1].legend(loc='upper left',bbox_to_anchor=(1.05,1.0),fontsize=16)

    if plot_local_time and ax2 is not None:
        return fig, ax, ax2
    else:
        return fig, ax


def plot_profile(datasets,
                 fields=None,
                 times=None,
                 fig=None,ax=None,
                 fieldlimits={},
                 heightlimits=None,
                 fieldlabels={},
                 cmap=None,
                 stack_by_datasets=None,
                 labelsubplots=False,
                 fieldorder='C',
                 ncols=None,
                 subfigsize=(4,5),
                 datasetkwargs={},
                 **kwargs
                ):
    """
    Plot vertical profile at specified time(s) for various dataset(s)
    and/or field(s).

    By default, data for multiple datasets or multiple times are
    stacked in a single subplot. When multiple datasets and multiple
    times are specified together, times are stacked in a subplot
    per field and per dataset.

    Usage
    =====
    datasets : pandas.DataFrame or dict 
        Dataset(s). If more than one set, datasets should
        be a dictionary with entries <dataset_name>: dataset
    fields : str, list, 'all' (or None)
        Fieldname(s) corresponding to particular column(s) of
        the datasets. fields can be None if input are Series.
        'all' means all fields will be plotted (in this case all
        datasets should have the same fields)
    times : str, int, float, list (or None)
        Time(s) for which vertical profiles are plotted, specified as
        either datetime strings or numerical values (seconds, e.g.,
        simulation time). times can be None if all datasets combined
        have no more than one time value.
    fig : figure handle
        Custom figure handle. Should be specified together with ax
    ax : axes handle, or list or numpy ndarray with axes handles
        Customand axes handle(s).
        Size of ax should equal nfields * (ndatasets or ntimes)
    fieldlimits : list or tuple, or dict
        Value range for the various fields. If only one field is 
        plotted, fieldlimits can be a list or tuple. Otherwise, it
        should be a dictionary with entries <fieldname>: fieldlimit.
        Missing fieldlimits are set automatically
    heightlimits : list or tuple
        Height axis limits
    fieldlabels : str or dict
        Custom field labels. If only one field is plotted, fieldlabels
        can be a string. Otherwise it should be a dictionary with
        entries <fieldname>: fieldlabel
    cmap : str
        Colormap used when stacking times
    stack_by_datasets : bool (or None)
        Flag to specify what is plotted ("stacked") together per subfigure.
        If True, stack datasets together, otherwise stack by times. If
        None, stack_by_datasets will be set based on the number of times
        and datasets. 
    labelsubplots : bool
        Label subplots as (a), (b), (c), ...
    fieldorder : 'C' or 'F'
        Index ordering for assigning fields and datasets/times (depending
        on stack_by_datasets) to axes grid (row by row). Fields is considered the
        first axis, so 'C' means fields change slowest, 'F' means fields
        change fastest.
    ncols : int
        Number of columns in axes grid, must be a true divisor of total
        number of axes.
    subfigsize : list or tuple
        Standard size of subfigures
    datasetkwargs : dict
        Dataset-specific options that are passed on to the actual
        plotting function. These options overwrite general options
        specified through **kwargs. The argument should be a dictionary
        with entries <dataset_name>: {**kwargs}
    **kwargs : other keyword arguments
        Options that are passed on to the actual plotting function.
        Note that these options should be the same for all datasets,
        fields and times, and they can not be used to set dataset,
        field or time specific colors, limits, etc.
        Example uses include setting linestyle/width, marker, etc.
    """

    args = PlottingInput(
        datasets=datasets,
        fields=fields,
        times=times,
        fieldlimits=fieldlimits,
        fieldlabels=fieldlabels,
        fieldorder=fieldorder,
    )

    nfields = len(args.fields)
    ntimes = len(args.times)
    ndatasets = len(args.datasets)

    # Concatenate custom and standard field labels
    # (custom field labels overwrite standard fields labels if existent)
    args.fieldlabels = {**standard_fieldlabels, **args.fieldlabels}

    # Set up subplot grid
    if stack_by_datasets is None:
        if ntimes>1:
            stack_by_datasets = False
        else:
            stack_by_datasets = True

    if stack_by_datasets:
        ntotal = nfields * ntimes
    else:
        ntotal = nfields * ndatasets

    fig, ax, nrows, ncols = _create_subplots_if_needed(
                                    ntotal,
                                    ncols,
                                    default_ncols=int(ntotal/nfields),
                                    fieldorder=args.fieldorder,
                                    avoid_single_column=True,
                                    sharey=True,
                                    subfigsize=subfigsize,
                                    wspace=0.2,
                                    fig=fig,
                                    ax=ax,
                                    )

    # Create flattened view of axes
    axv = np.asarray(ax).reshape(-1)

    # Loop over datasets, fields and times 
    for i, dfname in enumerate(args.datasets):
        df = args.datasets[dfname]
        heightvalues = _get_height_values(df)
        assert(heightvalues is not None), 'profile plot needs a height axis'
        timevalues = _get_time_values(df)

        # Create list with available fields only
        available_fields = _get_available_fieldnames(df,args.fields)

        # Pivot all fields in a dataset at once
        if not timevalues is None:
            df_pivot = _get_pivot_table(df,available_fields)

        for j, field in enumerate(args.fields):
            # Skip loop if field not available
            if not field in available_fields:
                print('Warning: field "'+field+'" not available in dataset '+dfname)
                continue

            for k, time in enumerate(args.times):
                plotting_properties = {}

                # Axis order, label and title depend on value of stack_by_datasets 
                if stack_by_datasets:
                    # Index of axis corresponding to field j and time k
                    if args.fieldorder == 'C':
                        axi = j*ntimes + k
                    else:
                        axi = k*nfields + j

                    # Use datasetname as label
                    plotting_properties['label'] = dfname

                    # Set title if multiple times are compared
                    if ntimes>1:
                        if isinstance(time, (int,float,np.number)):
                            tstr = '{:g} s'.format(time)
                        else:
                            tstr = pd.to_datetime(time).strftime('%Y-%m-%d %H%M UTC')
                        axv[axi].set_title(tstr, fontsize=16)

                    # Set color
                    plotting_properties['color'] = default_colors[i]
                else:
                    # Index of axis corresponding to field j and dataset i
                    if args.fieldorder == 'C':
                        axi = j*ndatasets + i
                    else:
                        axi = i*nfields + j
                    
                    # Use time as label
                    if isinstance(time, (int,float,np.number)):
                        plotting_properties['label'] = '{:g} s'.format(time)
                    else:
                        plotting_properties['label'] = pd.to_datetime(time).strftime('%Y-%m-%d %H%M UTC')

                    # Set title if multiple datasets are compared
                    if ndatasets>1:
                        axv[axi].set_title(dfname,fontsize=16)

                    # Set colors
                    if cmap is not None:
                        cmap = mpl.cm.get_cmap(cmap)
                        plotting_properties['color'] = cmap(k/(ntimes-1))
                    else:
                        plotting_properties['color'] = default_colors[k]
                
                # Extract data from dataframe
                if timevalues is None:
                    # Dataset will not be pivoted
                    fieldvalues = _get_field(df,field).values
                else:
                    fieldvalues = _get_pivoted_field(df_pivot,field).loc[time].values.squeeze()

                # Gather label, color, general options and dataset-specific options
                # (highest priority to dataset-specific options, then general options)
                try:
                    plotting_properties = {**plotting_properties,**kwargs,**datasetkwargs[dfname]}
                except KeyError:
                    plotting_properties = {**plotting_properties,**kwargs}

                # Plot data
                axv[axi].plot(fieldvalues,heightvalues,**plotting_properties)

                # Set field label if known
                try:
                    axv[axi].set_xlabel(args.fieldlabels[field])
                except KeyError:
                    pass
                # Set field limits if specified
                try:
                    axv[axi].set_xlim(args.fieldlimits[field])
                except KeyError:
                    pass
    
    for axi in axv:
        axi.grid(True,which='both')

    # Set height limits if specified
    if not heightlimits is None:
        axv[0].set_ylim(heightlimits)

    # Add y labels
    for r in range(nrows): 
        axv[r*ncols].set_ylabel(r'Height [m]')
    
    # Number sub figures as a, b, c, ...
    if labelsubplots and axv.size > 1:
        for i,axi in enumerate(axv):
            axi.text(-0.14,-0.18,'('+chr(i+97)+')',transform=axi.transAxes,size=16)
    
    # Add legend if more than one entry
    if (stack_by_datasets=='datasets' and ndatasets>1) or (not stack_by_datasets=='times' and ntimes>1):
        leg = axv[ncols-1].legend(loc='upper left',bbox_to_anchor=(1.05,1.0),fontsize=16)

    return fig,ax


def plot_spectrum(datasets,
                  height,
                  times,
                  fields=None,
                  fig=None,ax=None,
                  Tperiod=3600.0,
                  Tsegment=600.0,
                  fieldlimits={},
                  freqlimits=None,
                  fieldlabels={},
                  labelsubplots=False,
                  ncols=None,
                  subfigsize=(4,5),
                  datasetkwargs={},
                  **kwargs
                  ):
    """
    Plot frequency spectrum at a given height for different datasets,
    times and fields, using a subplot per time and per field

    The frequency spectrum is computed using scipy.signal.welch, which
    estimates the power spectral density by dividing the data into over-
    lapping segments, computing a modified periodogram for each segment
    and averaging the periodograms.

    Usage
    =====
    datasets : pandas.DataFrame or dict 
        Dataset(s). If more than one set, datasets should
        be a dictionary with entries <dataset_name>: dataset
    height : float
        Height for which frequency spectrum is plotted
    times : str, list
        Start time(s) of the time period(s) for which the frequency
        spectrum is computed.
    fields : str, list, 'all' (or None)
        Fieldname(s) corresponding to particular column(s) of
        the datasets. fields can be None if input are Series.
        'all' means all fields will be plotted (in this case all
        datasets should have the same fields)
    fig : figure handle
        Custom figure handle. Should be specified together with ax
    ax : axes handle, or list or numpy ndarray with axes handles
        Customand axes handle(s).
        Size of ax should equal nfields * ntimes
    Tperiod : float
        Length of the time period in seconds over which frequency
        spectrum is computed.
    Tsegment : float
        Length of time segments of the welch method in seconds
    fieldlimits : list or tuple, or dict
        Value range for the various fields. If only one field is 
        plotted, fieldlimits can be a list or tuple. Otherwise, it
        should be a dictionary with entries <fieldname>: fieldlimit.
        Missing fieldlimits are set automatically
    freqlimits : list or tuple
        Frequency axis limits
    fieldlabels : str or dict
        Custom field labels. If only one field is plotted, fieldlabels
        can be a string. Otherwise it should be a dictionary with
        entries <fieldname>: fieldlabel
    labelsubplots : bool
        Label subplots as (a), (b), (c), ...
    ncols : int
        Number of columns in axes grid, must be a true divisor of total
        number of axes.
    subfigsize : list or tuple
        Standard size of subfigures
    datasetkwargs : dict
        Dataset-specific options that are passed on to the actual
        plotting function. These options overwrite general options
        specified through **kwargs. The argument should be a dictionary
        with entries <dataset_name>: {**kwargs}
    **kwargs : other keyword arguments
        Options that are passed on to the actual plotting function.
        Note that these options should be the same for all datasets,
        fields and times, and they can not be used to set dataset,
        field or time specific colors, limits, etc.
        Example uses include setting linestyle/width, marker, etc.
    """

    args = PlottingInput(
        datasets=datasets,
        fields=fields,
        times=times,
        fieldlimits=fieldlimits,
        fieldlabels=fieldlabels,
    )

    nfields = len(args.fields)
    ntimes = len(args.times)
    ndatasets = len(args.datasets)
    ntotal = nfields * ntimes

    # Concatenate custom and standard field labels
    # (custom field labels overwrite standard fields labels if existent)
    args.fieldlabels = {**standard_spectrumlabels, **args.fieldlabels}

    fig, ax, nrows, ncols = _create_subplots_if_needed(
                                    ntotal,
                                    ncols,
                                    default_ncols=ntimes,
                                    avoid_single_column=True,
                                    sharex=True,
                                    subfigsize=subfigsize,
                                    wspace=0.3,
                                    fig=fig,
                                    ax=ax,
                                    )

    # Create flattened view of axes
    axv = np.asarray(ax).reshape(-1)

    # Loop over datasets, fields and times 
    for j, dfname in enumerate(args.datasets):
        df = args.datasets[dfname]

        heightvalues = _get_height_values(df)
        timevalues   = _get_time_values(df)
        dt = (timevalues[1]-timevalues[0]) / pd.Timedelta(1,unit='s')     #Sampling rate in seconds

        # Create list with available fields only
        available_fields = _get_available_fieldnames(df,args.fields)

        # Pivot all fields of a dataset at once
        df_pivot = _get_pivot_table(df,available_fields)

        for k, field in enumerate(args.fields):
            # Skip loop if field not available
            if not field in available_fields:
                print('Warning: field "'+field+'" not available in dataset '+dfname)
                continue

            for i, tstart in enumerate(args.times):
                plotting_properties = {'label':dfname}

                # Index of axis corresponding to field k and time i
                axi = k*ntimes + i
                
                # Axes mark up
                if j==0:
                    axv[axi].set_title(pd.to_datetime(tstart).strftime('%Y-%m-%d %H%M UTC'),fontsize=16)

                # Compute frequency spectrum
                istart = np.where(timevalues==pd.to_datetime(tstart))[0][0]
                signal = interp1d(heightvalues,_get_pivoted_field(df_pivot,field).interpolate(method='linear').values,axis=-1,fill_value="extrapolate")(height)
                f, P = welch(signal[istart:istart+np.int(Tperiod/dt)],fs=1./dt,nperseg=np.int(Tsegment/dt),
                            detrend='linear',window='hanning',scaling='density')
                
                # Gather label, general options and dataset-specific options
                # (highest priority to dataset-specific options, then general options)
                try:
                    plotting_properties = {**plotting_properties,**kwargs,**datasetkwargs[dfname]}
                except KeyError:
                    plotting_properties = {**plotting_properties,**kwargs}

                # Plot data
                axv[axi].loglog(f[1:],P[1:],**plotting_properties)
   

    # Set frequency label
    for c in range(ncols):
        axv[ncols*(nrows-1)+c].set_xlabel('f [Hz]')

    # Specify field label and field limits if specified 
    for r in range(nrows):
        try:
            axv[r*ncols].set_ylabel(args.fieldlabels[args.fields[r]])
        except KeyError:
            pass
        try:
            axv[r*ncols].set_ylim(args.fieldlimits[args.fields[r]])
        except KeyError:
            pass

    # Set frequency limits if specified
    if not freqlimits is None:
        axv[0].set_xlim(freqlimits)

    # Number sub figures as a, b, c, ...
    if labelsubplots and axv.size > 1:
        for i,axi in enumerate(axv):
            axi.text(-0.14,-0.18,'('+chr(i+97)+')',transform=axi.transAxes,size=16)

    # Add legend if more than one dataset
    if ndatasets>1:
        leg = axv[ncols-1].legend(loc='upper left',bbox_to_anchor=(1.05,1.0))

    return fig, ax


# ---------------------------------------------
#
# DEFINITION OF AUXILIARY CLASSES AND FUNCTIONS
#
# ---------------------------------------------

class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class PlottingInput(object):
    """
    Auxiliary class to collect input data and options for plotting
    functions, and to check if the inputs are consistent
    """

    def __init__(self, datasets, fields, **argd):
        # Add all arguments as class attributes
        self.__dict__.update({'datasets':datasets,
                              'fields':fields,
                              **argd})

        # Check consistency of all attributes
        self._check_consistency()

    def _check_consistency(self):
        """
        Check consistency of all input data
        """

        # ----------------------
        # Check dataset argument
        # ----------------------
        # If a single dataset is provided, convert to a dictionary
        # under a generic key 'Dataset'
        if isinstance(self.datasets,(pd.Series,pd.DataFrame)):
            self.datasets = {'Dataset': self.datasets}
        for dfname in self.datasets:
            assert(isinstance(self.datasets[dfname],(pd.Series,pd.DataFrame))), \
                "Currently only pandas Series or DataFrames are supported"
           
        # ----------------------
        # Check fields argument
        # ----------------------
        # If no fields are specified, check that
        # - all datasets are series
        # - the name of every series is either None or matches other series names
        if self.fields is None:
            assert(all([isinstance(self.datasets[dfname],pd.Series) for dfname in self.datasets])), \
                "'fields' argument must be specified unless all datasets are pandas Series"
            series_names = set()
            for dfname in self.datasets:
                if not self.datasets[dfname].name is None:
                    series_names.add(self.datasets[dfname].name)
            if len(series_names)==0:
                # All names are None, so set generic field name
                self.fields = ['field',]
            elif len(series_names)==1:
                self.fields = list(series_names)
            else:
                raise InputError('attempting to plot multiple series with different field names')
        # If fields='all', retrieve fields from dataset
        elif self.fields=='all':
            self.fields = _get_fieldnames(list(self.datasets.values())[0])
            assert(all([_get_fieldnames(df)==self.fields for df in self.datasets.values()])), \
               "The option fields = 'all' only works when all datasets have the same fields"
            # No fieldnames, so use generic field name
            if self.fields == [None,]:
                self.fields = ['field',]
        # If fields is a single instance, convert to a list
        elif isinstance(self.fields,str):
            self.fields = [self.fields,]

        # ----------------------------------
        # Check match of fields and datasets
        # ----------------------------------
        # Check if all datasets have at least one of the requested fields
        for dfname in self.datasets:
            df = self.datasets[dfname]
            if isinstance(df,pd.DataFrame):
                assert(any([field in df.columns for field in self.fields])), \
                    'DataFrame '+dfname+' does not contain any of the requested fields'
            elif isinstance(df,pd.Series):
                if df.name is None:
                    assert(len(self.fields)==1), \
                        'Series must have a name if more than one fields is specified'
                else:
                    assert(df.name in self.fields), \
                        'Series '+dfname+' does not match any of the requested fields'

        # ---------------------------------
        # Check heights argument (optional)
        # ---------------------------------
        try:
            # If no heights are specified, check that all datasets combined have
            # no more than one height value
            if self.heights is None:
                av_heights = set()
                for df in self.datasets.values():
                    heightvalues = _get_height_values(df)
                    try:
                        for height in heightvalues:
                            av_heights.add(height)
                    except TypeError:
                        pass
                if len(av_heights)==0:
                    # None of the datasets have height values
                    self.heights = [None,]
                elif len(av_heights)==1:
                    self.heights = list(av_heights)
                else:
                    raise InputError("found more than one height value so 'heights' argument must be specified")
            # If heights='all', retrieve heights from dataset
            elif self.heights=='all':
                self.heights = _get_height_values(list(self.datasets.values())[0])
                assert(all([np.allclose(_get_height_values(df),self.heights) for df in self.datasets.values()])), \
                    "The option heights = 'all' only works when all datasets have the same vertical levels"
            # If heights is single instance, convert to list
            elif isinstance(self.heights,(int,float)):
                self.heights = [self.heights,]
        except AttributeError:
            pass

        # ---------------------------------
        # Check times argument (optional)
        # ---------------------------------
        # If times is single instance, convert to list
        try:
            # If no times are specified, check that all datasets combined have
            # no more than one time value
            if self.times is None:
                av_times = set()
                for df in self.datasets.values():
                    timevalues = _get_time_values(df)
                    try:
                        for time in timevalues.values:
                            av_times.add(time)
                    except AttributeError:
                        pass
                if len(av_times)==0:
                    # None of the datasets have time values
                    self.times = [None,]
                elif len(av_times)==1:
                    self.times = list(av_times)
                else:
                    raise InputError("found more than one time value so 'times' argument must be specified")
            elif isinstance(self.times,(str,int,float,np.number)):
                self.times = [self.times,]
        except AttributeError:
            pass

        # -------------------------------------
        # Check fieldlimits argument (optional)
        # -------------------------------------
        # If one set of fieldlimits is specified, check number of fields
        # and convert to dictionary
        try:
            if isinstance(self.fieldlimits, (list, tuple)):
                assert(len(self.fields)==1), 'Unclear to what field fieldlimits corresponds'
                self.fieldlimits = {self.fields[0]:self.fieldlimits}
        except AttributeError:
            self.fieldlimits = {}

        # -------------------------------------
        # Check fieldlabels argument (optional)
        # -------------------------------------
        # If one fieldlabel is specified, check number of fields
        try:
            if isinstance(self.fieldlabels, str):
                assert(len(self.fields)==1), 'Unclear to what field fieldlabels corresponds'
                self.fieldlabels = {self.fields[0]: self.fieldlabels}
        except AttributeError:
            self.fieldlabels = {}

        # -------------------------------------
        # Check colorscheme argument (optional)
        # -------------------------------------
        # If one colorscheme is specified, check number of fields
        try:
            self.cmap = {}
            if isinstance(self.colorschemes, str):
                assert(len(self.fields)==1), 'Unclear to what field colorschemes corresponds'
                self.cmap[self.fields[0]] = mpl.cm.get_cmap(self.colorschemes)
            else:
            # Set missing colorschemes to viridis
                for field in self.fields:
                    if field not in self.colorschemes.keys():
                        if field == 'wdir':
                            self.colorschemes[field] = 'twilight'
                        else:
                            self.colorschemes[field] = 'viridis'
                    self.cmap[field] = mpl.cm.get_cmap(self.colorschemes[field])
        except AttributeError:
            pass

        # -------------------------------------
        # Check fieldorder argument (optional)
        # -------------------------------------
        # Make sure fieldorder is recognized
        try:
            assert(self.fieldorder in ['C','F']), "Error: fieldorder '"\
                +self.fieldorder+"' not recognized, must be either 'C' or 'F'"
        except AttributeError:
            pass


    def set_missing_fieldlimits(self):
        """
        Set missing fieldlimits to min and max over all datasets
        """
        for field in self.fields:
            if field not in self.fieldlimits.keys():
                try:
                    self.fieldlimits[field] = [
                        min([_get_field(df,field).min() for df in self.datasets.values() if _contains_field(df,field)]),
                        max([_get_field(df,field).max() for df in self.datasets.values() if _contains_field(df,field)])
                        ]
                except ValueError:
                    self.fieldlimits[field] = [None,None]


def _get_available_fieldnames(df,fieldnames):
    """
    Return subset of fields available in df
    """
    available_fieldnames = []
    if isinstance(df,pd.DataFrame):
        for field in fieldnames:
            if field in df.columns:
                available_fieldnames.append(field)
    # A Series only has one field, so return that field name
    # (if that field is not in fields, an error would have been raised)
    elif isinstance(df,pd.Series):
        available_fieldnames.append(df.name)
    return available_fieldnames


def _get_fieldnames(df):
    """
    Return list of fieldnames in df
    """
    if isinstance(df,pd.DataFrame):
        return list(df.columns)
    elif isinstance(df,pd.Series):
        return [df.name,]


def _contains_field(df,fieldname):
    if isinstance(df,pd.DataFrame):
        return fieldname in df.columns
    elif isinstance(df,pd.Series):
        return (df.name is None) or (df.name==fieldname)


def _get_height_values(df):
    """
    Return list of heights in df
    """
    if isinstance(df.index,pd.MultiIndex):
        # Assuming level 0 is time and level 1 is height
        return df.index.get_level_values(1).unique().values
    elif isinstance(df,pd.DataFrame):
        for name in height_names:
            # First look among columns
            if name in df.columns:
                return df[name].unique()
            # Then check whether index corresponds to height
            elif df.index.name==name:
                return df.index.unique().values
        # No height name match
        return None
    elif isinstance(df,pd.Series):
        if df.index.name in height_names:
            return df.index.unique().values
        else:
            return None


def _get_time_values(df):
    """
    Return list of times in df
    """
    if isinstance(df.index,pd.MultiIndex):
        # Assuming level 0 is always the time index
        return df.index.get_level_values(0).unique()
    else:
        # If index has a height name, dataset has no time
        if df.index.name in height_names:
            return None
        else:
            return df.index.unique()


def _get_pivot_table(df,fieldnames):
    """
    Return pivot table with given fieldnames as columns
    """
    if isinstance(df,pd.DataFrame):
        if isinstance(df.index,pd.MultiIndex):
            # Assuming level 0 is time and level 1 is height
            return df[fieldnames].unstack()
        else:
            for name in height_names:
                if name in df.columns:
                    return df.pivot(columns=name,values=fieldnames)
            # No height name match
            return None
    elif isinstance(df,pd.Series):
        if isinstance(df.index,pd.MultiIndex):
            # Assuming level 0 is time and level 1 is height
            return df.unstack()
        else:
            return df


def _get_field(df,fieldname):
    """
    Return field from dataset
    """
    if isinstance(df,pd.DataFrame):
        return df[fieldname]
    elif isinstance(df,pd.Series):
        if df.name is None or df.name==fieldname:
            return df
        else:
            return None


def _get_pivoted_field(df,fieldname):
    """
    Return field from pivoted dataset
    """
    if isinstance(df.columns,pd.MultiIndex):
        return df[fieldname]
    else:
        return df


def _create_subplots_if_needed(ntotal,
                               ncols=None,
                               default_ncols=1,
                               fieldorder='C',
                               avoid_single_column=False,
                               sharex=False,
                               sharey=False,
                               subfigsize=(12,3),
                               wspace=0.4,
                               hspace=0.4,
                               fig=None,
                               ax=None
                               ):
    """
    Auxiliary function to create fig and ax

    If fig and ax are None:
    - Set nrows and ncols based on ntotal and specified ncols,
      accounting for fieldorder and avoid_single_column
    - Create fig and ax with nrows and ncols, taking into account
      sharex, sharey, subfigsize, wspace, hspace

    If fig and ax are not None:
    - Try to determine nrows and ncols from ax
    - Check whether size of ax corresponds to ntotal
    """

    if ax is None:
        if not ncols is None:
            # Use ncols if specified and appropriate
            assert(ntotal%ncols==0), 'Error: Specified number of columns is not a true divisor of total number of subplots'
            nrows = int(ntotal/ncols)
        else:
            # Defaut number of columns
            ncols = default_ncols
            nrows = int(ntotal/ncols)
    
            if fieldorder=='F':
                # Swap number of rows and columns
                nrows, ncols = ncols, nrows
            
            if avoid_single_column and ncols==1:
                # Swap number of rows and columns
                nrows, ncols = ncols, nrows

        # Create fig and ax with nrows and ncols
        fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharex=sharex,sharey=sharey,figsize=(subfigsize[0]*ncols,subfigsize[1]*nrows))

        # Adjust subplot spacing
        fig.subplots_adjust(wspace=wspace,hspace=hspace)

    else:
        # Make sure user-specified axes has appropriate size
        assert(np.asarray(ax).size==ntotal), 'Specified axes does not have the right size'

        # Determine nrows and ncols in specified axes
        if isinstance(ax,mpl.axes.Axes):
            nrows, ncols = (1,1)
        else:
            try:
                nrows,ncols = np.asarray(ax).shape
            except ValueError:
                # ax array has only one dimension
                # Determine whether ax is single row or single column based
                # on individual ax positions x0 and y0
                x0s = [axi.get_position().x0 for axi in ax]
                y0s = [axi.get_position().y0 for axi in ax]
                if all(x0==x0s[0] for x0 in x0s):
                    # All axis have same relative x0 position
                    nrows = np.asarray(ax).size
                    ncols = 1
                elif all(y0==y0s[0] for y0 in y0s):
                    # All axis have same relative y0 position
                    nrows = 1
                    ncols = np.asarray(ax).size
                else:
                    # More complex axes configuration,
                    # currently not supported
                    raise InputError('could not determine nrows and ncols in specified axes, complex axes configuration currently not supported')

    return fig, ax, nrows, ncols


def _format_time_axis(ax,
                      plot_local_time,
                      local_time_offset,
                      timelimits
                      ):
    """
    Auxiliary function to format time axis
    """
    ax[-1].xaxis_date()
    hour_interval = _determine_hourlocator_interval(ax[-1],timelimits)
    if plot_local_time:
        # Format first axis (local time)
        ax[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=hour_interval))
        ax[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%I %P'))
        ax[-1].xaxis.set_major_locator(mdates.DayLocator(interval=12)) #Choose large interval so dates are not plotted
        ax[-1].xaxis.set_major_formatter(mdates.DateFormatter(''))

        # Set time limits if specified
        if not timelimits is None:
            local_timelimits = pd.to_datetime(timelimits) + pd.to_timedelta(local_time_offset,'h')
            ax[-1].set_xlim(local_timelimits)

        tstr = 'Local time'

        for axi in ax:
            # Format second axis (UTC time)
            ax2 = axi.twiny()
            ax2.xaxis_date()
    
            # Set time limits if specified
            if not timelimits is None:
                ax2.set_xlim(timelimits)
            else:
                # Extract timelimits from main axis
                local_timelimits = mdates.num2date(axi.get_xlim())
                timelimits = pd.to_datetime(local_timelimits) - pd.to_timedelta(local_time_offset,'h')
                ax2.set_xlim(timelimits)
    
            # Move twinned axis ticks and label from top to bottom
            ax2.xaxis.set_ticks_position("bottom")
            ax2.xaxis.set_label_position("bottom")
    
            # Offset the twin axis below the host
            ax2.spines["bottom"].set_position(("axes", -0.35))
    
            # Turn on the frame for the twin axis, but then hide all 
            # but the bottom spine
            ax2.set_frame_on(True)
            ax2.patch.set_visible(False)
            #for sp in ax2.spines.itervalues():
            #    sp.set_visible(False)
            ax2.spines["bottom"].set_visible(True)
    
            ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=hour_interval))
            ax2.xaxis.set_minor_formatter(mdates.DateFormatter('%H%M'))
            ax2.xaxis.set_major_locator(mdates.DayLocator())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y-%m-%d'))
            ax2.set_xlabel('UTC time')
    else:
        ax[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=hour_interval))
        ax[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H%M'))
        ax[-1].xaxis.set_major_locator(mdates.DayLocator())
        ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('\n%Y-%m-%d'))

        # Set time limits if specified
        if not timelimits is None:
            ax[-1].set_xlim(timelimits)

        tstr = 'UTC time'
        ax2 = None

    # Set time label
    for axi in ax:
        axi.set_xlabel(tstr)

    return ax2

def _determine_hourlocator_interval(ax,timelimits=None):
    """
    Determine hour interval based on timelimits

    If plotted time period is
    - less than 36 hours: interval = 3
    - less than 72 hours: interval = 6
    - otherwise:          interval = 12
    """
    # Get timelimits
    if timelimits is None:
        timelimits = pd.to_datetime(mdates.num2date(ax.get_xlim()))
    elif isinstance(timelimits[0],str):
        timelimits = pd.to_datetime(timelimits)

    # Determine time period in hours
    timeperiod = (timelimits[1] - timelimits[0])/pd.to_timedelta(1,'h')
    # HourLocator interval
    if timeperiod < 36:
        return 3
    elif timeperiod < 72:
        return 6
    else:
        return 12
