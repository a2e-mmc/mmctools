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

def plot_timeheight(datasets,
                    fields,
                    fig=None,ax=None,
                    colorscheme={},
                    fieldlimits={},
                    heightlimits=None,
                    timelimits=None,
                    fieldlabels={},
                    labelsubplots=False,
                    showcolorbars=True,
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
    fields : str or list
        Fieldname(s) corresponding to particular column(s) of
        the datasets
    fig : figure handle
        Custom figure handle. Should be specified together with ax
    ax : axes handle or numpy ndarray with axes handles
        Customand axes handle(s).
        Size of ax should equal Ndatasets*Nfields
    colorscheme : str or dict
        Name of colorscheme. If only one field is plotted, colorscheme
        can be a string. Otherwise, it should be a dictionary with
        entries <fieldname>: name_of_colorscheme
        Missing colorschemes are set to 'viridis'
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
    datasetkwargs : dict
        Dataset-specific options that are passed on to the actual
        plotting function. These options overwrite general options
        specified through **kwargs. The argument should be a dictionary
        with entries <dataset_name>: {**kwargs}
    **kwargs : other keyword arguments
        Options that are passed on to the actual plotting function.
        Note that these options should be the same for all datasets and
        fields and can not be used to set dataset or field specific
        limits, colorschemes, norms, etc.
        Example uses include setting shading, rasterized, etc.
    """

    # If fields is a single instance,
    # convert to a list
    if isinstance(fields,str):
        fields = [fields,]
    Nfields = len(fields)

    # If a single dataset is provided, convert to a dictionary
    # under a generic key 'Dataset'
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}
    Ndatasets = len(datasets)

    # If one set of fieldlimits is specified, check number of fields
    # and convert to dictionary
    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}
    else:
        # Set missing fieldlimits to min and max over all datasets
        for field in fields:
            if field not in fieldlimits.keys():
                fieldlimits[field] = [ min([df[field].min() for df in datasets.values()]),
                                   max([df[field].max() for df in datasets.values()]) ]

    # If one colorscheme is specified, check number of fields
    cmap = {}
    if isinstance(colorscheme, str):
        assert(len(fields)==1), 'Unclear to what field colorscheme corresponds'
        cmap[fields[0]] = mpl.cm.get_cmap(colorscheme)
    else:
    # Set missing colorschemes to viridis
        for field in fields:
            if field not in colorscheme.keys():
                colorscheme[field] = 'viridis'
            cmap[field] = mpl.cm.get_cmap(colorscheme[field])

    # If one fieldlabel is specified, check number of fields
    if isinstance(fieldlabels, str):
        assert(len(fields)==1), 'Unclear to what field fieldlabels corresponds'
        fieldlabels = {fields[0]: fieldlabels}

    # Concatenate custom and standard field labels
    # (custom field labels overwrite standard fields labels if existent)
    fieldlabels = {**standard_fieldlabels, **fieldlabels}        

    # Create new figure and axes if not specified
    if ax is None:
        fig,ax = plt.subplots(nrows=Ndatasets*Nfields,ncols=1,sharex=True,sharey=True,figsize=(12.0,4.0*Ndatasets*Nfields))
        # Adjust subplot spacing
        fig.subplots_adjust(wspace=0.4,hspace=0.4)

    # Create flattened view of axes
    axv = np.asarray(ax).reshape(-1)

    # Make sure axv has right size (important when using user-specified axes)
    assert(axv.size==Ndatasets*Nfields), 'Number of axes does not match number of datasets and fields'

    # Initialise list of colorbars
    cbars = []

    # Loop over datasets, fields and times 
    for i, dfname in enumerate(datasets):
        df = datasets[dfname]

        heightvalues = df['height'].unique()
        timevalues = mdates.date2num(df.index.unique().values) # Convert to days since 0001-01-01 00:00 UTC, plus one
        Ts,Zs = np.meshgrid(timevalues,heightvalues,indexing='xy')

        # Create list with available fields only
        available_fields = []
        for field in fields:
            if field in df.columns:
                available_fields.append(field)
        assert(len(available_fields)>0), 'Dataset '+dfname+' does not contain any of the requested fields'

        # Pivot all fields in a dataset at once
        df_pivot = df.pivot(columns='height',values=available_fields)

        for j, field in enumerate(fields):
            # Skip loop if field not available
            if not field in available_fields:
                print('Warning: field "'+field+'" not available in dataset '+dfname)
                continue

            # Store plotting options in dictionary
            plotting_properties = {
                'vmin': fieldlimits[field][0],
                'vmax': fieldlimits[field][1],
                'cmap': cmap[field]
                }

            # Index of axis corresponding to dataset i and field j
            axi = i*Nfields + j

            # Extract data from dataframe
            fieldvalues = df_pivot[field].values 

            # Gather fieldlimits, colormap, general options and dataset-specific options
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
                    cbar.set_label(fieldlabels[field])
                except KeyError:
                    pass
                # Save colorbar
                cbars.append(cbar)

            # Set title if more than one dataset
            if len(datasets)>1:
                axv[axi].set_title(dfname,fontsize=16)



    # Axis mark up
    axv[-1].set_xlabel(r'UTC time')
    axv[-1].xaxis_date()
    axv[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=6))
    axv[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H%M'))
    axv[-1].xaxis.set_major_locator(mdates.DayLocator())
    axv[-1].xaxis.set_major_formatter(mdates.DateFormatter('\n%Y-%m-%d'))

    # Set time and height limits if specified
    if not timelimits is None:
        axv[-1].set_xlim(timelimits)
    if not heightlimits is None:
        axv[-1].set_ylim(heightlimits)

    # Add y labels
    for axi in axv: 
        axi.set_ylabel(r'Height [m]')
    
    # Number sub figures as a, b, c, ...
    if labelsubplots and axv.size > 1:
        for i,axi in enumerate(axv):
            axi.text(-0.14,1.0,'('+chr(i+97)+')',transform=axi.transAxes,size=16)

    return fig, ax, cbars


def plot_timehistory_at_height(datasets,
                               fields,
                               heights,
                               fig=None,ax=None,
                               fieldlimits={},
                               timelimits=None,
                               fieldlabels={},
                               colormap=None,
                               stack_by=None,
                               labelsubplots=False,
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
    fields : str or list
        Fieldname(s) corresponding to particular column(s) of
        the datasets
    heights : float or list
        Height(s) for which time history is plotted
    fig : figure handle
        Custom figure handle. Should be specified together with ax
    ax : axes handle or numpy ndarray with axes handles
        Customand axes handle(s).
        Size of ax should equal Nfields * (Ndatasets or Nheights)
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
    colormap : str
        Colormap used when stacking heights
    stack_by : str
        Stack by 'heights' or by 'datasets'
    labelsubplots : bool
        Label subplots as (a), (b), (c), ...
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

    # If any of fields or heights is a single instance,
    # convert to a list
    if isinstance(fields,str):
        fields = [fields,]
    Nfields = len(fields)

    if isinstance(heights,(int,float)):
        heights = [heights,]
    Nheights = len(heights)

    # If a single dataset is provided, convert to a dictionary
    # under a generic key 'Dataset'
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}
    Ndatasets = len(datasets)

    # If one set of fieldlimits is specified, check number of fields
    # and convert to dictionary
    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}

    # If one fieldlabel is specified, check number of fields
    if isinstance(fieldlabels, str):
        assert(len(fields)==1), 'Unclear to what field fieldlabels corresponds'
        fieldlabels = {fields[0]: fieldlabels}

    # Concatenate custom and standard field labels
    # (custom field labels overwrite standard fields labels if existent)
    fieldlabels = {**standard_fieldlabels, **fieldlabels}

    # Set up subplot grid
    if stack_by is None:
        if Nheights>1:
            stack_by = 'heights'
        else:
            stack_by = 'datasets'
    else:
        assert(stack_by in ['heights','datasets']), 'Error: stack by "'\
            +stack_by+'" not recognized, choose either "heights" or "datasets"'

    if stack_by=='heights':
        nrows = Nfields*Ndatasets
    else:
        nrows = Nfields*Nheights

    # Create new figure and axes if not specified
    if ax is None:
        fig,ax = plt.subplots(nrows=nrows,sharex=True,figsize=(12.0,3.0*nrows))

    # Create flattened view of axes
    axv = np.asarray(ax).reshape(-1)

    # Make sure axv has right size (important when using user-specified axes)
    assert(axv.size==nrows), 'Number of axes does not match number of datasets/heights and fields'

    # Loop over datasets and fields 
    for i,dfname in enumerate(datasets):
        df = datasets[dfname]
        timevalues = df.index.unique().values
        heightvalues = df['height'].unique()

        # Create list with available fields only
        available_fields = []
        for field in fields:
            if field in df.columns:
                available_fields.append(field)
        assert(len(available_fields)>0), 'Dataset '+dfname+' does not contain any of the requested fields'

        # If any of the requested heights is not available,
        # pivot the dataframe to allow interpolation.
        # Pivot all fields in a dataset at once to reduce computation time
        if not all([h in heightvalues for h in heights]):
            df_pivot = df.pivot(columns='height',values=available_fields)
            print('Pivoting '+dfname)

        for j, field in enumerate(fields):
            # Skip loop if field not available
            if not field in available_fields:
                print('Warning: field "'+field+'" not available in dataset '+dfname)
                continue


            for k, height in enumerate(heights):
                # Store plotting options in dictionary
                # Set default linestyle to '-' and no markers
                plotting_properties = {
                    'linestyle':'-',
                    'marker':None,
                    }

                # Axis order, label and title depend on value of stack_by 
                if stack_by=='heights':
                    # Index of axis corresponding to field j and dataset i 
                    axi = i*Nfields + j

                    # Use height as label
                    plotting_properties['label'] = 'z = {:.1f} m'.format(height)

                    # Set title if multiple datasets are compared
                    if Ndatasets>1:
                        axv[axi].set_title(dfname,fontsize=16)

                    # Set colors
                    if colormap is not None:
                        cmap = mpl.cm.get_cmap(colormap)
                        plotting_properties['color'] = cmap(k/(Nheights-1))
                    else:
                        plotting_properties['color'] = default_colors[k]
                else:
                    # Index of axis corresponding to field j and height k
                    axi = k*Nfields + j

                    # Use datasetname as label
                    plotting_properties['label'] = dfname

                    # Set title if multiple heights are compared
                    if Nheights>1:
                        axv[axi].set_title('z = {:.1f} m'.format(height),fontsize=16)

                    # Set colors
                    plotting_properties['color'] = default_colors[i]

                # Extract data from dataframe
                if height in heightvalues:
                    signal = df.loc[df.height==height,field].values
                else:
                    signal = interp1d(heightvalues,df_pivot[field].values,axis=1,fill_value="extrapolate")(height)
                
                # Gather label, color, general options and dataset-specific options
                # (highest priority to dataset-specific options, then general options)
                try:
                    plotting_properties = {**plotting_properties,**kwargs,**datasetkwargs[dfname]}
                except KeyError:
                    plotting_properties = {**plotting_properties,**kwargs}
                
                # Plot data
                axv[axi].plot_date(timevalues,signal,**plotting_properties)

                # Set field label if known
                try:
                    axv[axi].set_ylabel(fieldlabels[field])
                except KeyError:
                    pass
                # Set field limits if specified
                try:
                    axv[axi].set_ylim(fieldlimits[field])
                except KeyError:
                    pass
   
    # Set axis grid
    for axi in axv:
        axi.xaxis.grid(True,which='minor')
        axi.yaxis.grid()
    
    # Format time axis
    axv[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=6))
    axv[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H%M'))
    axv[-1].xaxis.set_major_locator(mdates.DayLocator())
    axv[-1].xaxis.set_major_formatter(mdates.DateFormatter('\n%Y-%m-%d'))
    axv[-1].set_xlabel(r'UTC time')

    # Set time limits if specified
    if not timelimits is None:
        axv[-1].set_xlim(timelimits)

    # Number sub figures as a, b, c, ...
    if labelsubplots and axv.size > 1:
        for i,axi in enumerate(axv):
            axi.text(-0.14,1.0,'('+chr(i+97)+')',transform=axi.transAxes,size=16)

    # Add legend if more than one entry
    if (stack_by=='datasets' and Ndatasets>1) or (stack_by=='heights' and Nheights>1):
        leg = axv[0].legend(loc='upper left',bbox_to_anchor=(1.05,1.0),fontsize=16)

    return fig, ax


def plot_profile(datasets,
                 fields,
                 times,
                 fig=None,ax=None,
                 fieldlimits={},
                 heightlimits=None,
                 fieldlabels={},
                 colormap=None,
                 stack_by=None,
                 labelsubplots=False,
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
    fields : str or list
        Fieldname(s) corresponding to particular column(s) of
        the datasets
    times : str, list
        Time(s) for which vertical profiles are plotted
    fig : figure handle
        Custom figure handle. Should be specified together with ax
    ax : axes handle or numpy ndarray with axes handles
        Customand axes handle(s).
        Size of ax should equal Nfields * (Ndatasets or Ntimes)
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
    colormap : str
        Colormap used when stacking times
    stack_by : str
        Stack by 'times' or by 'datasets'
    labelsubplots : bool
        Label subplots as (a), (b), (c), ...
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

    # If any of fields or times is a single instance,
    # convert to a list
    if isinstance(fields,str):
        fields = [fields,]
    Nfields = len(fields)

    if isinstance(times,str):
        times = [times,]
    Ntimes = len(times)

    # If a single dataset is provided, convert to a dictionary
    # under a generic key 'Dataset'
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}
    Ndatasets = len(datasets)

    # If one set of fieldlimits is specified, check number of fields
    # and convert to dictionary
    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}

    # If one fieldlabel is specified, check number of fields
    if isinstance(fieldlabels, str):
        assert(len(fields)==1), 'Unclear to what field fieldlabels corresponds'
        fieldlabels = {fields[0]: fieldlabels}

    # Concatenate custom and standard field labels
    # (custom field labels overwrite standard fields labels if existent)
    fieldlabels = {**standard_fieldlabels, **fieldlabels}

    # Set up subplot grid
    if stack_by is None:
        if Ntimes>1:
            stack_by = 'times'
        else:
            stack_by = 'datasets'
    else:
        assert(stack_by in ['times','datasets']), 'Error: stack by "'\
            +stack_by+'" not recognized, choose either "times" or "datasets"'

    if stack_by=='times':
        nrows, ncols = _calc_nrows_ncols(Ndatasets,Nfields)
    else:
        nrows, ncols = _calc_nrows_ncols(Ntimes,Nfields)

    # Create new figure and axes if not specified
    if ax is None:
        fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharey=True,figsize=(4*ncols,5*nrows))
        # Adjust subplot spacing
        fig.subplots_adjust(wspace=0.2,hspace=0.4)

    # Create flattened view of axes
    axv = np.asarray(ax).reshape(-1)

    # Make sure axv has right size (important when using user-specified axes)
    assert(axv.size==nrows*ncols), 'Number of axes does not match number of datasets/times and fields'

    # Loop over datasets, fields and times 
    for i, dfname in enumerate(datasets):
        df = datasets[dfname]
        heightvalues = df['height'].unique()

        # Create list with available fields only
        available_fields = []
        for field in fields:
            if field in df.columns:
                available_fields.append(field)
        assert(len(available_fields)>0), 'Dataset '+dfname+' does not contain any of the requested fields'

        # Pivot all fields in a dataset at once
        df_pivot = df.pivot(columns='height',values=available_fields)

        for j, field in enumerate(fields):
            # Skip loop if field not available
            if not field in available_fields:
                print('Warning: field "'+field+'" not available in dataset '+dfname)
                continue

            for k, time in enumerate(times):
                plotting_properties = {}

                # Axis order, label and title depend on value of stack_by 
                if stack_by=='times':
                    # Index of axis corresponding to field j and dataset i
                    axi = j*Ndatasets + i
                    
                    # Use time as label
                    plotting_properties['label'] = pd.to_datetime(time).strftime('%Y-%m-%d %H%M UTC')

                    # Set title if multiple datasets are compared
                    if Ndatasets>1:
                        axv[axi].set_title(dfname,fontsize=16)

                    # Set colors
                    if colormap is not None:
                        cmap = mpl.cm.get_cmap(colormap)
                        plotting_properties['color'] = cmap(k/(Ntimes-1))
                    else:
                        plotting_properties['color'] = default_colors[k]
                else:
                    # Index of axis corresponding to field j and time k
                    axi = j*Ntimes + k

                    # Use datasetname as label
                    plotting_properties['label'] = dfname

                    # Set title if multiple times are compared
                    if Ntimes>1:
                        axv[axi].set_title(pd.to_datetime(time).strftime('%Y-%m-%d %H%M UTC'),fontsize=16)

                    # Set color
                    plotting_properties['color'] = default_colors[i]
                
                # Extract data from dataframe
                fieldvalues = df_pivot[field].loc[time].values.squeeze()

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
                    axv[axi].set_xlabel(fieldlabels[field])
                except KeyError:
                    pass
                # Set field limits if specified
                try:
                    axv[axi].set_xlim(fieldlimits[field])
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
    if (stack_by=='datasets' and Ndatasets>1) or (stack_by=='times' and Ntimes>1):
        leg = axv[ncols-1].legend(loc='upper left',bbox_to_anchor=(1.05,1.0),fontsize=16)

    return fig,ax


def plot_spectrum(datasets,
                  height,
                  fields,
                  times,
                  fig=None,ax=None,
                  Tperiod=3600.0,
                  Tsegment=600.0,
                  fieldlimits={},
                  freqlimits=None,
                  fieldlabels={},
                  labelsubplots=False,
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
    fields : str or list
        Fieldname(s) corresponding to particular column(s) of
        the datasets
    times : str, list
        Start time(s) of the time period(s) for which the frequency
        spectrum is computed.
    fig : figure handle
        Custom figure handle. Should be specified together with ax
    ax : axes handle or numpy ndarray with axes handles
        Customand axes handle(s).
        Size of ax should equal Nfields * Ntimes
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

    # If any of fields or times is a single instance,
    # convert to a list
    if isinstance(fields,str):
        fields = [fields,]
    Nfields = len(fields)

    if isinstance(times,str):
        times = [times,]
    Ntimes = len(times)

    # If a single dataset is provided, convert to a dictionary
    # under a generic key 'Dataset'
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    # If one set of fieldlimits is specified, check number of fields
    # and convert to dictionary
    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}

    # If one fieldlabel is specified, check number of fields
    if isinstance(fieldlabels, str):
        assert(len(fields)==1), 'Unclear to what field fieldlabels corresponds'
        fieldlabels = {fields[0]: fieldlabels}

    # Concatenate custom and standard field labels
    # (custom field labels overwrite standard fields labels if existent)
    fieldlabels = {**standard_spectrumlabels, **fieldlabels}

    # Set up subplot grid
    nrows, ncols = _calc_nrows_ncols(Ntimes,Nfields)

    # Create new figure and axes if not specified
    if ax is None:
        fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(4*ncols,5*nrows))
        # Adjust subplot spacing
        fig.subplots_adjust(wspace=0.3,hspace=0.5)

    # Create flattened view of axes
    axv = np.asarray(ax).reshape(-1)

    # Make sure axv has right size (important when using user-specified axes)
    assert(axv.size==nrows*ncols), 'Number of axes does not match number of times and fields'

    # Loop over datasets, fields and times 
    for j, dfname in enumerate(datasets):
        df = datasets[dfname]

        heightvalues = df['height'].unique()
        timevalues   = df.index.unique()
        dt = (timevalues[1]-timevalues[0]) / pd.Timedelta(1,unit='s')     #Sampling rate in seconds

        # Create list with available fields only
        available_fields = []
        for field in fields:
            if field in df.columns:
                available_fields.append(field)
        assert(len(available_fields)>0), 'Dataset '+dfname+' does not contain any of the requested fields'

        # Pivot all fields of a dataset at once
        df_pivot = df.pivot(columns='height',values=available_fields)

        for k, field in enumerate(fields):
            # Skip loop if field not available
            if not field in available_fields:
                print('Warning: field "'+field+'" not available in dataset '+dfname)
                continue

            for i, tstart in enumerate(times):
                plotting_properties = {'label':dfname}

                # Index of axis corresponding to field k and time i
                axi = k*Ntimes + i
                
                # Axes mark up
                if j==0:
                    axv[axi].set_title(pd.to_datetime(tstart).strftime('%Y-%m-%d %H%M UTC'),fontsize=16)

                # Compute frequency spectrum
                istart = np.where(timevalues==pd.to_datetime(tstart))[0][0]
                signal = interp1d(heightvalues,df_pivot[field].interpolate(method='linear').values,axis=1,fill_value="extrapolate")(height)
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
            axv[r*ncols].set_ylabel(fieldlabels[fields[r]])
        except KeyError:
            pass
        try:
            axv[r*ncols].set_ylim(fieldlimits[fields[r]])
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
    if len(datasets)>1:
        leg = axv[ncols-1].legend(loc='upper left',bbox_to_anchor=(1.05,1.0))

    return fig, ax


def _calc_nrows_ncols(N1,N2):
    """
    Determine number of rows and columns in a subplot grid.
    """

    if N1==1 or N2==1:
        # Organize subplots in one row.
        # If more than three columns, split over two
        # two rows if total count is even
        ncols = max([N1,N2])
        if ncols > 3 and ncols%2 == 0:
            ncols = int(ncols/2)
            nrows = 2
        else:
            nrows=1
    elif N1<=3 and N2<=4:
        # Organize subplots in N1 columns and N2 rows
        ncols = N1
        nrows = N2
    else:
        # Organize subplots in N1 rows and N2 columns
        # If more than three columns, split and divide
        # each row over two rows if number of columns is even
        nrows = N1
        ncols = N2
        if ncols > 3 and ncols%2 == 0:
            ncols = int(ncols/2)
            nrows *= 2
    return nrows, ncols
