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
# - Fix FutureWarning:
#   .../python3.6/site-packages/pandas/plotting/_converter.py:129: FutureWarning:
#   Using an implicitly registered datetime converter for a matplotlib plotting
#   method. The converter was registered by pandas on import. Future versions of
#   pandas will require you to explicitly register matplotlib converters.
#
#   To register the converters:
#        >>> from pandas.plotting import register_matplotlib_converters
#            >>> register_matplotlib_converters()
#              warnings.warn(msg, FutureWarning)
# - Allow custom colors/styles/markers?
# - Separate out calculation of spectra?

# Standard field labels
fieldLabels = {'wspd': r'Wind speed [m/s]',
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

# Default color cycle
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_timeheight(datasets,
                    fields,
                    colorscheme={},
                    fieldlimits={},
                    heightlimits=None,
                    timelimits=None,
                    labelsubplots=False,
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
    labelsubplots : bool
        Label subplots as (a), (b), (c), ...
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

    # If a single dataset is provided, convert to a dictionary
    # under a generic key 'Dataset'
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

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

    #Order plots depending on number of datasets and fields
    Ndatasets = len(datasets)
    Nfields = len(fields)

    fig,ax = plt.subplots(nrows=Ndatasets*Nfields,ncols=1,sharex=True,sharey=True,figsize=(12.0,4.0*Ndatasets*Nfields))

    if Ndatasets*Nfields==1:
        axs = [ax,]
    else:
        axs = ax.ravel()

    # Initialise list of colorbars
    cbars = []

    # Adjust subplot spacing
    fig.subplots_adjust(wspace=0.4,hspace=0.4)

    # Loop over datasets, fields and times 
    for i, dfname in enumerate(datasets):
        df = datasets[dfname]

        heightvalues = df['height'].unique()
        timevalues = mdates.date2num(df.index.unique().get_values())
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

            # Index of axis corresponding to dataset i and field j
            axi = i*Nfields + j

            # Plot data
            fieldvalues = df_pivot[field].values 
            im = axs[axi].pcolormesh(Ts,Zs,fieldvalues.T,
                          vmin=fieldlimits[field][0],vmax=fieldlimits[field][1],
                          cmap=cmap[field],
                          **kwargs
                          )

            # Colorbar mark up
            cbar = fig.colorbar(im,ax=axs[axi],shrink=1.0)
            # Set field label if known
            try:
                cbar.set_label(fieldLabels[field])
            except KeyError:
                pass
            # Save colorbar
            cbars.append(cbar)

            # Set title if more than one dataset
            if len(datasets)>1:
                axs[axi].set_title(dfname)



    # Axis mark up
    axs[-1].set_xlabel(r'UTC time')
    axs[-1].xaxis_date()
    axs[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=6))
    axs[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H%M'))
    axs[-1].xaxis.set_major_locator(mdates.DayLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('\n%Y-%m-%d'))

    # Set time and height limits if specified
    if not timelimits is None:
        axs[-1].set_xlim(timelimits)
    if not heightlimits is None:
        axs[-1].set_ylim(heightlimits)

    # Add y labels
    for ax in axs: 
        ax.set_ylabel(r'Height [m]')
    
    # Number sub figures as a, b, c, ...
    if labelsubplots and len(axs) > 1:
        for i,ax in enumerate(axs):
            ax.text(-0.14,1.0,'('+chr(i+97)+')',transform=ax.transAxes,size=16)

    return fig, axs, cbars


def plot_timehistory_at_height(datasets,
                               fields,
                               heights,
                               fieldlimits={},
                               timelimits=None,
                               colormap=None,
                               stack_by=None,
                               labelsubplots=False,
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
    fieldlimits : list or tuple, or dict
        Value range for the various fields. If only one field is 
        plotted, fieldlimits can be a list or tuple. Otherwise, it
        should be a dictionary with entries <fieldname>: fieldlimit.
        Missing fieldlimits are set automatically
    timelimits : list or tuple
        Time axis limits
    colormap : str
        Colormap used when stacking heights
    stack_by : str
        Stack by 'heights' or by 'datasets'
    labelsubplots : bool
        Label subplots as (a), (b), (c), ...
    **kwargs : other keyword arguments
        Options that are passed on to the actual plotting function.
        Note that these options should be the same for all datasets,
        fields and heights, and they can not be used to set dataset,
        field or height specific colors, limits, etc.
        Example uses include setting linestyle/width, marker, etc.
    """

    # If any of fields or heights is a single instance,
    # convert to a list
    if isinstance(fields,str):
        fields = [fields,]
    if isinstance(heights,(int,float)):
        heights = [heights,]

    # If a single dataset is provided, convert to a dictionary
    # under a generic key 'Dataset'
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    # If one set of fieldlimits is specified, check number of fields
    # and convert to dictionary
    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}

    # Set up subplot grid
    Ndatasets = len(datasets)
    Nfields = len(fields)
    Nheights = len(heights)

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
    fig,axs = plt.subplots(nrows=nrows,sharex=True,figsize=(12,3.0*nrows))
    if Nfields==1: axs = [axs,]

    # Loop over datasets and fields 
    for i,dfname in enumerate(datasets):
        df = datasets[dfname]
        timevalues = df.index.unique()
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


            for k, height in enumerate(heights):
                # Axis order, label and title depend on value of stack_by 
                if stack_by=='heights':
                    # Index of axis corresponding to field j and dataset i 
                    axi = i*Nfields + j

                    # Use height as label
                    label = 'z = {:.1f} m'.format(height)

                    # Set title if multiple datasets are compared
                    if Ndatasets>1:
                        axs[axi].set_title(dfname)

                    # Set colors
                    if colormap is not None:
                        cmap = mpl.cm.get_cmap(colormap)
                        color = cmap(k/(Nheights-1))
                    else:
                        color = default_colors[k]
                else:
                    # Index of axis corresponding to field j and height k
                    axi = k*Nfields + j

                    # Use datasetname as label
                    label = dfname

                    # Set title if multiple heights are compared
                    if Nheights>1:
                        axs[axi].set_title('z = {:.1f} m'.format(height))

                    # Set colors
                    color = default_colors[i]

                # Plot data
                signal = interp1d(heightvalues,df_pivot[field].values,axis=1,fill_value="extrapolate")(height)
                axs[axi].plot_date(timevalues,signal,label=label,color=color,**kwargs)

                # Set field label if known
                try:
                    axs[axi].set_ylabel(fieldLabels[field])
                except KeyError:
                    pass
                # Set field limits if specified
                try:
                    axs[axi].set_ylim(fieldlimits[field])
                except KeyError:
                    pass
   
    # Set axis grid
    for ax in axs:
        ax.xaxis.grid(True,which='minor')
        ax.yaxis.grid()
    
    # Format time axis
    axs[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=6))
    axs[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H%M'))
    axs[-1].xaxis.set_major_locator(mdates.DayLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('\n%Y-%m-%d'))
    axs[-1].set_xlabel(r'UTC time')

    # Set time limits if specified
    if not timelimits is None:
        axs[-1].set_xlim(timelimits)

    # Number sub figures as a, b, c, ...
    if labelsubplots and len(axs) > 1:
        for i,ax in enumerate(axs):
            ax.text(-0.14,1.0,'('+chr(i+97)+')',transform=ax.transAxes,size=16)

    # Add legend if more than one entry
    if (stack_by=='datasets' and Ndatasets>1) or (stack_by=='heights' and Nheights>1):
        leg = axs[0].legend(loc='upper left',bbox_to_anchor=(1.05,1.0),fontsize=16)

    return fig, axs


def plot_profile(datasets,
                 fields,
                 times,
                 fieldlimits={},
                 heightlimits=None,
                 colormap=None,
                 stack_by=None,
                 labelsubplots=False,
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
    fieldlimits : list or tuple, or dict
        Value range for the various fields. If only one field is 
        plotted, fieldlimits can be a list or tuple. Otherwise, it
        should be a dictionary with entries <fieldname>: fieldlimit.
        Missing fieldlimits are set automatically
    heightlimits : list or tuple
        Height axis limits
    colormap : str
        Colormap used when stacking times
    stack_by : str
        Stack by 'times' or by 'datasets'
    labelsubplots : bool
        Label subplots as (a), (b), (c), ...
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
    if isinstance(times,(str,int,float,np.number)):
        times = [times,]

    # If a single dataset is provided, convert to a dictionary
    # under a generic key 'Dataset'
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    # If one set of fieldlimits is specified, check number of fields
    # and convert to dictionary
    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}

    #Order plots depending on number of times and fields
    Ndatasets = len(datasets)
    Nfields = len(fields)
    Ntimes = len(times)

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

    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharey=True,figsize=(4*ncols,5*nrows))
    if ncols*nrows==1:
        axs = [ax,]
    else:
        axs = ax.ravel()

    # Adjust subplot spacing
    fig.subplots_adjust(wspace=0.2,hspace=0.4)

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
                # Axis order, label and title depend on value of stack_by 
                if stack_by=='times':
                    # Index of axis corresponding to field j and dataset i
                    axi = j*Ndatasets + i
                    
                    # Use time as label
                    if isinstance(time, (int,float,np.number)):
                        label = '{:g} s'.format(time)
                    else:
                        label = pd.to_datetime(time).strftime('%Y-%m-%d %H%M UTC')

                    # Set title if multiple datasets are compared
                    if Ndatasets>1:
                        axs[axi].set_title(dfname)

                    # Set colors
                    if colormap is not None:
                        cmap = mpl.cm.get_cmap(colormap)
                        color = cmap(k/(Ntimes-1))
                    else:
                        color = default_colors[k]
                else:
                    # Index of axis corresponding to field j and time k
                    axi = j*Ntimes + k

                    # Use datasetname as label
                    label = dfname

                    # Set title if multiple times are compared
                    if Ntimes>1:
                        if isinstance(time, (int,float,np.number)):
                            tstr = '{:g} s'.format(time)
                        else:
                            tstr = pd.to_datetime(time).strftime('%Y-%m-%d %H%M UTC')
                        axs[axi].set_title(tstr, fontsize=16)

                    # Set color
                    color = default_colors[i]
                
                # Plot data
                fieldvalues = df_pivot[field].loc[time].values.squeeze()
                axs[axi].plot(fieldvalues,heightvalues,label=label,color=color,**kwargs)

                # Set field label if known
                try:
                    axs[axi].set_xlabel(fieldLabels[field])
                except KeyError:
                    pass
                # Set field limits if specified
                try:
                    axs[axi].set_xlim(fieldlimits[field])
                except KeyError:
                    pass
    
    for ax in axs:
        ax.grid(True,which='both')

    # Set height limits if specified
    if not heightlimits is None:
        axs[0].set_ylim(heightlimits)

    # Add y labels
    for r in range(nrows): 
        axs[r*ncols].set_ylabel(r'Height [m]')
    
    # Number sub figures as a, b, c, ...
    if labelsubplots and len(axs) > 1:
        for i,ax in enumerate(axs):
            ax.text(-0.14,-0.18,'('+chr(i+97)+')',transform=ax.transAxes,size=16)
    
    # Add legend if more than one entry
    if (stack_by=='datasets' and Ndatasets>1) or (stack_by=='times' and Ntimes>1):
        leg = axs[ncols-1].legend(loc='upper left',bbox_to_anchor=(1.05,1.0),fontsize=16)

    return fig,axs


def plot_spectrum(datasets,
                  height,
                  fields,
                  times,
                  Tperiod=3600.0,
                  Tsegment=600.0,
                  fieldlimits={},
                  freqlimits=None,
                  labelsubplots=False,
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
    labelsubplots : bool
        Label subplots as (a), (b), (c), ...
    **kwargs : other keyword arguments
        Options that are passed on to the actual plotting function.
        Note that these options should be the same for all datasets,
        fields and times, and they can not be used to set dataset,
        field or time specific colors, limits, etc.
        Example uses include setting linestyle/width, marker, etc.
    """

    # Some custom field labels for frequency spectra
    fieldSpectrumLabels = {'u': r'$E_{uu}\;[\mathrm{m^2/s}]$',
                           'v': r'$E_{vv}\;[\mathrm{m^2/s}]$',
                           'w': r'$E_{ww}\;[\mathrm{m^2/s}]$',
                           'wspd': r'$E_{UU}\;[\mathrm{m^2/s}]$',
                           }

    # If any of fields or times is a single instance,
    # convert to a list
    if isinstance(fields,str):
        fields = [fields,]
    if isinstance(times,str):
        times = [times,]

    # If a single dataset is provided, convert to a dictionary
    # under a generic key 'Dataset'
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    # If one set of fieldlimits is specified, check number of fields
    # and convert to dictionary
    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}

    #Order plots depending on number of datasets and fields
    Ntimes = len(times)
    Nfields = len(fields)

    nrows, ncols = _calc_nrows_ncols(Ntimes,Nfields)
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(4*ncols,5*nrows))

    if ncols*nrows==1:
        axs = [ax,]
    else:
        axs = ax.ravel()

    # Adjust subplot spacing
    fig.subplots_adjust(wspace=0.3,hspace=0.5)

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
                # Index of axis corresponding to field k and time i
                axi = k*Ntimes + i
                
                # Axes mark up
                if j==0:
                    axs[axi].set_title(pd.to_datetime(tstart).strftime('%Y-%m-%d %H%M UTC'),fontsize=16)

                # Compute frequency spectrum
                istart = np.where(timevalues==pd.to_datetime(tstart))[0][0]
                signal = interp1d(heightvalues,df_pivot[field].interpolate(method='linear').values,axis=1,fill_value="extrapolate")(height)
                f, P = welch(signal[istart:istart+np.int(Tperiod/dt)],fs=1./dt,nperseg=np.int(Tsegment/dt),
                            detrend='linear',window='hanning',scaling='density')

                # Plot data
                axs[axi].loglog(f[1:],P[1:],label=dfname,**kwargs)
   

    # Set frequency label
    for c in range(ncols):
        axs[ncols*(nrows-1)+c].set_xlabel('f [Hz]')

    # Specify field label and field limits if specified 
    for r in range(nrows):
        try:
            axs[r*ncols].set_ylabel(fieldSpectrumLabels[fields[r]])
        except KeyError:
            pass
        try:
            axs[r*ncols].set_ylim(fieldlimits[fields[r]])
        except KeyError:
            pass

    # Set frequency limits if specified
    if not freqlimits is None:
        axs[0].set_xlim(freqlimits)

    # Number sub figures as a, b, c, ...
    if labelsubplots and len(axs) > 1:
        for i,ax in enumerate(axs):
            ax.text(-0.14,-0.18,'('+chr(i+97)+')',transform=ax.transAxes,size=16)

    # Add legend if more than one dataset
    if len(datasets)>1:
        leg = axs[ncols-1].legend(loc='upper left',bbox_to_anchor=(1.05,1.0))

    return fig, axs


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
