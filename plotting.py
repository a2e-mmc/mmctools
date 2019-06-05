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


def plot_timeheight(datasets,
                    fields,
                    fieldlimits={},
                    timelimits=None,
                    heightlimits=None,
                    colorscheme={},
                    ):

    if isinstance(fields,str):
        fields = [fields,]
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}
    else:
        for field in fields:
            # Calculate missing fieldlimits
            if field not in fieldlimits.keys():
                fieldlimits[field] = [ min([df[field].min() for df in datasets.values()]),
                                   max([df[field].max() for df in datasets.values()]) ]

    cmap = {}
    if isinstance(colorscheme, str):
        assert(len(fields)==1), 'Unclear to what field colorscheme corresponds'
        cmap[fields[0]] = mpl.cm.get_cmap(colorscheme)
    else:
        for field in fields:
            # Set missing colorschemes to viridis
            if field not in colorscheme.keys():
                colorscheme[field] = 'viridis'
            cmap[field] = mpl.cm.get_cmap(colorscheme[field])

    Ndatasets = len(datasets)
    Nfields = len(fields)

    #Order plots depending on number of datasets and fields
    nrows, ncols = _calc_nrows_ncols(Ndatasets,Nfields)
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(6.4*ncols,4.8*nrows))
    if ncols*nrows==1:
        axs = [ax,]
    else:
        axs = ax.ravel()

    fig.subplots_adjust(wspace=0.4,hspace=0.4)

    # Loop over datasets, fields and times 
    for j, dfname in enumerate(datasets):
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


        df_pivot = df.pivot(columns='height',values=available_fields)

        for k, field in enumerate(fields):
            # Skip loop if field not available
            if not field in available_fields:
                continue

            axi = j*Nfields + k

            fieldvalues = df_pivot[field].values 
            im = axs[axi].pcolormesh(Ts,Zs,fieldvalues.T,
                          vmin=fieldlimits[field][0],vmax=fieldlimits[field][1],
                          cmap=cmap[field],
                          shading='flat')
            cbar = fig.colorbar(im,ax=axs[axi],shrink=1.0)
            try:
                cbar.set_label(fieldLabels[field])
            except KeyError:
                pass

            # axis mark up
            if len(datasets)>1:
                axs[axi].set_title(dfname)
            axs[axi].set_xlabel(r'UTC time')
            axs[axi].xaxis_date()
            axs[axi].xaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=12))
            axs[axi].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
            axs[axi].xaxis.set_major_locator(mdates.DayLocator())
            axs[axi].xaxis.set_major_formatter(mdates.DateFormatter('\n%d-%b'))

    if not timelimits is None:
        axs[-1].set_xlim(timelimits)
    if not heightlimits is None:
        axs[-1].set_ylim(heightlimits)

    # Add y labels
    for r in range(nrows): 
        axs[r*ncols].set_ylabel(r'Height [m]')
    
    # Number sub figures as a, b, c, ...
    if len(axs) > 1:
        for i,ax in enumerate(axs):
            ax.text(-0.14,-0.18,'('+chr(i+97)+')',transform=ax.transAxes,size=16)

    return fig, ax


def plot_timehistory_at_height(datasets,
                               fields,
                               height,
                               timelimits=None,
                               fieldlimits = {},
                               ):

    if isinstance(fields,str):
        fields = [fields,]
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}

    Nfields = len(fields)

    fig,axs = plt.subplots(nrows=Nfields,sharex=True,figsize=(12,3.0*Nfields))
    if Nfields==1: axs = [axs,]

    # Loop over datasets and fields 
    for j,dfname in enumerate(datasets):
        df = datasets[dfname]
        times = df.index.unique()
        heights = df['height'].unique()

        # Create list with available fields only
        available_fields = []
        for field in fields:
            if field in df.columns:
                available_fields.append(field)
        assert(len(available_fields)>0), 'Dataset '+dfname+' does not contain any of the requested fields'

        df_pivot = df.pivot(columns='height',values=available_fields)

        for i,field in enumerate(fields):
            # Skip loop if field not available
            if not field in available_fields:
                continue

            if j==0:
                axs[i].xaxis.grid(True,which='minor')
                axs[i].yaxis.grid()
                try:
                    axs[i].set_ylabel(fieldLabels[field])
                except KeyError:
                    pass
                try:
                    axs[i].set_ylim(fieldlimits[field])
                except KeyError:
                    pass

            signal = interp1d(heights,df_pivot[field].values,axis=1,fill_value="extrapolate")(height)
            axs[i].plot_date(times,signal,linewidth=2,label=dfname,linestyle='-',marker=None)
        
    
    axs[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=3))
    axs[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H%M'))
    axs[-1].xaxis.set_major_locator(mdates.DayLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('\n%Y-%m-%d'))
    axs[-1].set_xlabel(r'UTC time')

    if not timelimits is None:
        axs[-1].set_xlim(timelimits)

    # Number sub figures as a, b, c, ...
    if len(axs) > 1:
        for i,ax in enumerate(axs):
            ax.text(-0.14,1.0,'('+chr(i+97)+')',transform=ax.transAxes,size=16)

    # Add legend
    if len(datasets)>1:
        leg = axs[0].legend(loc='upper left',bbox_to_anchor=(1.05,1.0),fontsize=16)

    return fig, axs


def plot_timehistory_at_heights(datasets,
                                fields,
                                heights,
                                timelimits=None,
                                fieldlimits = {}
                                ):

    if isinstance(fields,str):
        fields = [fields,]
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}

    Nfields   = len(fields)
    Ndatasets = len(datasets)

    fig,axs = plt.subplots(nrows=Nfields*Ndatasets,sharex=True,figsize=(11,3*Nfields*Ndatasets))

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

        df_pivot = df.pivot(columns='height',values=available_fields)

        if Ndatasets > 1:
            ax[i*Nfields].set_title(dfname)

        for j,field in enumerate(fields):
            # Skip loop if field not available
            if not field in available_fields:
                continue

            axi = i*Nfields + j
            for z in heights:
                axs[axi].plot_date(timevalues,
                            interp1d(heightvalues,df_pivot[field].values,axis=1,fill_value="extrapolate")(z),
                            '-',label='z = {:.1f} m'.format(z),linewidth=2)
            axs[axi].grid(True,which='both')
            try:
                axs[axi].set_ylabel(fieldLabels[field])
            except KeyError:
                pass
            try:
                axs[axi].set_ylim(fieldlimits[field])
            except KeyError:
                pass

    axs[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=3))
    axs[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H%M'))
    axs[-1].xaxis.set_major_locator(mdates.DayLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('\n%Y-%m-%d'))
    axs[-1].set_xlabel(r'UTC time')

    if not timelimits is None:
        axs[-1].set_xlim(timelimits)

    # Number sub figures as a, b, c, ...
    if len(axs) > 1:
        for i,ax in enumerate(axs):
            ax.text(-0.14,1.0,'('+chr(i+97)+')',transform=ax.transAxes,size=16)

    # Add legend
    leg = axs[0].legend(loc='upper left',bbox_to_anchor=(1.05,1.0),fontsize=16)

    return fig, axs


def plot_profile(datasets,
                 fields,
                 times,
                 fieldlimits = {},
                 heightlimits=None,
                ):

    if isinstance(fields,str):
        fields = [fields,]
    if isinstance(times,str):
        times = [times,]
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}

    Ntimes = len(times)
    Nfields = len(fields)

    #Order plots depending on number of datasets and fields
    nrows, ncols = _calc_nrows_ncols(Ntimes,Nfields)
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharey=True,figsize=(4*ncols,5*nrows))
    if ncols*nrows==1:
        axs = [ax,]
    else:
        axs = ax.ravel()

    fig.subplots_adjust(wspace=0.2,hspace=0.4)

    # Loop over datasets, fields and times 
    for j, dfname in enumerate(datasets):
        df = datasets[dfname]
        heightvalues = df['height'].unique()

        # Create list with available fields only
        available_fields = []
        for field in fields:
            if field in df.columns:
                available_fields.append(field)
        assert(len(available_fields)>0), 'Dataset '+dfname+' does not contain any of the requested fields'


        df_pivot = df.pivot(columns='height',values=available_fields)

        for k, field in enumerate(fields):
            # Skip loop if field not available
            if not field in available_fields:
                continue

            for i, time in enumerate(times):
                axi = k*Ntimes + i
                
                # axes mark up
                if j==0:
                    axs[axi].grid(True,which='both')
                    # Set title if mutliple datasets are compared
                    if len(datasets)>1:
                        axs[axi].set_title(pd.to_datetime(time).strftime('%Y-%m-%d %H%M UTC'),fontsize=16)
                    try:
                        axs[axi].set_xlabel(fieldLabels[field])
                    except KeyError:
                        pass
                    try:
                        axs[axi].set_xlim(fieldlimits[field])
                    except KeyError:
                        pass
                
                # Plot data
                fieldvalues = df_pivot[field].loc[time].values
                axs[axi].plot(fieldvalues,heightvalues,linewidth=2,label=dfname)
    
    if not heightlimits is None:
        axs[0].set_ylim(heightlimits)

    # Add y labels
    for r in range(nrows): 
        axs[r*ncols].set_ylabel(r'Height [m]')
    
    # Number sub figures as a, b, c, ...
    if len(axs) > 1:
        for i,ax in enumerate(axs):
            ax.text(-0.14,-0.18,'('+chr(i+97)+')',transform=ax.transAxes,size=16)
    
    # Add legend
    if len(datasets)>1:
        leg = axs[ncols-1].legend(loc='upper left',bbox_to_anchor=(1.05,1.0),fontsize=16)

    return fig,axs


def plot_profile_evolution(datasets,
                           fields,
                           times,
                           fieldlimits = {},
                           heightlimits=None,
                           ):

    if isinstance(fields,str):
        fields = [fields,]
    if isinstance(times,str):
        times = [times,]
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}

    Ndatasets = len(datasets)
    Nfields = len(fields)

    #Order plots depending on number of datasets and fields
    nrows, ncols = _calc_nrows_ncols(Ndatasets,Nfields)
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharey=True,figsize=(4*ncols,5*nrows))
    if ncols*nrows==1:
        axs = [ax,]
    else:
        axs = ax.ravel()

    fig.subplots_adjust(wspace=0.2,hspace=0.4)

    # Loop over datasets, fields and times 
    for j, dfname in enumerate(datasets):
        df = datasets[dfname]
        heightvalues = df['height'].unique()

        # Create list with available fields only
        available_fields = []
        for field in fields:
            if field in df.columns:
                available_fields.append(field)
        assert(len(available_fields)>0), 'Dataset '+dfname+' does not contain any of the requested fields'


        df_pivot = df.pivot(columns='height',values=available_fields)

        for k, field in enumerate(fields):
            # Skip loop if field not available
            if not field in available_fields:
                continue
            axi = j*Nfields + k
            
            # axes mark up
            axs[axi].grid(True,which='both')
            if len(datasets)>1:
                axs[axi].set_title(dfname)
            try:
                axs[axi].set_xlabel(fieldLabels[field])
            except KeyError:
                pass
            try:
                axs[axi].set_xlim(fieldlimits[field])
            except KeyError:
                pass

            for time in times:
                # Plot data
                fieldvalues = df_pivot[field].loc[time].values
                axs[axi].plot(fieldvalues,heightvalues,linewidth=2,label=pd.to_datetime(time).strftime('%Y-%m-%d %H%M UTC'))
    
    if not heightlimits is None:
        axs[0].set_ylim(heightlimits)

    # Add y labels
    for r in range(nrows): 
        axs[r*ncols].set_ylabel(r'Height [m]')
    
    # Number sub figures as a, b, c, ...
    if len(axs) > 1:
        for i,ax in enumerate(axs):
            ax.text(-0.14,-0.18,'('+chr(i+97)+')',transform=ax.transAxes,size=16)
    
    # Add legend
    leg = axs[ncols-1].legend(loc='upper left',bbox_to_anchor=(1.05,1.0),fontsize=16)

    return fig,axs


def plot_spectrum(datasets,
                  times,
                  fields,
                  height,
                  Tperiod=3600.0,
                  Twindow=600.0,
                  freqlimits = None,
                  fieldlimits = {},
                  ):

    fieldLabels = {'u': r'$E_{uu}\;[\mathrm{m^2/s}]$',
                   'v': r'$E_{vv}\;[\mathrm{m^2/s}]$',
                   'w': r'$E_{ww}\;[\mathrm{m^2/s}]$',
                   'wspd': r'$E_{UU}\;[\mathrm{m^2/s}]$',
                   }


    if isinstance(fields,str):
        fields = [fields,]
    if isinstance(times,str):
        times = [times,]
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    if isinstance(fieldlimits, (list, tuple)):
        assert(len(fields)==1), 'Unclear to what field fieldlimits corresponds'
        fieldlimits = {fields[0]:fieldlimits}

    Ntimes = len(times)
    Nfields = len(fields)

    #Order plots depending on number of datasets and fields
    nrows, ncols = _calc_nrows_ncols(Ntimes,Nfields)
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(4*ncols,5*nrows))
    if ncols*nrows==1:
        axs = [ax,]
    else:
        axs = ax.ravel()

    fig.subplots_adjust(wspace=0.3,hspace=0.5)

    # Loop over datasets, fields and times 
    for j, dfname in enumerate(datasets):
        df = datasets[dfname]
        heightvalues = df['height'].unique()
        timevalues   = df.index.unique()
        dt = (timevalues[1]-timevalues[0]) / pd.Timedelta(1,unit='s')     #Sampling rate in seconds
        fielddata = df.pivot(columns='height',values=fields)

        for k, field in enumerate(fields):
            for i, tstart in enumerate(times):
                axi = k*Ntimes + i
                
                # axes mark up
                if j==0:
                    axs[axi].set_title(pd.to_datetime(tstart).strftime('%Y-%m-%d %H%M UTC'),fontsize=16)
                    axs[axi].set_xlabel('f [Hz]')

                # Plot data
                istart = np.where(timevalues==pd.to_datetime(tstart))[0][0]
                signal = interp1d(heightvalues,fielddata[field].interpolate(method='linear').values,axis=1,fill_value="extrapolate")(height)
                f, P = welch(signal[istart:istart+np.int(Tperiod/dt)],fs=1./dt,nperseg=np.int(Twindow/dt),
                            detrend='linear',window='hanning',scaling='density')

                axs[axi].loglog(f[1:],P[1:],label = dfname)
    
    for r in range(nrows):
        try:
            axs[r*ncols].set_ylabel(fieldLabels[fields[r]])
        except KeyError:
            pass
        try:
            axs[r*ncols].set_ylim(fieldlimits[fields[r]])
        except KeyError:
            pass

    if not freqlimits is None:
        axs[0].set_xlim(freqlimits)

    # Number sub figures as a, b, c, ...
    if len(axs) > 1:
        for i,ax in enumerate(axs):
            ax.text(-0.14,-0.18,'('+chr(i+97)+')',transform=ax.transAxes,size=16)

    # Add legend
    if len(datasets)>1:
        leg = axs[ncols-1].legend(loc='upper left',bbox_to_anchor=(1.05,1.0))

    return fig, axs


def _calc_nrows_ncols(N1,N2):
    if N1==1 or N2==1:
        ncols = max([N1,N2])
        if ncols > 3 and ncols%2 == 0:
            ncols = int(ncols/2)
            nrows = 2
        else:
            nrows=1
    elif N1<=3 and N2<=4:
        ncols = N1
        nrows = N2
    else:
        nrows = N1
        ncols = N2
        if ncols > 3 and ncols%2 == 0:
            ncols = int(ncols/2)
            nrows *= 2
    return nrows, ncols
