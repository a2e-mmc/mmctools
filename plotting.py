"""
Library of standardized plotting functions for basic plot formats
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy.interpolate import interp1d

# TODO:
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
                    vlimits=None,
                    xlimits=None,
                    colorscheme=None,
                    ):

    if isinstance(fields,str):
        fields = [fields,]
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    if vlimits is None:
        vlimits = {}
        for field in fields:
            vlimits[field] = [ min([df[field].min() for df in datasets.values()]),
                               max([df[field].max() for df in datasets.values()]) ]

    if colorscheme == None:
        cmap = mpl.cm.get_cmap('viridis')
    else:
        cmap = mpl.cm.get_cmap(colorscheme)

    Ndatasets = len(datasets)
    Nfields = len(fields)

    #Order plots depending on number of datasets and fields
    if Ndatasets==1 or Nfields==1:
        ncols = max([Ndatasets,Nfields])
        if ncols > 3 and ncols%2 == 0:
            ncols = int(ncols/2)
            nrows = 2
        else:
            nrows=1
        fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(6.4*ncols,4.8*nrows))
        if ncols*nrows==1:
            axs = [ax,]
        else:
            axs = ax.ravel()
    elif Ndatasets<=3 and Nfields<=4:
        ncols = Ndatasets
        nrows = Nfields
        fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(6.4*ncols,4.8*nrows))
        axs = ax.ravel()
    else:
        nrows = Ndatasets
        ncols = Nfields
        if ncols > 3 and ncols%2 == 0:
            ncols = int(ncols/2)
            nrows *= 2
        fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(6.4*ncols,4.8*nrows))
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
                          vmin=vlimits[field][0],vmax=vlimits[field][1],
                          cmap=cmap,
                          shading='flat')
            cbar = fig.colorbar(im,ax=axs[axi],shrink=1.0)
            try:
                cbar.set_label(fieldLabels[field])
            except KeyError:
                pass

            # axis mark up
            axs[axi].set_title(dfname)
            axs[axi].set_xlabel(r'UTC time')
            axs[axi].xaxis_date()
            axs[axi].xaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=12))
            axs[axi].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
            axs[axi].xaxis.set_major_locator(mdates.DayLocator())
            axs[axi].xaxis.set_major_formatter(mdates.DateFormatter('\n%d-%b'))

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
                     xlimits=None,
                     ):

    if isinstance(fields,str):
        fields = [fields,]
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    Nfields = len(fields)

    fig,axs = plt.subplots(nrows=Nfields,sharex=True,figsize=(12,3.0*Nfields))
    if Nfields==1: axs = [axs,]

    # Loop over datasets and fields 
    for j,dfname in enumerate(datasets):
        df = datasets[dfname]
        times = df.index.unique().get_values()
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

            signal = interp1d(heights,df_pivot[field].values,axis=1,fill_value="extrapolate")(height)
            axs[i].plot_date(times,signal,linewidth=2,label=dfname,linestyle='-',marker=None)
        
    
    axs[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=range(24),interval=3))
    axs[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H%M'))
    axs[-1].xaxis.set_major_locator(mdates.DayLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('\n%Y-%m-%d'))
    axs[-1].set_xlabel(r'UTC time')
    if not xlimits is None:
        axs[-1].set_xlim(xlimits)

    # Number sub figures as a, b, c, ...
    if len(axs) > 1:
        for i,ax in enumerate(axs):
            ax.text(-0.14,1.0,'('+chr(i+97)+')',transform=ax.transAxes,size=16)

    # Add legend    
    leg = axs[0].legend(loc='upper left',bbox_to_anchor=(1.05,1.0),fontsize=16)

    return fig, axs


def plot_timehistory_at_heights():
    print('do something')
    return


def plot_profile(datasets,
                 fields,
                 times,
                 ylimits=None,
                ):

    if isinstance(fields,str):
        fields = [fields,]
    if isinstance(times,str):
        times = [times,]
    if isinstance(datasets,pd.DataFrame):
        datasets = {'Dataset': datasets}

    Ntimes = len(times)
    Nfields = len(fields)

    #Order plots depending on number of times and fields
    if Ntimes==1 or Nfields==1:
        ncols = max([Ntimes,Nfields])
        if ncols > 3 and ncols%2 == 0:
            ncols = int(ncols/2)
            nrows = 2
        else:
            nrows=1
        fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharey=True,figsize=(4*ncols,5*nrows))
        if ncols*nrows==1:
            axs = [ax,]
        else:
            axs = ax.ravel()
    elif Ntimes<=3 and Nfields<=4:
        ncols = Ntimes
        nrows = Nfields
        fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharey=True,figsize=(4*ncols,5*nrows))
        axs = ax.ravel()
    else:
        nrows = Ntimes
        ncols = Nfields
        if ncols > 3 and ncols%2 == 0:
            ncols = int(ncols/2)
            nrows *= 2
        fig,ax = plt.subplots(nrows=nrows,ncols=ncols,sharey=True,figsize=(4*ncols,5*nrows))
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
                    axs[axi].set_title(pd.to_datetime(time).strftime('%Y-%m-%d %H%M UTC'),fontsize=16)
                    axs[axi].grid(True,which='both')
                    try:
                        axs[axi].set_xlabel(fieldLabels[field])
                    except KeyError:
                        pass
                
                # Plot data
                fieldvalues = df_pivot[field].loc[time].values
                axs[axi].plot(fieldvalues,heightvalues,linewidth=2,label=dfname)
    
    if not ylimits is None:
        axs[0].set_ylim(ylimits)

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


def plot_profile_evolution():
    print('do something')
    return


def plot_spectrum():
    print('do something')
    return
