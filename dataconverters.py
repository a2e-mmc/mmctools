import os
import sys
import string
import datetime as dt
from netCDF4 import Dataset as ncdf
from netCDF4 import stringtochar, num2date, date2num
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from math import *
import numpy as np
import xarray as xr
import pickle

def convertMMCToPickle(pathbase,year,dataDir,pklDir):
    inpath ="{pb:s}/{yr:s}/{dDir:s}/".format(pb=pathbase,yr=year,dDir=dataDir)
    outpath ="{pb:s}/{yr:s}/{pDir:s}/".format(pb=pathbase,yr=year,pDir=pklDir)
    inDirContents = os.listdir(inpath)
    print("inpath: {:s}".format(inpath))
    print("--contains: {:d} files/directories".format(len(inDirContents)))    
    for item in inDirContents:
        print("\t{:s}".format(item))
    print("\n")
    outDirContents = os.listdir(outpath)
    inCnt = 0
    for i in inDirContents:
        if(not os.path.isfile(inpath+i)):
            print("STOP! You may need to manually recurse subdirectory-- {:s}\n".format(i))
        else:
            print("pkl-izing MMC file-- {:s}".format(i))
            if(i.find(".dat") == -1):
                if(i.find(".txt") == -1):
                    print("input file has neither .dat nor .txt extension. Bailing out!!\n")
                    exit()
                else:
                    fileExtension = ".txt"
            else:
                fileExtension = ".dat"
            if(i.split(fileExtension)[0]+".pkl" in outDirContents):
                print("MMC file-- "+i+" already pkl-ized!\n")
            else:
                fptr = open(inpath+i,'r')
                db = readMMC_database(fptr)
                fptr.close()
                outfilename = i.split(fileExtension)[0]+".pkl"
                outfptr = open(outpath+outfilename,'wb')
                pickle.dump(db,outfptr)
                outfptr.close()
                print("MMC file-- "+i+" now pkl-ized!\n")
    print("Done pkl-izing input files! ")
    print("outpath: {:s}".format(outpath))
    outDirContents = os.listdir(outpath)
    print("--contains: {:d} files/directories".format(len(outDirContents)))
    for item in outDirContents:
        print("\t{:s}".format(item))
    print("\n")

def convertMMCToXarrayNCDF(pathbase,year,dataDir,ncDir):
    inpath ="{pb:s}/{yr:s}/{dDir:s}/".format(pb=pathbase,yr=year,dDir=dataDir)
    outpath ="{pb:s}/{yr:s}/{pDir:s}/".format(pb=pathbase,yr=year,pDir=ncDir)
    inDirContents = os.listdir(inpath)
    print("inpath: {:s}".format(inpath))
    print("--contains: {:d} files/directories".format(len(inDirContents)))
    for item in inDirContents:
        print("\t{:s}".format(item))
    print("\n")
    outDirContents = os.listdir(outpath)
    inCnt = 0
    for i in inDirContents:
        if(not os.path.isfile(inpath+i)):
            print("STOP! You may need to manually recurse subdirectory-- {:s}\n".format(i))
        else:
            print("xarray/netcdf-izing MMC file-- {:s}".format(i))
            if(i.find(".dat") == -1):
                if(i.find(".txt") == -1):
                    print("input file has neither .dat nor .txt extension. Bailing out!!\n")
                    exit()
                else:
                    fileExtension = ".txt"
            else:
                fileExtension = ".dat"
            if(i.split(fileExtension)[0]+".nc" in outDirContents):
                print("MMC file-- "+i+" already xarray/netcdf-ized!\n")
            else:
                fptr = open(inpath+i,'r')
                db = readMMC_database(fptr)
                fptr.close()
                #Now make an xarrays object out of the db-dictionary/database
                xrDS = dbToXarray(db)
                outfilename ="{:s}{:s}.nc".format(outpath,i.split(fileExtension)[0])
                xrDS.to_netcdf(outfilename,unlimited_dims='Times')
                print("MMC file-- "+i+" now xarray'd!\n")
    print("Done xarray/netcdf-izing input files! ")
    print("outpath: {:s}".format(outpath))
    outDirContents = os.listdir(outpath)
    print("--contains: {:d} files/directories".format(len(outDirContents)))
    for item in outDirContents:
        print("\t{:s}".format(item))
    print("\n")

def dbToXarray(db,specified_date=None):
    """Convert db to xarray

    If specified_date is not None, then the Time array will use the
    specified date; otherwise, the date will be read from the input 
    database. This is used as a hack to handle LLNL's bogus dates and
    times since the WRF run was 'ideal'.
    """
    #Deal with the times by converting the metadata strings to datetime objects
    Times = np.ndarray([len(db)-1])
    for i in range(1,len(db)):
        #date_string='{:s} {:s}'.format(db[i][0]['date'],db[i][0]['time']).strip()
        if specified_date is None:
            time = dt.datetime.strptime(date_string,'%Y-%m-%d %H:%M:%S')
        else:
            date_string = '{:s} {:s} {:s}'.format(specified_date,db[i][0]['time'].strip(),'UTC').strip()
            #print("date_string = {:s}".format(date_string))
            time = dt.datetime.strptime(date_string,'%Y-%m-%d %H:%M:%S %Z')
        time = time.replace(tzinfo=dt.timezone.utc)
        Times[i-1] = time.timestamp()
        #print(time.strftime('%m-%d-%Y %H:%M:%S'))
    bigArray = np.ndarray([len(db[1][0]['varnames']),db[0]['levels'],len(db)-1]) #array(flds,levels,times)
    for i in range(1,len(db)-1):
        bigArray[:,:,i-1] = db[i][1].transpose()
    bigArray[bigArray==(-999)] = np.nan   #Convert any -999 labeled 'missing values' to np.nan
    attrs = {'units': 'seconds since 1970-01-01 00:00:00.0'}
    coords = {}
    coords['Times'] = ('Times',Times,attrs)
    coords['levels'] = np.arange(0,db[0]['levels']).astype('int32')    
    xrDS = xr.Dataset(coords=coords)  #Set the coordinates of the xarrays DataSet as (Times and levels)
    for i in range(len(db[1][0]['varnames'])): #Add each variable field to the xarray-Dataset
        xrDS[db[1][0]['varnames'][i]] = (('levels','Times'),bigArray[i,:,:])
    xrDS = xr.decode_cf(xrDS)  #Make sure the Times coordinate is of type  datetime64
    return xrDS
