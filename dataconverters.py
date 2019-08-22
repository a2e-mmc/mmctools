import os
import sys
import datetime as dt
import numpy as np
import xarray as xr
import pickle

from mmctools.mmcdata import *


def convertMMCToPickle(pathbase,year,dataDir,pklDir):
    inpath = os.path.join(pathbase,year,dataDir)
    outpath = os.path.join(pathbase,year,pklDir)
    inDirContents = os.listdir(inpath)
    print("inpath: {:s}".format(inpath))
    print("--contains: {:d} files/directories".format(len(inDirContents)))    
    for item in inDirContents:
        print("\t{:s}".format(item))
    print("\n")
    outDirContents = os.listdir(outpath)
    inCnt = 0
    for fname in inDirContents:
        fpath = os.path.join(inpath,fname)
        if os.path.isdir(fpath):
            print("STOP! You may need to manually recurse subdirectory-- {:s}\n".format(fname))
        else:
            print("pkl-izing MMC file-- {:s}".format(fname))
            if (not fname.endswith('.dat')) and (not fname.endswith('.txt')):
                raise IOError('input file has neither .dat nor .txt extension. Bailing out!!\n')
            else:
                name,_ = os.path.splitext(fpath)
            if(name+".pkl" in outDirContents):
                print("MMC file-- "+fname+" already pkl-ized!\n")
            else:
                db = MMCData(pklfile=fpath)
                outfile = os.path.join(outpath,name+".pkl")
                outfptr = open(outfile,'wb')
                pickle.dump(db,outfptr)
                outfptr.close()
                print("MMC file-- "+fname+" now pkl-ized!\n")
    print("Done pkl-izing input files! ")
    print("outpath: {:s}".format(outpath))
    outDirContents = os.listdir(outpath)
    print("--contains: {:d} files/directories".format(len(outDirContents)))
    for item in outDirContents:
        print("\t{:s}".format(item))
    print("\n")

def convertMMCToXarrayNCDF(pathbase,year,dataDir,ncDir):
    inpath = os.path.join(pathbase,year,dataDir)
    outpath = os.path.join(pathbase,year,ncDir)
    inDirContents = os.listdir(inpath)
    print("inpath: {:s}".format(inpath))
    print("--contains: {:d} files/directories".format(len(inDirContents)))
    for item in inDirContents:
        print("\t{:s}".format(item))
    print("\n")
    outDirContents = os.listdir(outpath)
    inCnt = 0
    for fname in inDirContents:
        fpath = os.path.join(inpath,fname)
        if os.path.isdir(fpath):
            print("STOP! You may need to manually recurse subdirectory-- {:s}\n".format(fname))
        else:
            print("xarray/netcdf-izing MMC file-- {:s}".format(fname))
            if (not fname.endswith('.dat')) and (not fname.endswith('.txt')):
                raise IOError('input file has neither .dat nor .txt extension. Bailing out!!\n')
            else:
                name,fileExtension = os.path.splitext(fpath)
                print(fileExtension)
            if(name+".nc" in outDirContents):
                print("MMC file-- "+fname+" already xarray/netcdf-ized!\n")
            else:
                db = MMCData(asciifile=fpath)
                #Now make an xarrays object out of the db-dictionary/database
                xrDS = db.to_xarray()
                outfilename ="{:s}/{:s}.nc".format(outpath,fname.split(fileExtension)[0])
                print(outfilename)
                xrDS.to_netcdf(outfilename,unlimited_dims='Times',encoding={'datetime':{'units': 'seconds since 1970-01-01 00:00:00.0'}})
                print("MMC file-- "+fname+" now xarray'd!\n")
    print("Done xarray/netcdf-izing input files! ")
    print("outpath: {:s}".format(outpath))
    outDirContents = os.listdir(outpath)
    print("--contains: {:d} files/directories".format(len(outDirContents)))
    for item in outDirContents:
        print("\t{:s}".format(item))
    print("\n")

def dbToXarray(db,specified_date=None):
    """Convert db instance of MMCData class from mmcdata.py to xarray

    If specified_date is not None, then the Time array will use the
    specified date; otherwise, the date will be read from the input 
    database. This is used as a hack to handle LLNL's bogus dates and
    times since the WRF run was 'ideal'.
    """
    #Deal with the times by converting the metadata strings to datetime objects
    print(type(db))
    Times = np.ndarray(db.dataSetLength)
    for i in range(1,db.dataSetLength):
        date_string='{:s} {:s}'.format(db[i][0]['date'],db[i][0]['time']).strip()
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
    for i in range(1,db.dataSetLength-1):
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
