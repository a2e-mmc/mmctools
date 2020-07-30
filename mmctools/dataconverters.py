import os
import sys
import datetime as dt
import numpy as np
import xarray as xr
import pickle

from mmctools.mmcdata import MMCData


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

def convertMMCToXarrayNCDF(pathbase,year,dataDir,ncDir, **kwargs):
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
                db = MMCData(asciifile=fpath, **kwargs)
                #Now make an xarrays object out of the db-dictionary/database
                xrDS = db.to_xarray()
                outfilename ="{:s}/{:s}.nc".format(outpath,fname.split(fileExtension)[0])
                print(outfilename)
                xrDS.to_netcdf(outfilename,
                               unlimited_dims='Times',
                               encoding={
                                  'datetime':{'units': 'seconds since 1970-01-01 00:00:00.0'}
                               })
                print("MMC file-- "+fname+" now xarray'd!\n")
    print("Done xarray/netcdf-izing input files! ")
    print("outpath: {:s}".format(outpath))
    outDirContents = os.listdir(outpath)
    print("--contains: {:d} files/directories".format(len(outDirContents)))
    for item in outDirContents:
        print("\t{:s}".format(item))
    print("\n")

