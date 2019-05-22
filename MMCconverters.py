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

##############
def readMMC_fileheader(f):
    """Read header from legacy MMC file"""
    head1 = f.readline()
    head2 = f.readline()
    head3 = f.readline()
    head4 = f.readline()
    head5 = f.readline()
    head6 = f.readline()
    head7 = f.readline()
    head8 = f.readline()
    head9 = f.readline()
    lab = head1[12:25].strip()
    print("lab: {:s}".format(lab))
    location = head2[12:25].strip()
    latitude = float(head3[12:25].strip())
    longitude = float(head4[12:25].strip())
    codename = head5[12:25].strip()
    print("codename: {:s}".format(codename))
    codetype = head6[12:25].strip()
    casename = head7[12:25].strip()
    benchmark = head8[12:25].strip()
    levels = int(head9[12:25].strip())
    print("levels: {:d}".format(levels))

    fileheader={'lab':lab,'location':location,'latitude':latitude,
                'longitude':longitude,'codename':codename,
                'codetype':codetype,'casename':casename,
                'benchmark':benchmark,'levels':levels }

    return fileheader

##############
def readMMC_recordheader(f):
    """Read a record from legacy MMC file"""
    try:
        head1 = f.readline()
        head2 = f.readline()
        head3 = f.readline()
        head4 = f.readline()
        head5 = f.readline()
        head6 = f.readline()
        head7 = f.readline()
        date  = head1[12:22]
        time  = head2[12:22]
        ustar = float(head3[26:36].strip())
        z0    = float(head4[26:36].strip())
        tskin = float(head5[26:36])
        hflux = float(head6[26:36])
        varlist = head7.split()

        varnames=[]
        varunits=[]

        for i in range(len(varlist)):
            if (i % 2) == 0:
                varnames.append(varlist[i])
            if (i % 2) == 1:
                varunits.append(varlist[i])

        recordheader={'date':date,'time':time,'ustar':ustar,'z0':z0,
                      'tskin':tskin,'hflux':hflux,'varnames':varnames,
                      'varunits':varunits}
    except:
        print("Error in readrecordheader... Check your datafile for bad records!!\n Lines read are")
        print("head1 = ",head1)
        print("head2 = ",head2)
        print("head3 = ",head3)
        print("head4 = ",head4)
        print("head5 = ",head5)
        print("head6 = ",head6)
        print("head7 = ",head7)
    return recordheader

##############
def readMMC_records(f,Nlevels):
    """Read specified number of records from legacy MMC file"""
    record=[]
    for i in range(Nlevels):
        line = f.readline()
        #data = map(float,line.split())
        for data in map(float,line.split()):
            record.append(data)
        #print("len(data) = {:d}",len(data))
        #record.append(data)
        #print("len(record) = {:d}",len(record))
    recordarray=np.array(record).reshape(Nlevels,floor(len(record)/Nlevels))
    #print("recordarray.shape = ",recordarray.shape)
    return recordarray

##############
def readMMC_database(f):
    """Read entire legacy MMC file"""
    fileheader=readMMC_fileheader(f);
    database = [fileheader]
    l=0
    while True:
        line=f.readline()
        if line == '':
            break
        l=l+1
        recordheader=readMMC_recordheader(f);
        recordarray = readMMC_records(f,fileheader['levels'])
        database.append([recordheader,recordarray])
    return database

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

def dbToXarray(db):
    #Deal with the times by converting the metadata strings to datetime objects
    Times = np.ndarray([len(db)-1])
    for i in range(1,len(db)):
       #date_string='{:s} {:s}'.format(db[i][0]['date'],db[i][0]['time']).strip()
       #time=dt.datetime.strptime(date_string,'%Y-%m-%d %H:%M:%S')
       ###Hack to handle LLNL's bogus dates & times since the WRF run was 'ideal'
       date_string = '{:s} {:s} {:s}'.format('2013-11-08',db[i][0]['time'].strip(),'UTC').strip()
       #print("date_string = {:s}".format(date_string))
       time = dt.datetime.strptime(date_string,'%Y-%m-%d %H:%M:%S %Z')
       #### END HACK LLNL IDEAL WRF
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
