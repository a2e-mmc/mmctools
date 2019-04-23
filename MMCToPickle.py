import os
import sys
import string
from math import *
import numpy as np
import pickle

##############
def readMMC_fileheader(f):
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

    fileheader={'lab':lab,'location':location,'latitude':latitude, \
                'longitude':longitude,'codename':codename,         \
                'codetype':codetype,'casename':casename,           \
                'benchmark':benchmark,'levels':levels }

    return fileheader;

##############
def readMMC_recordheader(f):
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

        recordheader={'date':date,'time':time,'ustar':ustar,'z0':z0,  \
                      'tskin':tskin,'hflux':hflux,'varnames':varnames, \
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
    return recordheader;

##############
def readMMC_records(f,levels):
    record=[]
    for i in range(levels):
        line = f.readline()
        #data = map(float,line.split())
        [record.append(data) for data in map(float,line.split())]
        #print("len(data) = {:d}",len(data))
        #record.append(data)
        #print("len(record) = {:d}",len(record))
    recordarray=np.array(record).reshape(levels,floor(len(record)/levels))
    #print("recordarray.shape = ",recordarray.shape)
    return recordarray;

##############
def readMMC_database(f):
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
    return database;

def convertMMCToPickle(pathbase,year,dataDir,pklDir):
    inpath ="{pb:s}/{yr:s}/{dDir:s}/".format(pb=pathbase,yr=year,dDir=dataDir)
    outpath ="{pb:s}/{yr:s}/{pDir:s}/".format(pb=pathbase,yr=year,pDir=pklDir)
    inDirContents = os.listdir(inpath)
    print("inpath: {:s}".format(inpath))
    print("--contains: {:d} files/directories".format(len(inDirContents)))    
    [print("\t{:s}".format(item)) for item in inDirContents]
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
    [print("\t{:s}".format(item)) for item in outDirContents]
    print("\n")

