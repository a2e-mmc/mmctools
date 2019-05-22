##### Parse raw SWiFT TTU tower data by the hour for 24 hours...
#
import os
import sys
import numpy as np
from math import *
#
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def TTURawToMMC(path,startyear,startmonth,startday,outpath):
   #startyear="2013"
   #startmonth="11"
   #startday="09"
   dateStr = startyear+"-"+startmonth+"-"+startday
   print("dateStr = {:s}".format(dateStr))
   endyear = startyear
   endmonth = startmonth
   endday = startday
   starttimes=["00","01","02","03","04","05","06","07","08","09","10","11", \
               "12","13","14","15","16","17","18","19","20","21","22","23"]
   endtimes  =["01","02","03","04","05","06","07","08","09","10","11","12", \
               "13","14","15","16","17","18","19","20","21","22","23","00"]
   ftlevels=[3,8,13,33,55,155,245,382,519,656]
   ftlevel=np.array(ftlevels)
   z=0.3048*ftlevel
   secondsPerMinute = 60
   minutesPerFile = 60
   sampleRateRaw = 50
   sampleRateTarg = 1
   signalRawSamples = sampleRateRaw*secondsPerMinute*minutesPerFile
   signalTargSamples = sampleRateTarg*secondsPerMinute*minutesPerFile
   mean_period=1  #60== 1min avg
   #JAS 1108-1111, 4-day tilt-corrected
   reg_coefs = [[-0.02047518907375512, -0.011649366757767144, -0.005668625739156408],
                [-0.10467766095942455, -0.0010341521077107858, -0.03058912859410552],
                [0.01667915952069285, 0.005848208752347085, 0.006706449805682085],
                [0.03109302695107044, -0.0010297838685371808, 0.006173865882031913],
                [0.0652410809553023, 0.0009344061198990692, -0.0007300303260188616],
                [0.11644312251929698, -0.010642857833143332, 0.004278938434437668],
                [0.14503074198177446, -0.008678657118724405, 0.018625128308748237],
                [0.17950082482974902, -0.014857133643938156, 0.012520780908011505],
                [0.2464041495610511, -0.037471231121111864, 0.012055903868292708],
                [0.2770725270201596, -0.004728345168161448, 0.01770705056549219]]
   tilts = [[0.012954624102180665, 0.4528733026774165],
                [0.03059705313218443, 1.5370013668280538],
                [0.008897968050189109, -2.2879403125120583],
                [0.0062590775316462705, -1.405520595252335],
                [0.0011857730625258979, 2.4783726846562715],
                [0.011470318042450842, -0.38227054372399627],
                [0.020544967485295315, -1.1347455430302287],
                [0.01942702745609606, -0.7002672837032288],
                [0.0393425898140506, -0.31127834356150985],
                [0.01832543831154384, -1.3098530959146042]]
   #Declare a few lists for later use as indices, etc.
   tme=list()
   iu=[1,11,21,31,41,51,61,71,81,91]
   iv=list()
   iw=list()
   ius=list()
   ivc=list()
   iwd=list()
   its=list()
   it=list()
   ip=list()
   irh=list()

   for i in range(10):     #Define indices for each data feature over 10 levels of sonics
      iv.append(iu[i]+1)
      iw.append(iu[i]+2)
      ius.append(iu[i]+3)
      ivc.append(iu[i]+4)
      iwd.append(iu[i]+5)
      its.append(iu[i]+6)
      it.append(iu[i]+7)
      ip.append(iu[i]+8)
      irh.append(iu[i]+9)
  
   #declare and initialize to zero, named numpy arrays used here 
   u=np.zeros((10,signalRawSamples))
   v=np.zeros((10,signalRawSamples))
   w=np.zeros((10,signalRawSamples))
   us=np.zeros((10,signalRawSamples))
   vc=np.zeros((10,signalRawSamples))
   wd=np.zeros((10,signalRawSamples))
   ts=np.zeros((10,signalRawSamples))
   t=np.zeros((10,signalRawSamples))
   th=np.zeros((10,signalRawSamples))
   p=np.zeros((10,signalRawSamples))
   rh=np.zeros((10,signalRawSamples))
   tmpu_sonic=np.zeros((10,1))
   tmpv_sonic=np.zeros((10,1))
   um=np.zeros((10,signalTargSamples))
   vm=np.zeros((10,signalTargSamples))
   wm=np.zeros((10,signalTargSamples))
   usm=np.zeros((10,signalTargSamples))
   vcm=np.zeros((10,signalTargSamples))
   wdm=np.zeros((10,signalTargSamples))
   tsm=np.zeros((10,signalTargSamples))
   tm=np.zeros((10,signalTargSamples))
   thm=np.zeros((10,signalTargSamples))
   pm=np.zeros((10,signalTargSamples))
   rhm=np.zeros((10,signalTargSamples))
   uf=np.zeros((10,signalTargSamples))
   vf=np.zeros((10,signalTargSamples))
   wf=np.zeros((10,signalTargSamples))
   tf=np.zeros((10,signalTargSamples))
   thf=np.zeros((10,signalTargSamples))
   pf=np.zeros((10,signalTargSamples))
   auxf=np.zeros((10,signalTargSamples))
   tkem=np.zeros((10,signalTargSamples))
   tau11=np.zeros((10,signalTargSamples))
   tau12=np.zeros((10,signalTargSamples))
   tau13=np.zeros((10,signalTargSamples))
   tau22=np.zeros((10,signalTargSamples))
   tau23=np.zeros((10,signalTargSamples))
   tau33=np.zeros((10,signalTargSamples))
   hflux=np.zeros((10,signalTargSamples)) 

   #Open the output file
   outfilename = "TTU200m_{:s}_{:s}{:s}-1Hz.dat".format(startyear,startmonth,startday)
   outpathandname = '{:s}/{:s}'.format(outpath,outfilename)
   fout = open(outpathandname,'w')
   #Write the MMC file-header metadata
   fout.write("INSTITUTION:SNL\n")
   fout.write("   LOCATION:TTUTOWER\n")
   fout.write("   LATITUDE:   33.6105\n")   
   fout.write("  LONGITUDE: -102.0505\n")
   fout.write("   CODENAME:TOWER\n")
   fout.write("   CODETYPE:DATA \n")
   fout.write("   CASENAME:DIURNAL\n")
   fout.write("  BENCHMARK:CASE1\n")
   fout.write("     LEVELS:     10\n")
   ### For each hourly 50Hz file of TTU data...
   for starttime in starttimes[0:24]:
       if starttime == "23":
          endday='{:02d}'.format(int(startday)+1)
       endtime = endtimes[starttimes.index(starttime)]
       #filename = "TTU200m_{:s}_{:s}{:s}_{:s}00_{:s}{:s}_{:s}00.dat".format(startyear,startmonth,startday,starttime,\
       #                                                                     endmonth,endday,endtime) 
       filename = "tower.z01.00.{:s}{:s}{:s}.{:s}0000.ttu200m.dat".format(startyear,startmonth,startday,starttime)
       pathandname = '{:s}/{:s}'.format(path,filename)
       filelines = file_len(pathandname)
       f = open(pathandname,'r')
       head1 = f.readline()
       head2 = f.readline()
       head3 = f.readline()
       head4 = f.readline()
       head5 = f.readline()
       varnames = head5.split(",")
       ####Subsample or complete sample the raw data and store in named numpy arrays
       sampleStride=int(sampleRateRaw/sampleRateTarg)
       ufRaw=np.zeros((10,sampleStride))
       vfRaw=np.zeros((10,sampleStride))
       wfRaw=np.zeros((10,sampleStride))
       tfRaw=np.zeros((10,sampleStride))
       thfRaw=np.zeros((10,sampleStride))
       pfRaw=np.zeros((10,sampleStride))
       for k in range(filelines-5):  #For each line in the file
           aline = f.readline()
           varvalues = aline.split(",")
           tme.append(varvalues[0])
           #Note for each raw TTU u_zonal = vsonic, and v_meridional = -usonic
           tmpu_sonic = [ varvalues[l] for l in iv] 
           tmpv_sonic = [ varvalues[l] for l in iu]
           u[:,k] = tmpu_sonic
           v[:,k] = tmpv_sonic
           v[:,k] = -v[:,k]
           w[:,k] = [ varvalues[l] for l in iw]
           us[:,k] = [ varvalues[l] for l in ius]
           vc[:,k] = [ varvalues[l] for l in ivc]
           wd[:,k] = [ varvalues[l] for l in iwd]
           ts[:,k] = [ varvalues[l] for l in its]
           t[:,k] = [ varvalues[l] for l in it]
           p[:,k] = [ varvalues[l] for l in ip]
           rh[:,k] = [ varvalues[l] for l in irh]
       tStop=len(tme)
       tStrt=tStop-3600*sampleRateRaw
       print("tStrt,tStop = {:d},{:d}".format(tStrt,tStop))
       tmem=tme[tStrt:tStop:sampleStride]
       #A couple unit conversions on temperature(F->K) and pressure( 1 kPa to 10 mbars)
       t = (t - 32.)*5./9. + 273.15
       p = 10.*p
       R = 287.04
       cv = 718.0
       cp = R+cv
       R_cp = R/cp
       gamma = cp/cv
       p00 = 1.0e5 #(Pa)
       th = np.multiply(t,np.power(p00/(100.0*p),R_cp))
### As of 4_15_19 JAS added Branko form of tilt correction from EOL description
# Tilt correction
       uv=[]
       ut=[]
       vt=[]
       wt=[]
       for lvl in range(u.shape[0]):
          a = reg_coefs[lvl][0]
          b = reg_coefs[lvl][1]
          c = reg_coefs[lvl][2]
          tilt = tilts[lvl][0]
          tiltaz = tilts[lvl][1]
          #Wf = ( sin(tilt)*cos(tiltaz), sin(tilt)*sin(tiltaz), cos(tilt) )
          wf1 = sin(tilt)*cos(tiltaz)
          wf2 = sin(tilt)*sin(tiltaz)
          wf3 = cos(tilt)
          #U'f = ((cos(tilt), 0, -sin(tilt)*cos(tiltaz))
          uf1 = cos(tilt)
          uf2 = 0.
          uf3 = -sin(tilt)*cos(tiltaz)
          ufm = sqrt(uf1*uf1+uf2*uf2+uf3*uf3)
          uf1 = uf1/ufm
          uf2 = uf2/ufm
          uf3 = uf3/ufm
          #vf = wf x uf 
          vf1 = wf2 * uf3 - wf3 * uf2
          vf2 = wf3 * uf1 - wf1 * uf3
          vf3 = wf1 * uf2 - wf2 * uf1
          ug = uf1 * u[lvl,:] + uf2 * v[lvl,:] + uf3 * (w[lvl,:] - a)
          vg = vf1 * u[lvl,:] + vf2 * v[lvl,:] + vf3 * (w[lvl,:] - a)
          wg = wf1 * u[lvl,:] + wf2 * v[lvl,:] + wf3 * (w[lvl,:] - a)
          u[lvl,:] = ug
          v[lvl,:] = vg
          w[lvl,:] = wg
### END Tilt Correction
       i=0
       j=0
       subSampleByMean = False
       dummyval=np.array(-999.000).astype(np.float32) 
       for k in range(u.shape[1]):  #For each line in the file
           if(subSampleByMean):
            if (k%sampleStride == 0 and k > 0) or k == u.shape[1]-1:#Take the mean of the raw data over this sampleStride then compute fluctuations
              #print("k = {:d}: k-sampleStride = {:d}".foramt(k,k-sampleStride))
              #Compute the means
              um[:,i] = np.nanmean(u[:,k-sampleStride:k],axis=1)
              vm[:,i] = np.nanmean(v[:,k-sampleStride:k],axis=1)
              wm[:,i] = np.nanmean(w[:,k-sampleStride:k],axis=1)
              usm[:,i] = np.nanmean(us[:,k-sampleStride:k],axis=1)
              vcm[:,i] = np.nanmean(vc[:,k-sampleStride:k],axis=1)
              wdm[:,i] = np.nanmean(wd[:,k-sampleStride:k],axis=1)
              tsm[:,i] = np.nanmean(ts[:,k-sampleStride:k],axis=1)
              tm[:,i] = np.nanmean(t[:,k-sampleStride:k],axis=1)
              thm[:,i] = np.nanmean(th[:,k-sampleStride:k],axis=1)
              pm[:,i] = np.nanmean(p[:,k-sampleStride:k],axis=1)
              rhm[:,i] = np.nanmean(rh[:,k-sampleStride:k],axis=1)
              #print(i,um[:,i])        
              #print "       TIME:"+str(tme[k][10:19])+" um[:,i]="+str(um[:,i])+" vm[:,i]="+str(vm[:,i])
              #Compute the core variable fluctuations 
              for l in range(u.shape[0]):
                ufRaw[l,:] = np.subtract(u[l,k-sampleStride:k],um[l,i])
                vfRaw[l,:] = np.subtract(v[l,k-sampleStride:k],vm[l,i])
                wfRaw[l,:] = np.subtract(w[l,k-sampleStride:k],wm[l,i])
                tfRaw[l,:] = np.subtract(t[l,k-sampleStride:k],tm[l,i])
                thfRaw[l,:] = np.subtract(t[l,k-sampleStride:k],thm[l,i])
                pfRaw[l,:] = np.subtract(p[l,k-sampleStride:k],pm[l,i])
              #print "uff.shape : ",ufRaw.shape
              uf[:,i] = np.nanmean(ufRaw,axis=1)
              vf[:,i] = np.nanmean(vfRaw,axis=1)
              wf[:,i] = np.nanmean(wfRaw,axis=1)
              tf[:,i] = np.nanmean(tfRaw,axis=1)
              thf[:,i] = np.nanmean(thfRaw,axis=1)
              pf[:,i] = np.nanmean(pfRaw,axis=1)
              #Compute the auxilliary fluctuation products
              tkem[:,i] = np.nanmean(np.multiply(ufRaw,ufRaw)+np.multiply(vfRaw,vfRaw)+np.multiply(wfRaw,wfRaw),axis=1)
              tau11[:,i] = np.nanmean(np.multiply(ufRaw,ufRaw),axis=1)
              tau12[:,i] = np.nanmean(np.multiply(ufRaw,vfRaw),axis=1)
              tau13[:,i] = np.nanmean(np.multiply(ufRaw,wfRaw),axis=1)
              tau23[:,i] = np.nanmean(np.multiply(vfRaw,wfRaw),axis=1)
              tau22[:,i] = np.nanmean(np.multiply(vfRaw,vfRaw),axis=1)
              tau33[:,i] = np.nanmean(np.multiply(wfRaw,wfRaw),axis=1)
              hflux[:,i] = np.nanmean(np.multiply(thfRaw,wfRaw),axis=1)
              #increment the target sample counter and reset the subfilter counter
              i = i + 1
              j = 0
            else:
              j = j + 1
           else:  #subSampleByMean is False so just subSample
            if (k%sampleStride == 0 and k > 0) or k == u.shape[1]-1:#Take the mean of the raw data over this sampleStride then compute fluctuations
              #print("k = {:d}: k-sampleStride = {:d}".format(k,k-sampleStride))
              #set the ith instance to be this sample
              um[:,i] = u[:,k]
              vm[:,i] = v[:,k]
              wm[:,i] = w[:,k]
              usm[:,i] = us[:,k]
              vcm[:,i] = vc[:,k]
              wdm[:,i] = wd[:,k]
              tsm[:,i] = ts[:,k]
              tm[:,i] = t[:,k]
              thm[:,i] = th[:,k]
              pm[:,i] = p[:,k]
              rhm[:,i] = rh[:,k]
              #Compute the auxilliary fluctuation products
              tkem[:,i] = dummyval
              tau11[:,i] = dummyval
              tau12[:,i] = dummyval
              tau13[:,i] = dummyval
              tau23[:,i] = dummyval
              tau22[:,i] = dummyval
              tau33[:,i] = dummyval
              hflux[:,i] = dummyval
              i = i + 1
              j = 0
            else:
              j=j+1
       for i in range(um.shape[1]):
           fout.write("                   \n")
           fout.write("       DATE:"+dateStr+"\n")
           fout.write("       TIME:"+str(tmem[i][10:19])+"\n")
           fout.write("FRICTION VELOCITY [m/s] =    0.25607\n")
           fout.write("SURFACE ROUGHNESS [m]   =    0.10000\n")
           fout.write("SKIN TEMPERATURE [K]    = -999.00000\n")
           fout.write("SURFACE FLUX [Km/s]     = -999.00000\n")
           fout.write("             Z [m]           U [m/s]           V [m/s]           W [m/s]            TH [K]           P [mbar]    TKE [m^2/s^2]   TAU11 [m^2/s^2]   TAU12 [m^2/s^2]   TAU13 [m^2/s^2]   TAU22 [m^2/s^2]   TAU23 [m^2/s^2]   TAU33 [m^2/s^2]      HFLUX [Km/s]\n")
           for j in range(10):
               fout.write(("%18.3f"*4 + "%18.2f"*2 + "%18.3f"*1 + "%18.5f"*7+"\n")%(z[j],um[j,i],vm[j,i],wm[j,i],thm[j,i],pm[j,i], \
                                                                               tkem[j,i],tau11[j,i],tau12[j,i],tau13[j,i], \
                                                                               tau22[j,i],tau23[j,i],tau33[j,i],hflux[j,i]))
   fout.close()
   print("Done!")
