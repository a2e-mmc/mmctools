import numpy as np
import pandas as pd


# names/units in LANDUSE.TBL
abbrev = {
    'ALBD': 'albedo',  # [%]
    'SLMO': 'soil_moisture_availability',  # [-]
    'SFEM': 'surface_emissivity',  # [-]
    'SFZ0': 'roughness_length',  # [cm]
    'THERIN': 'thermal_inertia',  # [100 cal cm^-2 K^-1 s^-1/2] == [4.184e2 J m^-2 K^-1 s^-1/2]?
    'SCFX': 'snow_cover_effect',  # [-]
    'SFHC': 'surface_heat_capacity',  # [J m^-3 K^-1]
}

# standard conversions
conversions = {
    'ALBD': 0.01,
    'SFZ0': 0.01,
}

class LandUseTable(dict):
    """Container for land-use information from WRF"""

    def __init__(self,fpath='LANDUSE.TBL'):
        """Read specified LANDUSE.TBL file"""
        with open(fpath,'r') as f:
            name = f.readline().strip()
            while not name == '':
                print('Reading',name)
                self.__setitem__(name, self._read_def(f))
                name = f.readline().strip()
                
    def _read_def(self,f):
        headerinfo = f.readline().split(',')
        Ndef = int(headerinfo[0])
        Nseason = int(headerinfo[1])
        header = ['index']
        header += headerinfo[2].strip().strip("'").split()
        header += ['description']
        newdict = dict()
        index = pd.RangeIndex(1,Ndef+1)
        for iseason in range(Nseason):
            season = f.readline().strip()
            newdict[season] = pd.DataFrame(index=index, columns=header[1:])
            for idef in range(Ndef):
                line = f.readline().split(',')
                if len(line) < len(header):
                    assert len(line) == len(header)-1, \
                            'No workaround for reading '+str(line)+'... abort'
                    # workaround for rows with missing comma after index
                    line = line[0].split() + line[1:]
                line[1:-1] = [float(val) for val in line[1:-1]]
                line[-1] = line[-1].strip().strip("'")
                idx = int(line[0]) 
                newdict[season].loc[idx] = line[1:]
            # do unit conversions
            for varn, fac in conversions.items():
                newdict[season][varn] *= fac
            # rename known abbreviations
            newdict[season].rename(columns=abbrev, inplace=True)
            #print(newdict[season])
        if Nseason == 1:
            # return dataframe as is
            return newdict[season]
        else:
            # return dataframe with multiindex
            for season in newdict.keys():
                newdict[season]['season'] = season
            df = pd.concat(newdict.values())
            df = df.set_index('season', append=True)
            return df.sort_index()

