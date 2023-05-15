#================================================================#
# Description:
# This file is used to process satellite data products
# from across the study region (Benguela Upwelling System; BUS)
# for the timeframe of individual cruises to the BUS
#================================================================#



#======================== Loading packages ======================#

import netCDF4
from netCDF4 import Dataset

#filename = "/Users/csi/private/Data_Scientist/Digethic/data/chl/AQUA_MODIS.20190215.L3b.DAY.CHL.nc"
filename = "/Volumes/LaCie/data/CHL/AQUA_MODIS.20190215.L3m.DAY.CHL.chlor_a.4km.nc"
chl = Dataset(filename)
print(chl)
print(chl.variables)



