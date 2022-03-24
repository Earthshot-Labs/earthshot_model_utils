###############################################################################
###############################################################################
## 
## 03/24/2022
##
## Written by: Meghan Blumstein (meghan@earthshot.eco) 
##
## Objective: Get feature data for each lat/lon datapoint from Soil Cation Exchange Capacity 
##            Soil CEC is a metric of soil fertility/nutrient availability 
##
## Inputs: lat/lon, location of Soil CEC layer
##
## Origin of datasets: https://docs.google.com/spreadsheets/d/1Dm5SAeQbrzrhpwVERnMqy04l8IY_50ayEx2fvsevezg/edit#gid=329806828
##
## Outputs: dataframe with feature data added
##
###############################################################################
###############################################################################


def Get_SoilCEC(Latitude, Longitude, SoilCEC_Location):
  
  ## Spatial
  import netCDF4 as nc
  
  ## Matrix Manipulation
  import pandas as pd
  import numpy as np
  
  ## Open NetCDDF and print information
  ds = nc.Dataset(SoilCEC_Location, "r")
  cec = ds['T_CEC_CLAY']
  # print(ds)

  ## Loop through all lat/lon combinations and find the 
  out_list = []
  for i in range(len(Latitude)):
    
    print(i)
    
    lat_i = Latitude[i]
    lon_i = Longitude[i]
      
    ## Find the minimum distance between all lat/lon and current lat/lon of interest
    i = np.abs(ds.variables["lon"][:] - lon_i).argmin()
    j = np.abs(ds.variables["lat"][:] - lat_i).argmin()
    
    ## Get CEC Value based on lat/lon index
    cec_value = float(cec[j,i])
    
    ## Add value to list
    out_list.append(cec_value)
    
  ## Set up output dataframe 
  output              = pd.DataFrame(Latitude, columns = ["Latitude"])
  output["Longitude"] = Longitude
  output["Soil_CEC"]  = out_list
  
  return output


  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
