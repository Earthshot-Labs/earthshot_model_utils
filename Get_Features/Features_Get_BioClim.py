###############################################################################
###############################################################################
## 
## 03/24/2022
##
## Written by: Meghan Blumstein (meghan@earthshot.eco) 
##
## Objective: Get feature data for each lat/lon datapoint from bioclim data 
##
## Inputs: Any bioclim variable (1-15+)
##
## Origin of datasets: https://docs.google.com/spreadsheets/d/1Dm5SAeQbrzrhpwVERnMqy04l8IY_50ayEx2fvsevezg/edit#gid=329806828
##     
## Outputs: dataframe with feature data added
##
###############################################################################
###############################################################################

def Get_BioClim(Latitude, Longitude, Bio_Location, BioName):
  
  ## Spatial
  import rasterio 
  
  ## Matrix Manipulation
  import pandas as pd
  import numpy as np
  
  with rasterio.open(Bio_Location) as dataset:

    ## Get Data located at those coordinates 
    x,y = Longitude, Latitude
    row,col = dataset.index(x,y)
    vals = dataset.read(1)[row,col]
    
    ## Create a pandas dataframe 
    output              = pd.DataFrame(vals, columns = [BioName])
    output["Latitude"]  = Latitude
    output["Longitude"] = Longitude
    
  return output




