###############################################################################
###############################################################################
## 
## 03/24/2022
##
## Written by: Meghan Blumstein (meghan@earthshot.eco) 
##
## Objective: Get feature data for each lat/lon datapoint from temperature
##
## Inputs: maximum temperature by month (12 rasters in a folder)
##
## Origin of datasets: https://docs.google.com/spreadsheets/d/1Dm5SAeQbrzrhpwVERnMqy04l8IY_50ayEx2fvsevezg/edit#gid=329806828
## This is WorldClim 2.1 (January 2020) downloaded from http://worldclim.org
## They represent average monthly climate data for 1970-2000. 
##
## Outputs: dataframe with feature data added
##
###############################################################################
###############################################################################


def Get_MaxT(Latitude, Longitude, MaxT_Folder):
    
  ## Pacakges
  import glob
  import calendar
  import rasterio
  
  ## Get list of all precip files in the folder
  tmaxfiles = glob.glob(MaxT_Folder + "*.tif")
  
  ## Loop over all the max temp files and find the max temp value
  ## for each location 
  ## Set up output dataframe 
  output              = pd.DataFrame(Latitude, columns = ["Latitude"])
  output["Longitude"] = Longitude
  
  count = 1
  for f in tmaxfiles:
    
    with rasterio.open(f) as dataset:
  
      ## Get Data located at those coordinates 
      x,y = Longitude, Latitude
      row,col = dataset.index(x,y)
      vals = dataset.read(1)[row,col]
        
      ## Create a pandas dataframe 
      colname         = "Tmax_" + list(calendar.month_abbr)[count]
      output[colname] = vals
    
    count = count + 1

 Annual_Max = output.iloc[:,2:14].max(axis = 1)
 output["Annual_MaxT"] = Annual_Max
 
 return output
