###############################################################################
###############################################################################
## 
## 03/24/2022
##
## Written by: Meghan Blumstein (meghan@earthshot.eco) 
##
## Objective: Get feature data for each lat/lon datapoint from precipitation
##
## Inputs: average precipitation by month (12 rasters in a folder)
##
## Origin of datasets: https://docs.google.com/spreadsheets/d/1Dm5SAeQbrzrhpwVERnMqy04l8IY_50ayEx2fvsevezg/edit#gid=329806828
## This is WorldClim 2.1 (January 2020) downloaded from http://worldclim.org
## They represent average monthly climate data for 1970-2000. 
##
## Outputs: dataframe with feature data added
##
###############################################################################
###############################################################################


def Features_Get_AnnualPPT(Latitude, Longitude, PPT_Folder):
    
  ## Pacakges
  import glob
  import rasterio
  
  ## Get list of all precip files in the folder
  pptfiles = glob.glob(PPT_Folder + "*.tif")
  
  ## Loop over all the precip files and add them together
  ## to get total annual precipitation 
  count = 1
  for f in pptfiles:
    
    if count == 1:
      rast_ini      = rasterio.open(f)
      output_raster = rast_ini.read(1)
    
    if count > 1: 
      rast_sec      = rasterio.open(f)
      sec_raster    = rast_sec.read(1)
      output_raster = output_raster + sec_raster
      
    count = count + 1
    
  ## Extract Precipitation values by lat/lon
  with rasterio.open(f) as dataset:
  
    ## Get Data located at those coordinates 
    x,y = Longitude, Latitude
    row,col = dataset.index(x,y)
    vals = output_raster[row,col]
    
    ## Create a pandas dataframe 
    output              = pd.DataFrame(vals, columns = ["Annual_PPT_mm"])
    output["Latitude"]  = Latitude
    output["Longitude"] = Longitude
    
  return output
  
