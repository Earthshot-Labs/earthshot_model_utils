###############################################################################
###############################################################################
##
## Objective: Get feature data for each lat/lon datapoint 
##
## Origin of datasets: https://docs.google.com/spreadsheets/d/1Dm5SAeQbrzrhpwVERnMqy04l8IY_50ayEx2fvsevezg/edit#gid=329806828
##
## Outputs: dataframe with feature data added
##
###############################################################################
###############################################################################


def Get_AnnualPPT(Latitude, Longitude, PPT_Folder):
    
  ## Pacakges
  import glob
  import rasterio
  
  ## Get list of all precip files in the folder
  pptfiles = glob.glob(PPT_Folder + "*.tif")
  
  ## Loop over all the precip files and add them together
  ## to get total annual precipitation 
  count = 1
  for f in pptfiles:
    
    print(f)
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


def Get_CWD(Latitude, Longitude, CWD_Location):
  
  ## Spatial
  import rasterio 
  
  ## Matrix Manipulation
  import pandas as pd
  import numpy as np
  
  with rasterio.open(CWD_Location) as dataset:

    ## Get Data located at those coordinates 
    x,y = Longitude, Latitude
    row,col = dataset.index(x,y)
    vals = dataset.read(1)[row,col]
    
    ## Create a pandas dataframe 
    output              = pd.DataFrame(vals, columns = ["CWD"])
    output["Latitude"]  = Latitude
    output["Longitude"] = Longitude
    
  return output


def Get_Biome(Latitude, Longitude, Bio_Location, BioName):
  
  ## Get packages
  from geopandas import gpd
  
  ## Set up output dataframe 
  output              = pd.DataFrame(Latitude, columns = ["Latitude"])
  output["Longitude"] = Longitude
    
  pnts = gpd.points_from_xy(Longitude, Latitude)
  eco = gpd.read_file(ecoreg_layer)
  
  gdf = gpd.GeoDataFrame(output, geometry=gpd.points_from_xy(output.Longitude, output.Latitude), crs = "EPSG:4326")
  eco = eco.to_crs("EPSG:4326")
  
  gpd.sjoin(eco, gdf)


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


  
  
