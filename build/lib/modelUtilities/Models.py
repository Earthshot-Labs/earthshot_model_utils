###############################################################################
###############################################################################
##
## Objective: Model functions for prediction carbon by age and other features
##
## Inputs: Formatted plot data with additional features attached
##
## Outputs: Predicted Biomass
##
###############################################################################
###############################################################################


def Load_KMZ(kmz_file_location, kmz_file_name):
  
  ## Packages
  from zipfile import ZipFile
  import geopandas as gpd
  import fiona
  gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
  
  ## Extract zipped KML
  kmz = ZipFile(kmz_file_location + kmz_file_name, 'r')
  kmz.extract('doc.kml', kmz_file_location)
  site = gpd.read_file(kmz_file_location + 'doc.kml', driver = "KML")
  
  return site

def Zonal_Stats_Raster(Raster_Layer, Site_KML, NODATA):
  
  ## Import Packages
  import rasterio
  from rasterstats import zonal_stats
  
  ## Load Raster
  Raster_Input = rasterio.open(Raster_Layer)
  
  ## Perform Zonal Stats to get summary of CWD raster where site is located
  array = Raster_Input.read(1)
  affine = Raster_Input.transform
  stats_list = zonal_stats(Site_KML, array, affine=affine, stats=['min', 'max', 'mean', 'median', 'majority'], nodata = NODATA, copy_properties = True)
  site_cwd = stats_list[0]['mean']
  
  return site_cwd

def Regression_CWD_logAge(Model_Location, CWD_Value, Initial_Biomass = False, Bio_Value = 0):
  
  ## Get packages
  import pandas as pd
  import numpy as np
  import statsmodels.api as sm

  ## Load Model
  m1 = sm.load(Model_Location)
  
  ## Create prediction set for model w/site CWD as input
  x_test = pd.DataFrame(range(1, 100, 1), columns = ['Age'])
  x_test['Project_Year'] = range(1,100,1)
  x_test['log_Age'] = np.log(x_test['Age'])
  x_test['CWD'] = CWD_Value
  
  ## Predict the Y's for each X given above
  predictions = m1.predict(x_test)
  x_test['AGB_Mg_ha'] = predictions

  ## Set negative values to 0
  x_test['AGB_Mg_ha'][x_test['AGB_Mg_ha'] < 0] = 0
  
  if Initial_Biomass == True:
  
    ## Find index where biomass map most closely matches predictions
    ## This is the age of the current forest
    a = x_test['AGB_Mg_ha'] - Bio_Value
    b = min(a[a > 0])
    i = a[a==b].index[0]
    
    ## However, biomass map was generated for 2010, so there are several
    ## years of growth since 
    lapsed_years = int(year_now) - 2010
    initial_age = lapsed_years + i
    
    ## Regenerate estimates for project
    x_test = pd.DataFrame(range(initial_age, 100+initial_age, 1), columns = ['Age'])
    x_test['Project_Year'] = range(1,101,1)
    x_test['log_Age'] = np.log(x_test['Age'])
    x_test['CWD'] = site_cwd
    
    ## Predict Line
    predictions = m1.predict(x_test)
    x_test['AGB_Mg_ha'] = predictions
  
  return x_test























