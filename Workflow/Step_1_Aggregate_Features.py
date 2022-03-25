###############################################################################
###############################################################################
## 
## 03/17/2022
##
## Written by: Meghan Blumstein (meghan@earthshot.eco) 
##
## Objective: Get feature data for each lat/lon datapoint from spatial files
##
## Inputs: chronic water deficit (cwd), elevation, climate, soil fertility, ecoregion
##
##      Origin of datasets: https://docs.google.com/spreadsheets/d/1Dm5SAeQbrzrhpwVERnMqy04l8IY_50ayEx2fvsevezg/edit#gid=329806828
##     
## Outputs: dataframe with feature data added
##
###############################################################################
###############################################################################

# General
import os
from zipfile import ZipFile
from datetime import date

## Spatial
import fiona
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
import geopandas as gpd
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
import xarray
import rioxarray

## Plotting 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as lines
import seaborn as sns

## Stats/Math
import statsmodels.api as sm
import pandas as pd
import numpy as np

## Get todays date 
today = date.today()
year_now = today.strftime("%Y")

##---------------------------------------------------------------------------##
## USER INPUT
##---------------------------------------------------------------------------##

## Dropbox directory = working directory
wd = "/Users/meghs/My Drive/Science/model_utilities_data/"
os.chdir(wd)

## Data and figure folders
dd = wd + "AGB_Data/"
sd = wd + 'Spatial_Data/'

## Files 
poorter       = dd + 'AGB_Data_12.29.2021.csv'
cwd_layer     = sd + 'CWD/CWD.bil/CWD.bil'
biomass_layer = sd + "Forest_AGB.tif"
ecoreg_layer  = sd + "Terrestrial_Ecoregions/Terrestrial_Ecoregions.shp"
bio04_layer   = sd + "WorldClim/Bio/wc2.1_2.5m_bio_4.tif"
bio15_layer   = sd + "WorldClim/Bio/wc2.1_2.5m_bio_15.tif"
ppt_folder    = sd + "WorldClim/wc2.1_30s_prec/"
tmax_folder   = sd + "WorldClim/wc2.1_30s_tmax/" 

cec_layer     = 'netcdf:' + sd + "T_CEC_CLAY.nc4" + ':T_CEC_CLAY'
cec_layer     = sd + "T_CEC_CLAY.nc4"

##---------------------------------------------------------------------------##
## Load Data
##---------------------------------------------------------------------------##

## Load Pooter data (with lat/lon coordinates)
df = pd.read_csv(poorter)

## Climate Water Deficit
cwd = rasterio.open(cwd_layer)

## Aboveground Biomass in 2010
bio  = rasterio.open(biomass_layer)

## Terrestrial Ecoregions (WWF designation)
eco = gpd.read_file(ecoreg_layer)

## Import BioClim Variables (represent different aspects of climate)
bio04 = rasterio.open(bio04_layer)
bio15 = rasterio.open(bio15_layer)

## Import Bioclim Temperature and Precipitation 


## Soil Cation Exchange [metric of soil fertility]
cec = rasterio.open(cec_layer)
rds = rioxarray.open_rasterio()

##---------------------------------------------------------------------------##
## Convert to same projection
##---------------------------------------------------------------------------##

## Put all projections in Lat/Lon
## WGS84 (EPSG: 4326)
crs_string = 'EPSG:4326'
crs        = rasterio.crs.CRS.from_string(crs_string)

with rasterio.open(cec_layer, mode='r+') as src:
    src.crs = crs

## Extract Values

# band1 = ds.read(1)
# >>> x, y = (dataset.bounds.left + 100000, dataset.bounds.top - 50000)
# >>> row, col = dataset.index(x, y)
# >>> row, col
# (1666, 3333)
# >>> band1[row, col]
# 7566

lat = df['Latitude']
lon = df['Longitude']
with rasterio.open(cwd_layer) as dataset:
  
  src_crs = 'EPSG:4326'
  
  ## Get Data located at those coordinates 
  x,y = lon, lat
  row,col = dataset.index(x,y)
  vals = dataset.read(1)[row,col]
  
  ## Create a pandas dataframe 
  output              = pd.DataFrame(vals, columns = ["CWD"])
  output["Latitude"]  = lat
  output["Longitude"] = lon




##---------------------------------------------------------------------------##
## Clean up Poorter et al. 2016 Data 
##---------------------------------------------------------------------------##

#---------------------------------------#
## Coerece age to numeric 
## [need to reclass "OG" codes]
## ** OG > 100 Years
#---------------------------------------#

## Start by dropping old growth as we are not using it in these models
df = df[df['Age']!= "OG"]
    
## Then coerce age a numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

#---------------------------------------#
## Create log column of AGB & Age 
#---------------------------------------#

## First set anything that is 0 to 1 to avoid log(0) == -inf
df.loc[df['AGB (Mg/ha)'] < 1, 'AGB (Mg/ha)'] = 0.01
df.loc[df['Age'] < 1, 'Age'] = 1
   
## Log column
df['log_AGB'] = np.log(df['AGB (Mg/ha)'])
df['log_Age'] = np.log(df['Age'])

#---------------------------------------#
## Remove Columbian Islands
#---------------------------------------#

## Remove the Columbian island as can't get predictors for it and it looks
## like Poorter et al. 2016 also dropped this data 
df = df[df['Chronosequence'] != 'Providencia Island']

#---------------------------------------#
## Rename columns to remove spaces & characters
## and sort by age
#---------------------------------------#

df = df.rename(columns={'AGB (Mg/ha)':'AGB_Mg_ha'})
df.sort_values(by=['Age'])

#---------------------------------------#
## sense check values 
#---------------------------------------#

# min(df['log_AGB'])
# max(df['log_AGB'])

##---------------------------------------------------------------------------##
## Clean up Poorter et al. 2016 Data 
##---------------------------------------------------------------------------##














