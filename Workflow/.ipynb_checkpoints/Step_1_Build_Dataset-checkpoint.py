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

##---------------------------------------------------------------------------##
## USER INPUT [Data Paths]
##---------------------------------------------------------------------------##

## Dropbox directory = working directory
wd = "/Users/meghs/Google Drive/My Drive/Science/model_utilities_data/"
swd = "/Users/meghs/Dropbox/Tropical_Forests_Data_Discovery/"

## Data and figure folders
dd = wd + "AGB_Data/"
sd = swd + 'Spatial_Data/'

##---------------------------------------------------------------------------##
## MODULES
##---------------------------------------------------------------------------##

## General
import os
import glob
from zipfile import ZipFile
from datetime import date
import modelUtilities 
from modelUtilities import GetFeatures as gf
from modelUtilities import SetupData

## Get todays date 
today = date.today()
year_now = today.strftime("%Y")

##---------------------------------------------------------------------------##
## FILE LOCATIONS
##---------------------------------------------------------------------------##

## Files 
poorter       = dd + 'Poorter_2016_Data.csv'
cwd_layer     = sd + 'CWD/CWD.bil/CWD.bil'
biomass_layer = sd + "Forest_AGB.tif"
ecoreg_layer  = sd + "Terrestrial_Ecoregions/Terrestrial_Ecoregions.shp"
bio04_layer   = sd + "WorldClim/Bio/wc2.1_2.5m_bio_4.tif"
bio15_layer   = sd + "WorldClim/Bio/wc2.1_2.5m_bio_15.tif"
ppt_folder    = sd + "WorldClim/wc2.1_30s_prec/"
tmax_folder   = sd + "WorldClim/wc2.1_30s_tmax/" 
cec_layer     = sd + "T_CEC_CLAY.nc4"

##---------------------------------------------------------------------------##
## GET FEATURES
##---------------------------------------------------------------------------##

## Step 1. Format Poorter Data to pandas 
df = SetupData.Format_Poorter(poorter)

## Step 2. Get List of unique Location Information
Location_Info = df[['Chronosequence','Latitude','Longitude']]
Location_Info = Location_Info.drop_duplicates()
Location_Info = Location_Info.reset_index(drop = True)

## Step 2: Call Features from spatial data

  ## Step 2a. Annual Precipitation
  ppt = gf.Get_AnnualPPT(Location_DF = Location_Info, PPT_Folder = ppt_folder)
  
  ## Step 2b. BioClim Variables (rainfall & rainfall seasonality)
  bio04 = gf.Get_BioClim(Location_DF = Location_Info, Bio_Location=bio04_layer, BioName= "Bio04")
  bio15 = gf.Get_BioClim(Location_DF = Location_Info, Bio_Location=bio15_layer, BioName= "Bio15")
  
  ## Step 2c. Climatic Water Deficit
  CWD = gf.Get_CWD(Location_DF = Location_Info, CWD_Location=cwd_layer)
  
  ## Step 2d. Biome/Ecoregion Information: Currently has error in geopandas sjoin function
  #biome = gf.Get_Biome(Location_DF = Location_Info, Bio_Location, BioName)
  
  ## Step 2e. 
  tmax = gf.Get_MaxT(Location_DF = Location_Info, MaxT_Folder = tmax_folder)

  ## Step 2f. Soil cation exchange capacity - indicator of soil fertility
  cec = gf.Get_SoilCEC(Location_DF = Location_Info, SoilCEC_Location = cec_layer)


## Save to CSV File with data indicating versioning
## NB: CEC now has all the data together 
out_filename = dd + "AGB_Data_" + today.strftime("%m.%d.%Y") + ".csv"
cec.to_csv(out_filename, index = False)



