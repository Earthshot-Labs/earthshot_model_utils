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
poorter       = dd + 'AGB_Data_12.29.2021.csv'
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

## Step 2. Get List of Location Information
lat = df['Latitude']
lon = df['Longitude']


## Step 2: Call Features from spatial data

  ## Step 2a. Annual Precipitation
  ppt = gf.Get_AnnualPPT(Latitude = lat, Longitude = lon, PPT_Folder = ppt_folder)







