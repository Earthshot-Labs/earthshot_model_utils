###############################################################################
###############################################################################
##
## Objective: Workflow  demonstrating how model prediction functions work together
##
## Inputs: Formatted plot data with additional features attached
##
## Outputs: Predicted Biomass
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
od = wd + 'Code/science/Models/'

##---------------------------------------------------------------------------##
## FILE LOCATIONS
##---------------------------------------------------------------------------##

## Files 
site          = sd + 'Sites/Azuero Reforestation Corridor.kmz'
cwd_layer     = sd + 'CWD/CWD.bil/CWD.bil'
biomass_layer = sd + "Forest_AGB.tif"
model_input   = swd + 'Code/science/Models/LM_CWD_Age_Gamma.pickle'

##---------------------------------------------------------------------------##
## MODULES
##---------------------------------------------------------------------------##

## General
import os
import glob
from zipfile import ZipFile
from datetime import date

## Earthshot Model Utilities
import modelUtilities 
from modelUtilities import Models as md

## Plotting 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as lines

## Get todays date 
today = date.today()
year_now = today.strftime("%Y")

##---------------------------------------------------------------------------##
## Run Model 
##---------------------------------------------------------------------------##

## 1. Extract KML from KMZ
site = md.Load_KMZ(kmz_file_location= sd + "Sites/", kmz_file_name="Azuero Reforestation Corridor.kmz")

## 2. Average CWD values across the site polygon
cwd_value = md.Zonal_Stats_Raster(Raster_Layer = cwd_layer, Site_KML = site, NODATA = -999)

## 3. Optional: Get existing amount of Biomass (from 2010 map)
#bio_value = md.Zonal_Stats_Raster(Raster_Layer = biomass_layer, Site_KML = site, NODATA = -3.3999999521443642e+38)

## 4. Run CWD + log(Age) linear regression 
biomass_results = md.Regression_CWD_logAge(Model_Location = model_input, CWD_Value = cwd_value, Initial_Biomass = False, Bio_Value = 0)

## 5. Plot Results
mpl.rcParams['figure.dpi'] = 200 ## Change resolution
fig = biomass_results.plot.scatter(x='Age', y='AGB_Mg_ha')
fig.add_artist(lines.Line2D(biomass_results['Age'], biomass_results['AGB_Mg_ha'], color = "gray"))
plt.show()







