###############################################################################
###############################################################################
## 
## 03/24/2022
##
## Written by: Meghan Blumstein (meghan@earthshot.eco) 
##
## Objective: Get ecoregions by lat/lon 
##
## Inputs: lat/lon, location of ecoregion layer (World Wildlife Fund)
##
## Origin of datasets: https://docs.google.com/spreadsheets/d/1Dm5SAeQbrzrhpwVERnMqy04l8IY_50ayEx2fvsevezg/edit#gid=329806828
##
## Outputs: dataframe with feature data added
##
###############################################################################
###############################################################################
Latitude = df['Latitude']
Longitude = df['Longitude']

sd = "/Users/meghs/Dropbox/Tropical_Forests_Data_Discovery/Spatial_Data/"
ecoreg_layer  = sd + "Terrestrial_Ecoregions/Terrestrial_Ecoregions.shp"

from geopandas import gpd
#gpd.options.use_pygeos = True

  ## Set up output dataframe 
output              = pd.DataFrame(Latitude, columns = ["Latitude"])
output["Longitude"] = Longitude
  
pnts = gpd.points_from_xy(Longitude, Latitude)
eco = gpd.read_file(ecoreg_layer)

gdf = gpd.GeoDataFrame(output, geometry=gpd.points_from_xy(output.Longitude, output.Latitude), crs = "EPSG:4326")
eco = eco.to_crs("EPSG:4326")

gdf.sjoin(eco)






pip uninstall rtree
sudo apt install libspatialindex-dev
pip install rtree






