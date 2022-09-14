####### Imports ######
import sys
# setting path
sys.path.append('..')

from dataset_builder import EEDatasetBuilder

import ee
import geemap
import pandas as pd
import os
import glob

from osgeo import gdal, ogr, gdal_array # I/O image data
import numpy as np # math and array handling
import matplotlib.pyplot as plt # plot figures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # classifier
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix  # calculating measures for accuracy assessment
from sklearn import metrics

import seaborn as sn
import pickle
import datetime
from tqdm import tqdm
import subprocess

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

from google.cloud import storage
import warnings
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

########## Dataset builder #############
run_export = False

ee_dataset_builder = EEDatasetBuilder()
ee_dataset_builder.filtered_biomass_layer_from_raster(
    biomass_raster='Spawn_AGB_tCO2e',
    filter_dict={'forest_non_forest': {'date_range': ['2010-01-01', '2010-12-31']},
                 'min_forest_age': {'age': 40},
                # 'very_low_density_rural': {'year': 2010},
                # 'forest_loss': {'year': 2010, 'distance': 5000},
                # 'forest_gain' : {'distance': 5000},
                # 'roads': {'distance': 5000},
                # 'fire': {'year': 2010}
    }
)
ee_dataset_builder.spatial_covariates(covariates=['ecoregion', 'terrain', 'bioclim', 'terraclimate'])
print(ee_dataset_builder.image.bandNames().getInfo())

######### Export samples CSV to Cloud Storage ###########
gcp_bucket = 'earthshot-science'  # /potential-mature-forest-biomass

if run_export:
    # Gridded world shapefile asset in GEE
    shp_asset_path = 'projects/ee-margauxmasson21-shapefiles/assets/latin_america_gridded_5000km2'

    # This will take quite some time
    ee_dataset_builder.samples_csv_export(shp_asset_path,
                                        name_gcp_bucket=gcp_bucket,
                                        folder_in_gcp_bucket='potential-mature-forest-biomass',
                                        numPixels=2000, scale=100)

# we need to wait until all the csv files are uploaded to the GCP bucket
# TODO: how can we check all the csv files we uploaded to GCP from GEE? Should we just split those codes?
if run_export:
    import time
    time.sleep(5*60)
client = storage.Client()
gcp_folder_path = 'potential-mature-forest-biomass'
name_output_csv_merged_file = 'latin_america_gridded_5000km2_training_set_mature_forest_biomass_AGB.csv'
url_csv_merged_file_bucket = ee_dataset_builder.merge_samples_csv(client, gcp_bucket, gcp_folder_path,
                                                                  name_output_csv_merged_file)

df_merged = pd.read_csv(url_csv_merged_file_bucket)
print(df_merged.head(5))
print('Done!')





