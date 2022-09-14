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
    filter_dict={'forest_non_forest': {'date_range':['2010-01-01', '2010-12-31']},
                 'min_forest_age': {'age':40},
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

client = storage.Client()
list_csv_files = []
for blob in client.list_blobs(gcp_bucket, prefix='potential-mature-forest-biomass/'):
    # print(str(blob))
    file = str(blob).split('/')[-1].split(',')[0]
    if '.csv' in file:
        list_csv_files.append(f'gs://{gcp_bucket}/potential-mature-forest-biomass/{file}')

print(list_csv_files)
df = pd.read_csv(list_csv_files[-1])
print(df.head(2))

name_output_csv_merged_file = 'latin_america_gridded_5000km2_training_set_mature_forest_biomass_AGB.csv'
list_csv_files.sort()
print(f"There are {len(list_csv_files)} csv file to be merged.")

# merge files
dataFrame = pd.concat(map(pd.read_csv, list_csv_files), ignore_index=True)
pd.DataFrame.to_csv(dataFrame, os.path.join('.', name_output_csv_merged_file), index=False)

def upload_to_bucket(bucket_name, blob_path, local_path):
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

local_path = os.path.join('.', name_output_csv_merged_file) #local file path
gcp_folder_path = 'potential-mature-forest-biomass'
blob_path = f"{gcp_folder_path}/{local_path.split('/')[-1]}"
print(blob_path)
upload_to_bucket(gcp_bucket, blob_path, local_path)

df_merged = pd.read_csv(f'gs://{gcp_bucket}/{blob_path}')
print(df_merged.head(5))
print('Done!')





