# Usage:
# python3 rf_training_and_inference.py --gcp_bucket 'eartshot-science-team' --gcp_folder_name 'deforestation_risk' --samples_folder_name 'Brazil_samples_csv_scale30_2000numPixels' --tiles_folder_name 'Brazil_Deforestation_Risk_inference_1degree_grid_scale30' --path_to_tiles_local '/Users/margauxmforstyhe/Desktop/Brazil_Deforestation_Risk_inference_1degree_grid_scale30' --csv_samples_file 'Brazil_samples_csv_scale30_2000numPixels_merged.csv' --rf_trees 100 --max_depth 2 --tiles_in_GCP False --run_inference False --feature_names 'aspect' 'brazil_agriculture' 'brazil_pasture' 'brazil_protected_areas' 'brazil_roads' 'brazil_surrounding_forest' 'elevation' 'forest_age' 'hillshade' 'population_density' 'slope' 'south_america_rivers' --response_variable 'Deforestation_risk_response_variable_brazil'

####### Imports ######
import pandas as pd
from osgeo import gdal, ogr, gdal_array
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
import datetime
from google.cloud import storage
import warnings
from tqdm import tqdm
import subprocess
import shutil
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from utils import upload_to_bucket, glob_blob_in_GCP
import argparse
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

################## Variables Definitions ##################
##### 1. Global Variables #####

parser = argparse.ArgumentParser()
parser.add_argument('--gcp_bucket', help="GCP bucket name", type=str, required=True, default="eartshot-science-team")
parser.add_argument('--gcp_folder_name', help="GCP folder name in bucket", type=str, required=True, default="deforestation_risk")
parser.add_argument('--samples_folder_name', help="Name of samples folder in folder in bucket", type=str, 
                    required=True, default="Brazil_samples_csv_scale30_2000numPixels")
parser.add_argument('--tiles_folder_name', help="Name of inference tiles folder in folder in bucket", type=str, 
                    required=True, default="Brazil_Deforestation_Risk_inference_1degree_grid_scale30")
parser.add_argument('--path_to_tiles_local', help="Local path to inference tiles", type=str, 
                    required=False, default='/Users/margauxmforstyhe/Desktop/Brazil_Deforestation_Risk_inference_1degree_grid_scale30')
parser.add_argument('--csv_samples_file', help="Name of samples csv file", type=str, 
                    required=True, default='Brazil_samples_csv_scale30_2000numPixels_merged.csv')
parser.add_argument('--rf_trees', help="Number of trees in RF model", type=int, required=True, default=100)
parser.add_argument('--max_depth', help="max_depth of trees in RF model", type=int, required=True, default=4)
parser.add_argument('--random_state', help="random_state in RF model", type=int, required=False, default=None)
parser.add_argument('--n_cores', help="n_cores to run RF model", type=int, required=False, default=-1)
parser.add_argument('--tiles_in_GCP', help="If running with local tiles: tiles_in_GCP = False", type=bool, required=False, default=True)
parser.add_argument('--run_inference', help="If True: Training + Inference, if False: only running training", type=bool, required=False, default=True)
parser.add_argument('--use_test_val_buffered_sets', help="If True, uses the buffered test and val sets exported with dataset_builder", 
                    type=bool, required=False, default=True)
parser.add_argument('--feature_names', help="Name of predictors bands", default=None, nargs="+", required=True)
parser.add_argument('--response_variable', help="Name of response variable band", type=str, 
                    default='Response_Variable_Brazil_Atlantic_Forest_0forest_1deforested', required=True)


args = parser.parse_args()
gcp_bucket = args.gcp_bucket 
gcp_folder_name = args.gcp_folder_name
samples_folder_name = args.samples_folder_name
name_output_csv_samples_merged_file = args.csv_samples_file
tiles_folder_name = args.tiles_folder_name
path_to_tiles_local = args.path_to_tiles_local

# Define RF parameters
est = args.rf_trees
max_depth = args.max_depth
random_state = args.random_state
# how many cores should be used?
# -1 -> all available cores
n_cores = args.n_cores

# If running with local tiles: tiles_in_GCP = False
tiles_in_GCP = args.tiles_in_GCP
# If True: Training + Inference, if False: only running training
run_inference = args.run_inference
# If True, uses the buffered test and val sets exported with dataset_builder
use_test_val_buffered_sets = args.use_test_val_buffered_sets

##### 2. Variables for: Random Forest training #####
feature_names = args.feature_names
# ['aspect', 'brazil_agriculture', 'brazil_pasture', 'brazil_protected_areas', 'brazil_roads',
#                  'brazil_surrounding_forest', 'elevation', 'forest_age', 'hillshade', 'population_density',
#                  'slope', 'south_america_rivers']
print(f'There are {len(feature_names)} feature names')

label = [args.response_variable]  

print(f'\n\ngcp_bucket: {gcp_bucket}, \ngcp_folder_name: {gcp_folder_name}, \nsamples_folder_name: {samples_folder_name}, \nname_output_csv_samples_merged_file: {name_output_csv_samples_merged_file}, \ntiles_folder_name: {tiles_folder_name}, \npath_to_tiles_local: {path_to_tiles_local}, \nRF trees: {est}, \nmax_depth: {max_depth}, \nrandom_state: {random_state}, \nn_cores: {n_cores}, \ntiles_in_GCP: {tiles_in_GCP}, \nrun_inference: {run_inference}, \nuse_test_val_buffered_sets: {use_test_val_buffered_sets}, \nfeature_names: {feature_names}, \nlabel: {label}\n\n')

################## Functions ##################
def run_rf_inference_on_tile(tile, rf_regressor, tile_as_array):
    """
    Run Random Forest inference on all pixels of an array tile.

    Parameters
    ----------
    - rf_regressor: sklearn random forest pre-trained model
    - gcp_bucket: (string) name of the GCP bucket
    -------

    Return
    ----------
    - class_prediction: (numpy array)of the predictions
    -------
    """
    # first prediction will be tried on the entire image
    # if not enough RAM, the dataset will be sliced
    try:
        class_prediction = rf_regressor.predict(tile_as_array)
    except MemoryError:
        slices = int(round(len(tile_as_array) / 2))
        test = True
        while test == True:
            try:
                class_preds = list()

                temp = rf_regressor.predict(tile_as_array[0:slices + 1, :])
                class_preds.append(temp)

                for i in range(slices, len(tile_as_array), slices):
                    print('{} %, derzeit: {}'.format((i * 100) / (len(tile_as_array)), i))
                    temp = rf_regressor.predict(tile_as_array[i + 1:i + (slices + 1), :])
                    class_preds.append(temp)

            except MemoryError as error:
                slices = slices / 2
                print('Not enough RAM, new slices = {}'.format(slices))
            else:
                test = False
    else:
        print('Class prediction was successful without slicing!')

    # Concatenate all slices and re-shape it to the original extend
    try:
        class_prediction = np.concatenate(class_preds, axis=0)
    except NameError:
        print('No slicing was necessary!')
    class_prediction = class_prediction.reshape(tile[:, :, 0].shape)
    print('Reshaped back to {}'.format(class_prediction.shape))

    return class_prediction


################## Random Forest Training ##################
url_csv_merged_file_bucket = f'gs://{gcp_bucket}/{gcp_folder_name}/{samples_folder_name}/{name_output_csv_samples_merged_file}'
print(f'Reading sample csv file: {url_csv_merged_file_bucket}...')
df = pd.read_csv(url_csv_merged_file_bucket)
print(f"We have {len(df)} samples")

if use_test_val_buffered_sets:
    df_train = df[(df['test_set_10km_buffer'] == 0) & (df['val_set_10km_buffer'] == 0)]
    df_test = df[df['test_set_no_buffer'] == 1]
    df_val = df[df['val_set_no_buffer'] == 1]

    # get the features and labels into separate variables
    X_train = df_train[feature_names]
    y_train = df_train[label]
    X_test = df_test[feature_names]
    y_test = df_test[label]
    X_val = df_val[feature_names]
    y_val = df_val[label]
else:
    # Split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(df[feature_names], df[label], test_size=0.20, random_state=42)
    X_val = []
    y_val = []

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Validation samples: {len(X_val)}")

RF_output_folder_name = f'RF_{est}trees_max_depth_{max_depth}_test_val_train_sets_pred_tiles_south_am_inference_tiles_500km_scale300'

### Training ###
rf_regressor = RandomForestRegressor(n_estimators=est, oob_score=True, verbose=1, n_jobs=n_cores, max_depth=max_depth,
                                     bootstrap=True, random_state=random_state)
X_train = np.nan_to_num(X_train)
rf_regressor = rf_regressor.fit(X_train, y_train)

### Evaluation ###
print("\nEvaluation on test set...")
pred_test = rf_regressor.predict(X_test)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, pred_test))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, pred_test))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))
# Check out the "Out-of-Bag" (OOB) prediction score:
print('OOB prediction of accuracy is: {oob}%\n'.format(oob=rf_regressor.oob_score_ * 100))
print("R2:", metrics.r2_score(y_test, pred_test))

plt.figure(figsize=(20, 20))
plt.plot(list(range(0, int(y_test.max()))), ls='dashed', alpha=0.3)
plt.scatter(y_test, pred_test, color='black')
plt.title("Scatter plot of the Latin America model's performance predicting potential mature forest AGB")
plt.xlabel('Test AGB (tCO2)')
plt.ylabel('Predicted AGB (tCO2)')
plt.savefig(f'scatter_plot_test_set_{len(feature_names)}features_{est}trees_{max_depth}max_depth_{random_state}random_state.png')

feature_imp = pd.DataFrame({'feature_name': feature_names,
                            'importance': rf_regressor.feature_importances_}).sort_values('importance', ascending=False)
feature_imp.to_csv(f'features_importances_{len(feature_names)}features_{est}trees_{max_depth}max_depth_{random_state}random_state.csv')

fig, ax = plt.subplots(figsize=(20, 20))
sns.barplot(x=feature_imp.importance, y=feature_imp.feature_name, ax=ax)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features", pad=15, size=14)
plt.savefig(f'features_importance_{len(feature_names)}features_{est}trees_{max_depth}max_depth_{random_state}random_state.png')

### Evaluation ###
if use_test_val_buffered_sets:
    print("\nEvaluation on val set...")
    pred_val = rf_regressor.predict(X_val)
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_val, pred_val))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_val, pred_val))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_val, pred_val)))
    print("Val R2:", metrics.r2_score(y_val, pred_val))

    # Check out the "Out-of-Bag" (OOB) prediction score:
    print('OOB prediction of accuracy is: {oob}%\n'.format(oob=rf_regressor.oob_score_ * 100))


################## Random Forest Inference ##################
# Saving locally first in temp folder and then uploading to bucket
# See: https://rasterio.groups.io/g/main/topic/can_t_write_to_s3/87723847?p=,,,20,0,0,0::recentpostdate/sticky,,,20,2,0,87723847,previd=1645393360902016547,nextid=1634264224576831009&previd=1645393360902016547&nextid=1634264224576831009
if run_inference:
    RF_output_folder_temp = f'RF_outputs_temp_{samples_folder_name}'
    if not os.path.exists(RF_output_folder_temp):
        os.makedirs(RF_output_folder_temp)

    gcp_folder_path_inference_tiles = gcp_folder_name + '/' + tiles_folder_name
    # Find names of the predictors tiles in the GCP bucket and store them in list predictors_tiffiles
    if tiles_in_GCP:
        predictors_tiffiles = glob_blob_in_GCP(gcp_bucket=gcp_bucket,
                                               gcp_folder_name=gcp_folder_path_inference_tiles,
                                               extension='.tif')
    else:
        predictors_tiffiles = glob.glob(path_to_tiles_local + '/*.tif')
    print(f"There are {len(predictors_tiffiles)} inference tiles")

    i = 0
    # Loop through all predictors tiles and run inference on them
    # Results are saved locally, then uploaded to GCP, and deleted locally afterwards
    for path in predictors_tiffiles:
        print(f'\nStarting image: path')
        # Read image
        img_ds = gdal.Open(predictors_tiffiles[i], gdal.GA_ReadOnly)
        print('Image opened')
        bands = [img_ds.GetRasterBand(i).GetDescription() for i in range(1, img_ds.RasterCount + 1)]

        # Initialize tile that will only have the predictors features bands
        tile = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, len(feature_names)),
                        gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
        # Looping through the features used in the training and adding them as bands of {tile} in the same order as what
        # was used during the training
        for b in range(len(feature_names)):
            corresponding_raster_band_index = bands.index(feature_names[b])
            tile[:, :, b] = img_ds.GetRasterBand(corresponding_raster_band_index + 1).ReadAsArray()
        print(f"tile shape: {tile.shape}")

        # Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
        new_shape = (tile.shape[0] * tile.shape[1], tile.shape[2])
        tile_as_array = tile[:, :, :np.int(tile.shape[2])].reshape(new_shape)
        print('Reshaped from {o} to {n}'.format(o=tile.shape, n=tile_as_array.shape))
        tile_as_array = np.nan_to_num(tile_as_array)
        print(tile_as_array.shape)

        # Predict for each pixel
        class_prediction = run_rf_inference_on_tile(tile=tile, rf_regressor=rf_regressor,
                                                    tile_as_array=tile_as_array)

        # Generate mask from first band of our predictors
        mask = np.copy(tile[:, :, feature_names.index('forest_age')]).astype(
            np.uint8)  # using the BIOME_NUM layer here to have positive values
        # mask = np.copy(tile[:, :, feature_names.index('BIOME_NUM')]).astype(
        #     np.uint8)  # using the BIOME_NUM layer here to have positive values
        print(np.unique(mask))
        mask[mask > 0] = 1  # all actual pixels have a value of 1.0

        # Mask predictions raster
        class_prediction.astype(np.float16)
        class_prediction_ = class_prediction * mask

        # Save predictions raster
        classification_image = f"{RF_output_folder_temp}/RF_output_{str(datetime.datetime.now()).split('.')[0].replace(' ', '-').replace(':', '_')}_{i}.tif"
        class_prediction_.astype(np.float16)
        print(class_prediction_.shape)
        cols = tile.shape[1]
        rows = tile.shape[0]
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(classification_image, cols, rows, 1, gdal.GDT_UInt16)
        outdata.SetGeoTransform(img_ds.GetGeoTransform())  ##sets same geotransform as input
        outdata.SetProjection(img_ds.GetProjection())  ##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(class_prediction_)
        outdata.GetRasterBand(1).SetNoDataValue(0)
        outdata.FlushCache()  ##saves to disk
        outdata = None
        band = None
        imgs_ds = None
        print('Image saved to: {}\n\n\n'.format(classification_image))

        i = i + 1
        if tiles_in_GCP:
            # Upload to bucket
            blob_path = upload_to_bucket(gcp_bucket=gcp_bucket,
                                         folder_name=gcp_folder_name + '/' + RF_output_folder_name,
                                         file_name=classification_image.split('/')[-1],
                                         file_local_path=classification_image)
            print(f"Image uploaded to GCP bucket: {blob_path}")
            # Delete image locally
            os.remove(classification_image)

    print(f"Done: {i} tiles predicted")

    ################## Merge all predictions raster tiles ##################
    output_merged_tif = RF_output_folder_temp + '/merged.tif'

    # Find names of the predictions raster tiles in the GCP bucket and store them in list paths_pred_rasters
    if tiles_in_GCP:
        paths_pred_rasters = glob_blob_in_GCP(gcp_bucket=gcp_bucket,
                                              gcp_folder_name=gcp_folder_name + '/' + RF_output_folder_name,
                                              extension='.tif')
    else:
        paths_pred_rasters = glob.glob(RF_output_folder_temp + '/*.tif')

    paths_pred_rasters.sort()
    print(f'There are {len(paths_pred_rasters)} prediction rasters to be merged.')

    # Loop through the predictions raster tiles
    for i in tqdm(range(len(paths_pred_rasters))):
        print(paths_pred_rasters[i])
        # if it's the first tile, we merge the two first tiles together
        if i == 0:
            print('python3', 'gdal_merge.py', f"-o", f"{output_merged_tif.replace('.tif', f'_{i}.tif')}", f"{paths_pred_rasters[i]}", f"{paths_pred_rasters[i+1]}", "-a_nodata", "0", "-n", "0")
            subprocess.run(['python3', 'gdal_merge.py', f"-o", f"{output_merged_tif.replace('.tif', f'_{i}.tif')}", f"{paths_pred_rasters[i]}", f"{paths_pred_rasters[i+1]}", "-a_nodata", "0", "-n", "0"])
        # then we merge the previously merged output with the next tile
        else:
            subprocess.run(['python3', 'gdal_merge.py', f"-o", f"{output_merged_tif.replace('.tif', f'_{i}.tif')}", f"{output_merged_tif.replace('.tif', f'_{i-1}.tif')}", f"{paths_pred_rasters[i]}", "-a_nodata", "0", "-n", "0"])
            # We remove the previous merged output -- no need to keep it
            os.remove(output_merged_tif.replace('.tif', f'_{i-1}.tif'))

    print('Done. Upload final merge tif to GCP bucket')
    # upload final merge tif to GCP bucket
    local_path_final_merged_file = output_merged_tif.replace('.tif', f'_{i}.tif')
    blob_path = upload_to_bucket(gcp_bucket=gcp_bucket,
                                 folder_name=gcp_folder_name + '/' + RF_output_folder_name,
                                 file_name=local_path_final_merged_file.split('/')[-1],
                                 file_local_path=local_path_final_merged_file)

    print(f'Done! {i} prediction rasters were merged to {blob_path}')

    if tiles_in_GCP:
        # Remove temp directory
        shutil.rmtree(RF_output_folder_temp)

print('Done!')

