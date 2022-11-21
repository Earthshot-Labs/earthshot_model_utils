# Usage:
# python3 rf_training_and_inference.py --gcp_bucket 'eartshot-science-team' --gcp_folder_name 'deforestation_risk' --samples_folder_name 'Brazil_samples_csv_scale30_2000numPixels' --tiles_folder_name 'Brazil_Deforestation_Risk_inference_2degrees_grid_scale30_with_spawn_as_base_raster' --path_to_tiles_local '/Users/margauxmforstyhe/Desktop/Brazil_Deforestation_Risk_inference_2degrees_grid_scale30_with_spawn_as_base_raster' --csv_samples_file 'Brazil_samples_csv_scale30_2000numPixels_val_test_set_10km_buffer.csv' --rf_trees 100 --max_depth 2 --run_inference --use_test_val_buffered_sets --feature_names 'aspect' 'brazil_agriculture' 'brazil_pasture' 'brazil_protected_areas' 'brazil_roads' 'brazil_surrounding_forest' 'elevation' 'forest_age' 'hillshade' 'population_density' 'slope' 'south_america_rivers' --response_variable 'Response_Variable_Brazil_Atlantic_Forest_0forest_1deforested'

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
from sklearn.model_selection import GridSearchCV
import argparse
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()



class ModelBuilder():

    def __init__(self):
        """
        A class for building, training and running inference of an sklearn Randon Forest spatial model.
        
        
        Parameters
        ----------
        - nb_trees: (int) number of trees for the Random Forest model
        - max_depth: (int) The maximum depth of the tree. If None, then nodes are expanded until all leaves 
        are pure or until all leaves contain less than min_samples_split samples.
        - random_state: (int) Controls both the randomness of the bootstrapping of the samples used when 
        building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features)
        - max_features: (string) The number of features to consider when looking for the best split (“sqrt”, “log2”, None)
        - n_cores: (int) The number of jobs to run in parallel.
        - oob_score: Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True
        - bootstrap: Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
        - criterion: (string) {“squared_error”, “absolute_error”, “poisson”} The function to measure the quality of a split. 
        ----------
        
        """
        # Initialize the model 
        self.model = None
        
        # Inialize parameters
        self.feature_names = []
        self.response_variable = []
        self.gcp_bucket = None
        self.gcp_folder_name = None
        
    def initialize_model(self, nb_trees, max_depth, random_state=42, max_features=1.0, n_cores=-1, 
                         oob_score=True, bootstrap=True, criterion='squared_error'):
        """
        Initialize the sklearn Randon Forest spatial model.
        
        Parameters
        ----------
        - nb_trees: (int) number of trees for the Random Forest model
        - max_depth: (int) The maximum depth of the tree. If None, then nodes are expanded until all leaves 
        are pure or until all leaves contain less than min_samples_split samples.
        - random_state: (int) Controls both the randomness of the bootstrapping of the samples used when 
        building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features)
        - max_features: (string) The number of features to consider when looking for the best split (“sqrt”, “log2”, None)
        - n_cores: (int) The number of jobs to run in parallel.
        - oob_score: Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True
        - bootstrap: Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
        - criterion: (string) {“squared_error”, “absolute_error”, “poisson”} The function to measure the quality of a split. 
        ----------
        
        """
        self.model = RandomForestRegressor(n_estimators=nb_trees, oob_score=oob_score, verbose=1, 
                                             n_jobs=n_cores, max_depth=max_depth, max_features=max_features,
                                             bootstrap=bootstrap, random_state=random_state, criterion=criterion)
        
        
    def run_rf_inference_on_tile(self, tile, tile_as_array):
        """
        Run Random Forest inference on all pixels of an array tile.

        Parameters
        ----------
        - tile
        - tile_as_array: (array)
        -------

        Return
        ----------
        - class_prediction: (numpy array)of the predictions
        -------
        """
        # first prediction will be tried on the entire image
        # if not enough RAM, the dataset will be sliced
        try:
            class_prediction = self.model.predict(tile_as_array)
        except MemoryError:
            slices = int(round(len(tile_as_array) / 2))
            test = True
            while test == True:
                try:
                    class_preds = list()

                    temp = self.model.predict(tile_as_array[0:slices + 1, :])
                    class_preds.append(temp)

                    for i in range(slices, len(tile_as_array), slices):
                        print('{} %, derzeit: {}'.format((i * 100) / (len(tile_as_array)), i))
                        temp = self.model.predict(tile_as_array[i + 1:i + (slices + 1), :])
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


    def create_dataset(self, response_variable, feature_names, gcp_bucket, gcp_folder_name, samples_folder_name, 
                       name_csv_samples_merged_file, use_test_val_buffered_sets, samples_csv_local=False,
                       name_test_buffer_column='test_set_10km_buffer', name_val_buffer_column='val_set_10km_buffer', 
                       name_test_no_buffer_column='test_set_no_buffer', name_val_no_buffer_column='val_set_no_buffer'):
        """
            Create the dataset with split train, test and val sets from a csv files that contains the exported samples.

            Parameters
            ----------
            - response_variable: (list of strings) list of response variables
            - feature_names:  (list of strings) list of features/predictors variables
            - gcp_bucket: (string) name of GCP bucket
            - gcp_gcp_folder_namebucket: (string) name of folder in GCP bucker
            - samples_folder_name: (string) name of samples folder
            - name_csv_samples_merged_file: (string) name of the csv file with all exported samples from GEE
            - use_test_val_buffered_sets: (boolean) If True, uses the buffered test and val sets exported with dataset_builder
            -------

            """
        self.feature_names = feature_names
        self.response_variable = response_variable
        self.gcp_bucket = gcp_bucket
        self.gcp_folder_name = gcp_folder_name
        if samples_csv_local:
            url_csv_merged_file_bucket = name_csv_samples_merged_file
        else:
            url_csv_merged_file_bucket = f'gs://{self.gcp_bucket}/{self.gcp_folder_name}/{samples_folder_name}/{name_csv_samples_merged_file}'
        print(f'Reading sample csv file: {url_csv_merged_file_bucket}...')
        df = pd.read_csv(url_csv_merged_file_bucket)
        print(df.head(2))
        print(f"We have {len(df)} samples")

        if use_test_val_buffered_sets:
            df_train = df[(df[name_test_buffer_column] == 0) & (df[name_val_buffer_column] == 0)]
            df_test = df[df[name_test_no_buffer_column] == 1]
            df_val = df[df[name_val_no_buffer_column] == 1]

            # get the features and response_variables into separate variables
            self.X_train = df_train[self.feature_names]
            self.y_train = df_train[self.response_variable]
            self.X_test = df_test[self.feature_names]
            self.y_test = df_test[self.response_variable]
            self.X_val = df_val[self.feature_names]
            self.y_val = df_val[self.response_variable]
        else:
            # Split train/test sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df[self.feature_names], df[self.response_variable], 
                                                                                    test_size=0.20, random_state=42)
            self.X_val = []
            self.y_val = []

        # self.X_train = np.nan_to_num(self.X_train)
        # self.y_train = np.nan_to_num(self.y_train)
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Validation samples: {len(self.X_val)}")
        
    def grid_search(self, n_estimators=[100, 200, 500], max_features=['sqrt', 'log2'], max_depth=[4,5,6,7,8], 
                    criterion=['squared_error', 'absolute_error'], random_state=42):
        """
            Run Grid Search to find best hyper parameters

        Parameters
        ----------
        - random_state:
        - n_estimators: (list) list of number of trees to test 
        - max_features: (list) list of max_features to test 
        - max_depth: (list) list of max_depth to test 
        - criterion: (list) list of criterion to test 
        -------

        Returns
        ----------
        - GSCV.best_params_: best parameters to use for training
        -------
        """
        random_forest_tuning = RandomForestRegressor(random_state=random_state)  
        param_grid = { 
                'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth' : max_depth, 
                'criterion' : criterion
            }
        GSCV = GridSearchCV(estimator=random_forest_tuning, param_grid=param_grid, cv=5, verbose=1)
        GSCV.fit(self.X_val, self.y_val.values.ravel())
        print(GSCV.best_params_)
        return GSCV.best_params_
        

    def train(self):
        """
            Run Random Forest training
        -------

        """
        self.model = self.model.fit(self.X_train, self.y_train.values.ravel())

    def evaluate(self, X_test, y_test, save_figures=True, saving_base_output_name=''):
        """
            Run Random Forest evaluation

        Parameters
        ----------
        - X_test, y_test: Evaluation dataset inputs and targets
        - save_figures (boolean) if True: saves figures computed for evaluation locally
        - saving_base_output_name: (string) Base name for saving the evaluation files resulting  
        -------

        Returns
        ----------
        - y_pred_test: (array) predictions results on X_test
        - mae: (float) Mean Absolute Error
        - mse: (float) Mean Squared Error
        - rmse: (float) Root Mean Squared Error
        - oob_score: (float) OOB prediction of accuracy
        - r2: (float) R2 score
        - feature_imp: (dataFrame) Pandas datafame with features importance
        -------
        """
        print("\nEvaluation...")
        y_pred_test = self.model.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred_test)
        mse = metrics.mean_squared_error(y_test, y_pred_test)
        rmse =  np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
        oob_score = self.model.oob_score_ * 100
        r2 = metrics.r2_score(y_test, y_pred_test)
        print('\n\n\nMean Absolute Error (MAE):', mae)
        print('Mean Squared Error (MSE):', mse)
        print('Root Mean Squared Error (RMSE):', rmse)
        # Check out the "Out-of-Bag" (OOB) prediction score:
        print('OOB prediction of accuracy is: {oob}%\n'.format(oob=oob_score))
        print("R2:", r2)

        plt.figure(figsize=(5,5))
        plt.plot(list(range(0, int(y_test.max()))), ls='dashed', alpha=0.3)
        plt.scatter(y_test, y_pred_test, color='black')
        plt.title("Scatter plot test vs predicted values")
        plt.xlabel('Test')
        plt.ylabel('Predicted')
        if save_figures:
            plt.savefig(f'scatter_plot_{saving_base_output_name}.png')

        feature_imp = pd.DataFrame({'feature_name': self.feature_names,
                                    'importance': self.model.feature_importances_}).sort_values('importance', ascending=False)
        if save_figures:
            feature_imp.to_csv(f'features_importances_{saving_base_output_name}.csv')

        fig, ax = plt.subplots(figsize=(5,5))
        sns.barplot(x=feature_imp.importance, y=feature_imp.feature_name, ax=ax)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features", pad=15, size=14)
        if save_figures:
            plt.savefig(f'features_importance_{saving_base_output_name}.png')
        return y_pred_test, mae, mse, rmse, oob_score, r2, feature_imp


    def inference(self, mask_band, tiles_folder_name, tiles_in_GCP,
                 RF_output_folder_temp='RF_outputs_temp', path_to_tiles_local=''):
        """
            Run inference on tiles using trained sklearn rf model and merge the results.

        Parameters
        ----------
        - mask_band: (string) Name of band to use to remove "bleeding" from predictions
        - tiles_folder_name: (string) name of inference tiles folder
        - tiles_in_GCP: (boolean) If running with local tiles: tiles_in_GCP = False
        - RF_output_folder_temp: (string) path to output folder where the predictions will be saved
        - path_to_tiles_local: Local path to tiles if tiles_in_GCP = True
        -------

        """
        if not os.path.exists(RF_output_folder_temp):
            os.makedirs(RF_output_folder_temp)

        gcp_folder_path_inference_tiles = self.gcp_folder_name + '/' + tiles_folder_name
        # Find names of the predictors tiles in the GCP bucket and store them in list predictors_tiffiles
        if tiles_in_GCP:
            print("tiles_in_GCP")
            predictors_tiffiles = glob_blob_in_GCP(gcp_bucket=self.gcp_bucket,
                                                   gcp_folder_name=gcp_folder_path_inference_tiles,
                                                   extension='.tif')
        else:
            predictors_tiffiles = glob.glob(path_to_tiles_local + '/*.tif')
        print(f"There are {len(predictors_tiffiles)} inference tiles")

        i = 0
        # Loop through all predictors tiles and run inference on them
        # Results are saved locally, then uploaded to GCP, and deleted locally afterwards
        for path in predictors_tiffiles:
            print(f'\nStarting image: {path}')
            # Read image
            img_ds = gdal.Open(predictors_tiffiles[i], gdal.GA_ReadOnly)
            print('Image opened')
            bands = [img_ds.GetRasterBand(i).GetDescription() for i in range(1, img_ds.RasterCount + 1)]

            # Initialize tile that will only have the predictors features bands
            tile = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, len(self.feature_names)),
                            gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
            # Looping through the features used in the training and adding them as bands of {tile} in the same order as what
            # was used during the training
            for b in range(len(self.feature_names)):
                corresponding_raster_band_index = bands.index(self.feature_names[b])
                tile[:, :, b] = img_ds.GetRasterBand(corresponding_raster_band_index + 1).ReadAsArray()
            print(f"tile shape: {tile.shape}")

            # Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
            new_shape = (tile.shape[0] * tile.shape[1], tile.shape[2])
            tile_as_array = tile[:, :, :np.int(tile.shape[2])].reshape(new_shape)
            print('Reshaped from {o} to {n}'.format(o=tile.shape, n=tile_as_array.shape))
            tile_as_array = np.nan_to_num(tile_as_array)
            print(tile_as_array.shape)

            # Predict for each pixel
            class_prediction = self.run_rf_inference_on_tile(tile=tile, tile_as_array=tile_as_array)

            # Generate mask from first band of our predictors
            mask = np.copy(tile[:, :, self.feature_names.index(mask_band)]).astype(
                np.uint8)  # using the mask_branch layer here to have positive values
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
            outdata = driver.Create(classification_image, cols, rows, 1, gdal.GDT_Float32) # gdal.GDT_UInt16
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
                print('python3', 'gdal_merge.py', "-ot", "Float32", f"-o", f"{output_merged_tif.replace('.tif', f'_{i}.tif')}", f"{paths_pred_rasters[i]}", f"{paths_pred_rasters[i+1]}", "-a_nodata", "0", "-n", "0", "-co", "COMPRESS=DEFLATE")
                subprocess.run(['python3', 'gdal_merge.py', "-ot", "Float32", f"-o", f"{output_merged_tif.replace('.tif', f'_{i}.tif')}", f"{paths_pred_rasters[i]}", f"{paths_pred_rasters[i+1]}", "-a_nodata", "0", "-n", "0", "-co", "COMPRESS=DEFLATE"])
            # then we merge the previously merged output with the next tile
            else:
                print('\npython3', 'gdal_merge.py', "-ot", "Float32", f"-o", f"{output_merged_tif.replace('.tif', f'_{i}.tif')}", f"{output_merged_tif.replace('.tif', f'_{i-1}.tif')}", f"{paths_pred_rasters[i]}", "-a_nodata", "0", "-n", "0", "-co", "COMPRESS=DEFLATE")
            subprocess.run(['python3', 'gdal_merge.py', "-ot", "Float32", f"-o", f"{output_merged_tif.replace('.tif', f'_{i}.tif')}", f"{output_merged_tif.replace('.tif', f'_{i-1}.tif')}", f"{paths_pred_rasters[i]}", "-a_nodata", "0", "-n", "0", "-co", "COMPRESS=DEFLATE"])
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


