####### Imports ######
import sys
# setting path
sys.path.append('..')

from dataset_builder import EEDatasetBuilder
import warnings
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

################## Variables Definitions ##################
##### 1. Dataset builder #####
# Training target: Response variable
biomass_raster = 'Spawn_AGB_tCO2e'
filter_dict = {'forest_non_forest': {'date_range': ['2010-01-01', '2010-12-31']},
               'min_forest_age': {'age': 40},
               'protected_areas': {},
                # 'very_low_density_rural': {'year': 2010},
                # 'forest_loss': {'year': 2010, 'distance': 5000},
                # 'forest_gain' : {'distance': 5000},
                # 'roads': {'distance': 5000},
                # 'fire': {'year': 2010}
               }
# Predictors
covariates_image_list = ['ecoregion', 'terrain', 'bioclim', 'terraclimate', 'soil']

##### 2. Actions to run #####
run_export_csv_samples = False
run_export_inference_tiles = False
run_export_image_as_ee_asset = False

##### 3. "Global Variables #####
gcp_bucket = 'eartshot-science-team'
gcp_folder_name = 'potential-mature-forest-biomass'
samples_folder_name = 'samples_csv'
tiles_folder_name = 'inference_tiles'
scale = 100

##### 4. Variables for: Export samples csv to bucket #####
# Gridded world shapefile asset in GEE
shp_asset_path_samples = 'projects/ee-margauxmasson21-shapefiles/assets/latin_america_gridded_5000km2'
numPixels = 2000

##### 5. Variables for: Export inference tiles to bucket #####
shp_asset_path_tiles = 'projects/ee-margauxmasson21-shapefiles/assets/latin_america_gridded_10degrees'
maxPixels = 1e13

##### 6. Variables for: Export image as an EE asset #####
name_asset = 'projects/ee-earthshot/assets/mature_forest_biomass'

################## Dataset builder ##################
ee_dataset_builder = EEDatasetBuilder()
ee_dataset_builder.filtered_biomass_layer_from_raster(
    biomass_raster=biomass_raster,
    filter_dict=filter_dict
)
ee_dataset_builder.spatial_covariates(covariates=covariates_image_list)
print(ee_dataset_builder.image.bandNames().getInfo())

################## Export samples csv to GCP bucket ##################
if run_export_csv_samples:
    # This will take quite some time
    ee_dataset_builder.samples_csv_export(shp_asset_path_samples,
                                        name_gcp_bucket=gcp_bucket,
                                        folder_in_gcp_bucket=gcp_folder_name + '/' + samples_folder_name,
                                        numPixels=numPixels, scale=scale)

################## Export inference tiles to GCP bucket ##################
if run_export_inference_tiles:
    print(f'\nExport inference tiles using the shapefile: {shp_asset_path_tiles}...')
    # This will take quite some time
    ee_dataset_builder.tiles_export(shp_asset_path_tiles,
                                        name_gcp_bucket=gcp_bucket,
                                        folder_in_gcp_bucket=gcp_folder_name + '/' + tiles_folder_name,
                                        maxPixels=maxPixels, scale=scale)
    print('Inference tiles export done.\n')

################## Export ee image as an ee asset ##################
if run_export_image_as_ee_asset:
    print(f'\nExport ee image as an ee asset')
    ee_dataset_builder.export_image_as_asset(name_asset=name_asset,
                                             scale=scale,
                                             maxPixels=maxPixels)
