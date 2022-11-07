####### Imports ######
import sys
# setting path
sys.path.append('..')

from dataset_builder import EEDatasetBuilder
import warnings
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

################## Variables Definitions ##################
##### 1. Dataset builder #####
biomass_raster = 'Walker_AGB_500m_tCO2'   # 'Spawn_AGB_tCO2e', 'GEDI_Biomass_1km_tCO2', 'Walker_AGB_500m_tCO2'
filter_dict = {'forest_non_forest': {'date_range': ['2010-01-01', '2010-12-31']},
               'min_forest_age': {'age': 50},
               'protected_areas': {},
               'very_low_density_rural': {'year': 2010},
                # 'forest_loss': {'year': 2010, 'distance': 5000},
                # 'forest_gain' : {'distance': 5000},
                # 'roads': {'distance': 5000},
                # 'fire': {'year': 2010}
               }
covariates_image_list = ['ecoregion', 'terrain', 'bioclim', 'terraclimate', 'soil']

##### 2. Actions to run #####
run_export_csv_samples = True
run_export_inference_tiles = False
run_export_image_as_ee_asset = False

##### 3. "Global Variables #####
gcp_bucket = 'eartshot-science-team'
gcp_folder_name = 'potential-mature-forest-biomass'
scale = 300
samples_folder_name = f'Walker_latin_am_samples_csv_scale{scale}_test_val_sets_10km_buffer'
tiles_folder_name = f'Walker_inference_south_am_tiles_500km_scale{scale}'

##### 4. Variables for: Export samples csv to bucket #####
# Gridded world shapefile asset in GEE
shp_asset_path_samples = 'projects/ee-margauxmasson21-shapefiles/assets/latin_america_gridded_5000km2'
numPixels = 2000

##### 5. Variables for: Export inference tiles to bucket #####
shp_asset_path_tiles = 'projects/ee-earthshot/assets/south_america_gridded_500km'
maxPixels = 1e13

##### 6. Variables for: Export image as an EE asset #####
name_asset = 'Walker_mature_forest_biomass_layer'

################## Dataset builder ##################
ee_dataset_builder = EEDatasetBuilder()
ee_dataset_builder.filtered_biomass_layer_from_raster(
    biomass_raster=biomass_raster,
    filter_dict=filter_dict
)
ee_dataset_builder.spatial_covariates(covariates=covariates_image_list)
print(ee_dataset_builder.image.bandNames().getInfo())

ee_dataset_builder.test_set(feature_collection='projects/ee-mature-forest-biomass-mmf/assets/test_set_mf_latin_am_v2',
                            buffer=10000, test_set_name='test_set_10km_buffer')
ee_dataset_builder.test_set(feature_collection='projects/ee-mature-forest-biomass-mmf/assets/test_set_mf_latin_am_v2',
                            buffer=None, test_set_name='test_set_no_buffer')
ee_dataset_builder.test_set(feature_collection='projects/ee-mature-forest-biomass-mmf/assets/validation_set_mf_latin_am_v2',
                            buffer=10000, test_set_name='val_set_10km_buffer')
ee_dataset_builder.test_set(feature_collection='projects/ee-mature-forest-biomass-mmf/assets/validation_set_mf_latin_am_v2',
                            buffer=None, test_set_name='val_set_no_buffer')

################## Export samples csv to GCP bucket ##################
if run_export_csv_samples:
    # This will take quite some time
    ee_dataset_builder.samples_csv_export(shp_asset_path_samples,
                                        name_gcp_bucket=gcp_bucket,
                                        folder_in_gcp_bucket=gcp_folder_name + '/' + samples_folder_name,
                                        numPixels=numPixels, scale=scale)

################## Export inference tiles to GCP bucket ##################
if run_export_inference_tiles:
    ee_dataset_builder_inference = EEDatasetBuilder()
    # New dataset builder with no filtering
    ee_dataset_builder_inference.filtered_biomass_layer_from_raster(
        biomass_raster=biomass_raster,
        filter_dict={}
    )
    ee_dataset_builder_inference.spatial_covariates(covariates=covariates_image_list)
    print(f'\nExport inference tiles using the shapefile: {shp_asset_path_tiles}...')
    # This will take quite some time
    ee_dataset_builder_inference.tiles_export(shp_asset_path_tiles,
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