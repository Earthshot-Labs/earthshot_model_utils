#Todo: all dataset building, including IPCC tier 1, and datasets from earth engine
#Include feature and response variable code for ML project (mature biomass filtering should be consistent)
import ee
from utils import create_mask_forest_non_forest, create_mask_forest_age, create_mask_degree_urbanisation, \
    create_mask_forest_loss_proximity, create_mask_forest_gain_proximity, create_mask_roads_proximity, \
    create_mask_past_fires, export_ee_image_as_asset

def create_mature_forest_biomass_layer(export_layer=False):
    """
      Create Mature Forest Biomass (AGB in tCO2) Layer using the Spawn map

      Parameters
      ----------
      export_layer: bool

      Returns
      ----------
      biomass_agb_tco2_masked: ee image of the Spawn map AGB in tCO2 masked to only keep mature forests

    """
    ####### 1: Variables Initialisation for Potential Mature Forest Biomass (mfb) project #######
    mfb_project_date_range = ['2010-01-01', '2010-12-31']
    mf_min_age = 50  # in years
    mfb_project_distance_to_forest_change = 500  # in meters
    mfb_distance_to_roads = 1000  # in meters
    mfb_region_map = ee.Geometry.Polygon([[[-168.89638385228787,-61.22283125956169],
        [188.99424114771213,-61.22283125956169],
        [188.99424114771213,83.26616044597048],
        [-168.89638385228787,83.26616044597048],
        [-168.89638385228787,-61.22283125956169]]])  # globe
    mfb_description = 'Biomass_AGB_tCO2_mature_forests'
    mfb_scale = 100
    mfb_asset_name = f'projects/ee-mmf-mature-forest-biomass/assets/mature_forest_{mfb_project_date_range[0][0:4]}_AGB_tco2_scale_{mfb_scale}'

    ####### 2: Get the distribution of biomass #######
    # Biomass - Spawn dataset: https:#www.nature.com/articles/s41597-020-0444-4
    biomass = ee.ImageCollection("NASA/ORNL/biomass_carbon_density/v1").first()
    # Get Above Ground Biomass band and convert it to tCO2
    biomass_agb_tco2 = (biomass.select('agb').multiply(3.66).select([0], ['tCO2e']))

    biomass_agb_tco2_masked = create_filtered_map(image=biomass_agb_tco2,
                                                    date_range=mfb_project_date_range,
                                                    min_age=mf_min_age,
                                                    distance_to_forest_change=mfb_project_distance_to_forest_change,
                                                    distance_to_roads=mfb_distance_to_roads
                                                    )
    # Rename layer
    biomass_agb_tco2_masked = biomass_agb_tco2_masked.rename('AGB_tCO2_mature_forest')

    if export_layer:
        export_ee_image_as_asset(image=biomass_agb_tco2_masked,
                                 region=mfb_region_map,
                                 description=mfb_description,
                                 asset_name=mfb_asset_name,
                                 scale=mfb_scale,
                                 maxPixels=1e13)
    return biomass_agb_tco2_masked

def create_filtered_map(image, date_range, min_age, distance_to_forest_change, distance_to_roads):
    """
      Filters the image passed with the masks selected

      Parameters
      ----------
      image: ee image to be filtered
      date_range: list of two strings specifying which year to use for the masking
      min_age: age minimum of the forests to keep
      distance_to_forest_change: distance in meters to filter forest change close-by pixels
      distance_to_roads: distance in meters to filter roads close-by pixels

      Returns
      ----------
      image_masked = image masked using all the filters
    """
    year_limit = date_range[0][0:4]

    ####### 1: Create Masks #######
    # mask_forest = create_mask_forest_non_forest(date_range=date_range)
    mask_age_forest = create_mask_forest_age(age=min_age)
    mask_urbanisation_degree = create_mask_degree_urbanisation(year=year_limit)
    mask_forest_loss_proximity = create_mask_forest_loss_proximity(year=year_limit,
                                                                   distance_to_forest_change=distance_to_forest_change)
    mask_forest_gain_proximity = create_mask_forest_gain_proximity(
        distance_to_forest_change=distance_to_forest_change)
    mask_roads_proximity = create_mask_roads_proximity(distance_to_roads=distance_to_roads)
    mask_past_fires = create_mask_past_fires(year=year_limit)

    ####### 2: Apply masks #######
    print('Apply Forest age mask')
    image_masked = image.updateMask(mask_age_forest)
    # print('Apply Forest mask -- TODO: not used right now: do we want to use or not?')
    # image_masked = image_masked.updateMask(mask_forest)
    print('Apply Urbanisation mask')
    image_masked = image_masked.updateMask(mask_urbanisation_degree)
    print('Apply Forest Loss proximity mask')
    image_masked = image_masked.updateMask(mask_forest_loss_proximity)
    print('Apply Forest Gain proximity mask')
    image_masked = image_masked.updateMask(mask_forest_gain_proximity)
    print('Apply Roads proximity mask')
    image_masked = image_masked.updateMask(mask_roads_proximity)
    print('Apply Past Fires mask')
    image_masked = image_masked.updateMask(mask_past_fires)

    return image_masked

if __name__ == "__main__":
    ####### GEE Authentification #######
    # ee.Authenticate()
    ee.Initialize()

    print("====== Dataset Builder ======")
    do_create_mature_forest_biomass_layer = True
    export_mature_forest_biomass_layer = True
    print('\nCreating Mature Forest Biomass Layer...')
    mature_forest_biomass_layer = create_mature_forest_biomass_layer(export_layer=export_mature_forest_biomass_layer)
    print(mature_forest_biomass_layer.getInfo())
    print(f'\nMature Forest Biomass Layer created successfully. export_layer was set to: {export_mature_forest_biomass_layer}')
