import ee

def create_mask_forest_non_forest(date_range=['2010-01-01','2010-12-31']):
  """
    Create a forest/non forest mask using the Global PALSAR-2/PALSAR Forest/Non-Forest Layer.

    Parameters
    ----------
    date_range: list of two strings specifying which year to use for the forest/non forest map -- available years are: 2007–2017

    Returns
    ----------
    mask_forest = pixels belonging to forests for the date range specified
  """
  dataset_forest = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/FNF').filterDate(date_range).first()  # global 25m resolution PALSAR-2/PALSAR SAR mosaic
  # Select the Forest/Non-Forest landcover classification band
  fnf = dataset_forest.select('fnf')
  # Select pixels = 1 (= Forest pixels)
  mask_forest = fnf.eq(1)  # 1:forest, 2:nonforest, 3:water
  return mask_forest

def create_mask_forest_age(age=100):
  """
    Create a forest age mask using the forest age from "Mapping global forest age from forest inventories, biomass and climate data"
    https://essd.copernicus.org/articles/13/4881/2021/essd-13-4881-2021.pdf

    Parameters
    ----------
    age: age (in years) minimum to create mask: only forests older than age are kept, others are masked out

    Returns
    ----------
    mask_age_forest = pixels belonging to forests older than age
  """
  forestAge = ee.Image("projects/es-gis-resources/assets/forestage").select([0], ['forestage'])
  # Get forests older than age
  mask_age_forest = forestAge.gte(age)
  return mask_age_forest

def create_mask_degree_urbanisation(year='2010'):
  """
    Create degree of urbanisation mask using the GHSL - Global Human Settlement Layer: https://ghsl.jrc.ec.europa.eu/ghs_smod2022.php
    Values:
    # # 30: URBAN CENTRE GRID CELL
    # # 23: DENSE URBAN CLUSTER GRID CELL
    # # 22: SEMI-DENSE URBAN CLUSTER GRID CELL
    # # 21: SUBURBAN OR PERI-URBAN GRID CELL
    # # 13: RURAL CLUSTER GRID CELL
    # # 12: LOW DENSITY RURAL GRID CELL
    # # 11: VERY LOW DENSITY RURAL GRID CELL
    # # 10: WATER GRID CELL
    # # NoData [-200]
    Parameters
    ----------
    year: year (string) of the degree of urbanisation map (we only have 2010 and 2020 for now)

    Returns
    ----------
    mask_urbanisation_degree = pixels belonging to very low density rural grill cells (11)
  """
  dataset_urbanisation = ee.Image(f'projects/ee-mmf-mature-forest-biomass/assets/GHS_SMOD_E{year}_GLOBE_R2022A_54009_1000_V1_0')
  mask_urbanisation_degree = dataset_urbanisation.eq(11)
  return mask_urbanisation_degree

def create_mask_forest_loss_proximity(year='2010', distance_to_forest_change=500):
  """
    Create Forest loss proximity mask using the Hansen Global Forest Change v1.9 (2000-2021)

    Parameters
    ----------
    year: year (string) of the forest change year limit (up to this year)
    distance_to_forest_change: int distance to forest loss in meters

    Returns
    ----------
    mask_forest_loss_proximity = pixels belonging to areas that did not have forest loss between 2000 and {year}
  """
  dataset_hansen_loss = ee.Image('UMD/hansen/global_forest_change_2021_v1_9').select('lossyear')
  # Distance from forest loss from before {year}
  distance_forest_loss = dataset_hansen_loss.lte(int(year[-2] + year[-1])).distance(ee.Kernel.euclidean(distance_to_forest_change, 'meters'))
  mask_forest_loss_proximity = distance_forest_loss.mask().eq(0)
  return mask_forest_loss_proximity

def create_mask_forest_gain_proximity(distance_to_forest_change=500):
  """
    Create Forest gain proximity mask using the Hansen Global Forest Change v1.9 (2000-2021)

    Parameters
    ----------
    distance_to_forest_change: int distance to forest gain in meters

    Returns
    ----------
    mask_forest_gain_proximity = pixels belonging to areas that did not have forest gain between 2000 and 2012
  """
  dataset_hansen_gain = ee.Image('UMD/hansen/global_forest_change_2021_v1_9').select('gain')
  # Distance from forest gain from before {year}
  distance_forest_gain = dataset_hansen_gain.distance(ee.Kernel.euclidean(distance_to_forest_change, 'meters'))
  mask_forest_gain_proximity = distance_forest_gain.mask().eq(0)
  return mask_forest_gain_proximity

def create_mask_roads_proximity(distance_to_roads=1000):
  """
    Create roads proximity mask using the Global Roads Open Access Data Set (gROADS), v1 (1980–2010) dataset

    Parameters
    ----------
    distance_to_roads: int distance to roads in meters

    Returns
    ----------
    mask_roads_proximity = pixels belonging to areas that are more than {distance_to_roads} meters away from a road
  """
  dataset_roads = ee.FeatureCollection('projects/ee-mmf-mature-forest-biomass/assets/gROADSv1')
  distance_roads = dataset_roads.distance(ee.Number(distance_to_roads))
  mask_roads_proximity = distance_roads.mask().eq(0)
  return mask_roads_proximity

def create_mask_past_fires(year='2010'):
  """
    Create past fires mask using FireCCI51: MODIS Fire_cci Burned Area Pixel Product, Version 5.1

    Parameters
    ----------
    year: year (string) of the fire event year limit (up to this year)

    Returns
    ----------
    mask_past_fires = pixels belonging to areas that did not have a fire event before {year}
  """
  dataset = ee.ImageCollection('ESA/CCI/FireCCI/5_1').filterDate('2001-01-01', f'{year}-12-31')
  burnedArea = dataset.select('BurnDate')
  maxBA = burnedArea.max()
  mask_past_fires = maxBA.mask().eq(0)
  return mask_past_fires

def export_ee_image_as_asset(image, region, description, asset_name, scale, maxPixels=1e13):
    """
      Exports an ee Image as an ee asset.
      Doc: https://developers.google.com/earth-engine/apidocs/export-image-toasset

      Parameters
      ----------
      image: ee Image to be exported
      region: A LinearRing, Polygon, or coordinates representing region to export. These may be specified as the Geometry objects or coordinates serialized as a string.
      description: A human-readable name of the task.
      asset_name: The destination asset ID.
      scale: Resolution in meters per pixel.
      maxPixels: Restrict the number of pixels in the export.
    """
    print('\n== export_ee_image_as_asset ==')
    print(description)
    print(asset_name)
    print(scale)
    task = ee.batch.Export.image.toAsset(image=image,
                                         region=region,
                                         description=description,
                                         assetId=asset_name,
                                         scale=scale,
                                         maxPixels=maxPixels)
    task.start()
