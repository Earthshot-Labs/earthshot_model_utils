import pandas as pd
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
import seaborn as sns

import ee
from scipy.optimize import curve_fit

try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()



# constants
c_to_co2 = (44/12) #conversion factor c to co2 equivalent



# look up wood density from Zanne et al 2009: https://datadryad.org/stash/dataset/doi:10.5061/dryad.234
def wood_density_lookup(species_list, lat, lng):
    """
    function that takes species list and point location to get mean wood density (g/cm3) from Zanne et al 2009 DB
    look up wood density from Zanne et al 2009: https://datadryad.org/stash/dataset/doi:10.5061/dryad.234
    assumes that if country is listed in Zanne database then continent data does not apply to that country
    
    Parameters
    ----------
    species_list : [list]
                    genus and species strings in list
    lat : [float]
            decimal degree latitude for location
    lng : [float]
            decimal degree longitude for location
    
    Returns
    ------
    [pandas df], indexed by genus and species, column 'wd_gcm3' with mean wood density in g/cm3 based on loc
    """
    
    # load Zanne wood density DB
    wood_db_url = 'https://datadryad.org/stash/downloads/file_stream/94836'
    wood_db = pd.read_excel(wood_db_url, sheet_name='Data', index_col=0)
    wood_db.columns = ['family','binomial','wd_gcm3','region','ref_num']
    wood_db.region.unique() #regions are tricky, maybe input lat, lng and it spits out correct region

    # get geometries of world countries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # get coordinates of specific site
    #lng, lat = -42.769961, -22.453611 test values, actually get from func input
    df = pd.DataFrame({'name':['point1'],
                      'lat': lat,
                      'lng': lng})
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng,df.lat))
    gdf = gdf.set_crs(world.crs)

    pip = gpd.sjoin(gdf, world)

    # get list of north american countries that are in central america (minus mexico b/c tracked separately)
    countries_na_notcentral = ['United States of America','Canada','Mexico']

    # filter wood density by region based on lat lon
    # tropics are lat between -23.5 and 23.5
    if pip.loc[0,'name_right'] == 'China':
        # filter wood den for region = 'China'
        wood_db_here = wood_db[wood_db.region == 'China']
    elif pip.loc[0,'name_right'] == 'India':
        # filter wood den for region = 'India'
        wood_db_here = wood_db[wood_db.region == 'India']
    elif pip.loc[0,'name_right'] == 'Mexico':
        # filter wood den for region = 'Mexico'
        wood_db_here = wood_db[wood_db.region == 'Mexico']
    elif ((pip.loc[0,'name_right'] == 'Australia') & (pip.loc[0,'lat'] < -23.5)):
        # filter wood den for region = 'Australia'
        wood_db_here = wood_db[wood_db.region == 'Australia']
    elif ((pip.loc[0,'name_right'] == 'Australia') & (pip.loc[0,'lat'] >= -23.5) & (pip.loc[0,'lat'] <= 23.5)):
        #filter wood den for region = 'Australia/PNG (tropical)'
        wood_db_here = wood_db[wood_db.region == 'Australia/PNG (tropical)']
    elif (pip.loc[0,'name_right'] == 'Madagascar'):
        # filter wood den for region = 'Madagascar'
        wood_db_here = wood_db[wood_db.region == 'Madagascar']
    elif ((pip.loc[0,'continent'] == 'South America') & (pip.loc[0,'lat'] >= -23.5) & (pip.loc[0,'lat'] <= 23.5)):
        # filter wood den for region = South America (tropical) & South America (Tropical)
        wood_db_here = wood_db[(wood_db.region == 'South America (tropical)') | (wood_db.region == 'South America (Tropical)')]
    elif ((pip.loc[0,'continent'] == 'South America') & ((pip.loc[0,'lat'] < -23.5) | (pip.loc[0,'lat'] > 23.5))):
        # filter wood den for region = South America (extratropical)
        wood_db_here = wood_db[wood_db.region == 'South America (extratropical)']
    elif ((pip.loc[0,'continent'] == 'North America') & (pip.loc[0,'name_right'] not in countries_na_notcentral) & (pip.loc[0,'lat'] >= -23.5) & (pip.loc[0,'lat'] <= 23.5)):
        # filter wood den for region = Central America (tropical)
        wood_db_here = wood_db[wood_db.region == 'Central America (tropical)']
    elif (pip.loc[0,'continent'] == 'Europe'):
        # filter wood den for region = Europe
        wood_db_here = wood_db[wood_db.region == 'Europe']
    elif ((pip.loc[0,'name_right'] == 'United States of America') | (pip.loc[0,'name_right'] == 'Canada')):
        # filter wood den for region = NorthAmerica -- assuming that's USA + Canada b/c other countries in different categories
        wood_db_here = wood_db[wood_db.region == 'NorthAmerica']
    elif ((pip.loc[0,'continent'] == 'Africa') & (pip.loc[0,'lat'] >= -23.5) & (pip.loc[0,'lat'] <= 23.5)):
        # filter wood den for region = Africa (tropical)
        wood_db_here = wood_db[wood_db.region == 'Africa (tropical)']
    elif ((pip.loc[0,'continent'] == 'Africa') & ((pip.loc[0,'lat'] < -23.5) | (pip.loc[0,'lat'] > 23.5))):
        # filter wood den for region = Africa (extratropical)
        wood_db_here = wood_db[wood_db.region == 'Africa (extratropical)']
    elif ((pip.loc[0,'continent'] == 'Asia') & (pip.loc[0,'lat'] >= -23.5) & (pip.loc[0,'lat'] <= 23.5)):
        # filter wood den for region = South-East Asia (tropical)
        wood_db_here = wood_db[wood_db.region == 'South-East Asia (tropical)']
    elif ((pip.loc[0,'continent'] == 'Asia') & ((pip.loc[0,'lat'] > 23.5) | (pip.loc[0,'lat'] < -23.5))):
        #filter wood den for region = South-East Asia -- assuming that's extratropical
        wood_db_here = wood_db[wood_db.region == 'South-East Asia']
    elif (pip.loc[0,'continent'] == 'Oceana'):
        # filter wood den for region = Oceania
        wood_db_here = wood_db[wood_db.region == 'Oceania']
        
    # get standard variance for species that only have 1 measurement
    standard_std = wood_db_here.groupby(['binomial'])['wd_gcm3'].agg('std').mean()
    
    # filter species and get species mean
    #species_list = ['Abarema jupunba','Zygia latifolia'] for testing
    wood_db_here = wood_db_here[wood_db_here.binomial.isin(species_list)]
   # wood_den = wood_db_here.groupby(['binomial']).mean('wd_gcm3')[['wd_gcm3']] #used when only got mean
    wood_den = wood_db_here.groupby(['binomial'])['wd_gcm3'].agg(['mean','std','count'])
    # replace missing STD with the mean STD for that region
    wood_den['std'] = wood_den['std'].replace(np.nan, standard_std)
    return wood_den






# chapman richards curve fit
def curve_fun(x, k, p):
    """
    basic math for chapman richards curve fit, called by curve_fit_func
    
    Parameters
    ----------
    x : vector of data x values
    k : [float]
        k parameter
    p : [float]
        p parameter
    
    Returns
    -------
    vector of y values
    """
    y = x[:,1] * np.power( (1 - np.exp(-k * x[:,0])), p)
    return y


def sigmoid_fun(x, a, b):
    """
    basic math for sigmoid curve fit, called by curve_fit_func
    
    Parameters
    ----------
    x : vector of data x values
    a : [float]
        a parameter
    b : [float]
        b parameter
    
    Returns
    -------
    vector of y values
    """
    y = 1 / (1 + np.exp(-a * (x[:,0] - b)))
    return y


    # nice visual of effect of changing parameters: https://datascience.oneoffcoder.com/s-curve.html
def logistic_fun(x, L=1, x_0=0, k=1):
    """
    basic math for logistic curve fit, called by curve_fit_func
    y(t) = c / (1 + a*exp(-b*t)) where t is age and c is the maximum biomass (here replaced c in numerator with 1 since multiply by max biomass)
    
    Parameters
    ----------
    x : vector of data x values
    a : [float]
        a parameter
    b : [float]
        b parameter
    
    Returns
    -------
    vector of y values
    """
    y = x[:,1] / (1 + np.exp(-k * (x[:,0] - x_0)))
    return y


def mature_biomass_spawn(lat, lng, buffer=20):
    """
    Get mature biomass (ymax) in tCO2e/ha including aboveground and belowground biomass from Spawn et al. (2020)
    
    Parameters
    ---------
    lat : [float]
          latitude in decimal degrees for project location
    lng : [float]
          longitude in decimal degrees for project location
    buffer : [float]
             distance in km over which to search for mature biomass (default = 20 km)
    
    Returns
    -------
    mature biomass in tCO2e/ha including aboveground and belowground biomass
    """
    
    # get max from carbon density GEE -----------------
    pt = ee.Geometry.Point(lng, lat) #x,y
    buffered_pt = pt.buffer(distance=buffer*1000)
    biomass_image = ee.ImageCollection("NASA/ORNL/biomass_carbon_density/v1").first()
    
    # if want to swap out geojson shapefile instead of point
    #aoi = ee.FeatureCollection(geojson['features']) #function input should be geojson instead of lat, lng
    #bounds = ee.Geometry(aoi.geometry(maxError=100))
    #buffered_pt = bounds.buffer(distance=buffer*1000, maxError=1000) #distance=20000

    agb_image = biomass_image.select('agb')

    # Take the maximum AGB in a radius of 20km 
    sample_agb = agb_image.reduceRegion(
        geometry=buffered_pt, 
        reducer=ee.Reducer.max(),
        scale=300)

    bgb_image = biomass_image.select('bgb')

    # Take the maximum AGB in a radius of 20km 
    sample_bgb = bgb_image.reduceRegion(
        geometry=buffered_pt, 
        reducer=ee.Reducer.max(),
        scale=300)

    data_dict_agb = sample_agb.getInfo() #gets tC/ha
    data_dict_bgb = sample_bgb.getInfo() #gets tC/ha

    y_max_agb = data_dict_agb['agb'] * c_to_co2
    y_max_bgb = data_dict_bgb['bgb'] * c_to_co2
    y_max_agb_bgb = y_max_agb + y_max_bgb
    
    return y_max_agb_bgb


# maximum biomass using Joe's deciles method

def _formatDecileResponse(features):
    """
    Private function making data out of EE more accessible
    """
    return {p['ECO_ID']: {
                'area': p['SHAPE_AREA'],
                'biome_num': p['BIOME_NUM'],
                'biome_name': p['BIOME_NAME'],
                'eco_num': p['ECO_ID'],
                'eco_name': p['ECO_NAME'],
                'eco_biome_code': p['ECO_BIOME_'],
                'realm': p['REALM'],
                'nature_needs_half': p['NNH'],
                'tCO2e_decile_labels': [5,10,20,30,40,50,60,70,80,90,95],
                'tCO2e_deciles': [round(d,2) for d in
                                 [p['p5'], p['p10'],p['p20'],p['p30'],
                                  p['p40'],p['p50'],p['p60'],p['p70'],
                                  p['p80'],p['p90'],p['p95']]],
                'tCO2e_max': round(p['p100'],2)
                }
                for p in [f['properties'] for f in features['features']]
            }
    
def getNearbyMatureForestPercentiles(geojson, buffer=20):
    """
    Take a geojson structure (output from landOS) and get a list of the deciles of biomass (agb+bgb) in tCO2e/ha
    
    Parameters
    ----------
    geojson : [dict]
               dictionary of shapefile that you get as the output from landOS for the "shapefile"
    buffer : [float]
              buffer distance in km (default = 20 km)
    
    Returns
    -------
    return[0] : biomass (agb+bgb) in tCO2e/ha from 20km buffer at 5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 95%
    return[1] : maximum biomass (agb+bgb) in tCO2e/ha from 20km buffer
    """
    
    aoi = ee.FeatureCollection(geojson['features'])
    
    ## 1. Find ecoregions that overlap with the AOI
    ecoregions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017');

    # Get set of ecoregions that occur in the area of interest and limit distance to no more than 20km away
    bounds = ee.Geometry(aoi.geometry(maxError=100))
    buffered = bounds.buffer(distance=buffer*1000, maxError=1000) #distance=20000
    searchAreas = ecoregions.filterBounds(bounds).map(
                        lambda f: f.intersection(buffered))


    ## 2. Within those ecoregions, find "mature forests"
    #Additional Forest Non/Forest in 20210 from PALSAR
    forestMask= (ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF")
                        .filterDate('2009-01-01', '2011-12-31')
                        .first().select('fnf').remap([1],[1],0))

    #Forest Age as UInt8, 'old growth'==255
    forestAge = ee.Image("projects/es-gis-resources/assets/forestage").select([0], ['forestage']);

    # Find forests that are 50 years old, or at least older than 90% of forests in the ecoregion
    matureForest = forestAge.gte(
                        forestAge.reduceRegions(searchAreas, ee.Reducer.percentile([90]))
                        .reduceToImage(['p90'], 'first')
                        .min(50)
                    )

    ## 3. Get the distribution of biomass as deciles of those mature forests
    #Biomass - Spawn dataset: https://www.nature.com/articles/s41597-020-0444-4
    biomass = ee.ImageCollection("NASA/ORNL/biomass_carbon_density/v1").first()
    biomass = (biomass.select('agb').add(biomass.select('bgb'))
                    .multiply(3.66).select([0], ['tCO2e'])) #agb_bgb in tCO2e/ha

    # Mask away non forests and young forests, and then get the pdf
    featureDeciles = (biomass.mask(forestMask).mask(matureForest)
                            .reduceRegions(searchAreas,
                                   ee.Reducer.percentile([5,10,20,30,40,50,60,70,80,90,95,100]),
                                   scale=100)
                            ).map(lambda f: ee.Feature(None, f.toDictionary()))

    # Return a cleaned-up response
    output_dict = _formatDecileResponse(featureDeciles.getInfo()) #could go back to returning the whole dict

    for eco_id, ecozone in output_dict.items():
        tCO2eha_deciles = ecozone['tCO2e_deciles']
        tCO2eha_max = ecozone['tCO2e_max']
        
    return tCO2eha_deciles, tCO2eha_max



def getWalkerMatureForestPercentiles(geojson, buffer=20):
    """
    Take a geojson structure (output from landOS) and get a list of the deciles of max potential C (agb+bgb) in tCO2e/ha
    From Walker dataset
    
    Parameters
    ----------
    geojson : [dict]
               dictionary of shapefile that you get as the output from landOS for the "shapefile"
    buffer : [float]
              buffer distance in km (default = 20 km)
    
    Returns
    -------
    return[0] : biomass (agb+bgb) in tCO2e/ha from 20km buffer at 5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 95%
    return[1] : maximum biomass (agb+bgb) in tCO2e/ha from 20km buffer
    """
    
    aoi = ee.FeatureCollection(geojson['features'])
    
    ## 1. Find ecoregions that overlap with the AOI
    ecoregions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017');

    # Get set of ecoregions that occur in the area of interest and limit distance to no more than 20km away
    bounds = ee.Geometry(aoi.geometry(maxError=100))
    buffered = bounds.buffer(distance=buffer*1000, maxError=1000) #distance=20000
    searchAreas = ecoregions.filterBounds(bounds).map(
                        lambda f: f.intersection(buffered))


    ## 2. Within those ecoregions, find "mature forests"
    #Additional Forest Non/Forest in 20210 from PALSAR
    forestMask = (ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF")
                        .filterDate('2009-01-01', '2011-12-31')
                        .first().select('fnf').remap([1],[1],0))
    
    # Walker Potential C storage
    walker_potC = ee.Image('projects/ee-anikastaccone-regua/assets/Base_Pot_AGB_BGB_MgCha_500m')

    # Convert C -> CO2e
    walker_potC = (walker_potC.select('b1')
                   .multiply(3.66)
                   .select([0], ['tCO2e'])) #agb_bgb in tCO2e/ha

    # Mask away non forests and young forests, and then get the pdf
    featureDeciles = (walker_potC.mask(forestMask)
                            .reduceRegions(searchAreas,
                                   ee.Reducer.percentile([5,10,20,30,40,50,60,70,80,90,95,100]),
                                   scale=100)
                            ).map(lambda f: ee.Feature(None, f.toDictionary()))

    # Return a cleaned-up response
    output_dict = _formatDecileResponse(featureDeciles.getInfo()) #could go back to returning the whole dict

    for eco_id, ecozone in output_dict.items():
        tCO2eha_deciles = ecozone['tCO2e_deciles']
        tCO2eha_max = ecozone['tCO2e_max']
        
    return tCO2eha_deciles, tCO2eha_max    






def root_shoot_ipcc(lat, lng, veg_type='other broadleaf'):
    """
    Parameters
    ----------
    lat : [float]
          latitude of project in decimal degrees
    lng : [float]
          longitude of project in decimal degrees
    veg_type : [string]
              forest type from IPCC choices -- may need to look at IPCC documentation to determine which is relevant
              'coniferous', 'natural', 'broadleaf', 'cunninghamia sp.','eucalyptus sp.', 'picea abies', 
              'pinus massoniana', 'pinus sp.', 'other broadleaf', 'tectona grandis', 'other', 'larix sp.', 
              'pinus koraiensis', 'pinus sylvestris', 'pinus tabuliformis', 'poplar sp.', 'robinia pseudoacacia', 
              'abies sp.', 'oaks and other hardwoods', 'picea sp.', 'populus sp.','pseudotsuga menziesii', 
              'acacia crassicarpa','castanopsis hystrix', 'mixed plantation','quercus and other hardwoods', 
              'acacia auriculiformis','acaica mangium', 'cassia montana', 'cedeus libani', 'oil palm',
              'swietenia macrophylla', 'acacia mangium', 'gmelina arborea','hevea brasiliensis', 'mangifera indica',
              'mixed', 'acacia sp.','azadirachta indica', 'casuarina equisetifolia', 'pongamia pinnata'
              Default is 'other broadleaf' since that seems like a likely option for these projects
    
    Returns
    -------
    return[0] : agb biomass threshold between young and old root to shoot, root_shoot_break (t/ha)
    return[1] : root to shoot ratio for young trees, rs_young
    return[2] : root to shoot ratio for old trees, rs_old
    """
    
    # load IPCC root to shoot table (from Joe's scripts)
    root_shoot_ipcc = pd.read_csv('https://raw.githubusercontent.com/Earthshot-Labs/science/master/IPCC_tier_1/prediction/ipcc_table_intermediate_files/ipcc_tier1_all.csv?token=GHSAT0AAAAAABQWL3QREMVXR567IWPOZF22YV5W4YQ')

    # open gez2010 shapefile for Global Ecoological Zones
    dir_here = os.getcwd()
    gez_shp = dir_here + '/deepdive_automation/gez2010/gez_2010_wgs84.shp'
    gez_gdf = gpd.read_file(gez_shp)
    
    # Get GEZ
    point_xy = [[lng], [lat]]
    poi = gpd.points_from_xy(x=point_xy[0], y=point_xy[1])

    idx = gez_gdf['geometry'].contains(poi[0])
    eco_type = gez_gdf.loc[idx,'gez_name'].values[0].lower()

    # get geometries of world countries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    idx = world['geometry'].contains(poi[0])
    continent = world.loc[idx,'continent'].values[0].lower()

    # get relevant rows
    rs_rows = root_shoot_ipcc[(root_shoot_ipcc.continent == continent) & 
                              (root_shoot_ipcc.nump == eco_type) & 
                              (root_shoot_ipcc.forest_type == for_type)]

    # if multiple rows then average values???
    root_shoot_break = rs_rows['root_shoot_break'].mean()
    rs_young = rs_rows['rs_young'].mean()
    rs_old = rs_rows['rs_old'].mean()
    
    return root_shoot_break, rs_young, rs_old

    





def curve_fit_func(input_df, d_type, curve_type, y_max_agb_bgb, rs_break, rs_young=0.285, rs_old=0.285, biomass_to_c=0.47):
    """
    function to take dataframe of agb+bgb biomass, location, and constants to execute curve fitting
    
    Parameters
    ----------
    input_df : [pandas dataframe]
                pandas dataframe with columns 'agb_t_ha', 'bgb_t_ha', 'agb_bgb_t_ha' for the biomass data at location
    d_type : [string]
            'biomass' if input_df contains biomass in t/ha
            'carbon' if input_df contains carbon in t/ha
    curve_type : [string]
                'chapman-richards' if want default curve fit
                'sigmoid' if want sigmoid curve fit
            longitude of site in decimal degrees
    y_max_agb_bgb : [float]
                    maximum biomass from mature forest in tCO2e/ha
    rs_break : [float]
                aboveground biomass at which the root-to-shoot ratio switches from young value to old value from IPCC tier 1 tables 
    rs_young : [float]
               root-to-shoot ratio for young forests (with lower biomass than rs_break)
               default = 0.285, value from IPCC Tier 1 table for moist tropical forest with biomass < 125
    rs_old : [float]
              root-to-shoot ratio for old forests (with hiher biomass than rs_break)
              default = 0.285, value from IPCC Tier 1 table for moist tropical forests
    biomass_to_c : [float]
                    fraction of biomass that is C, here default is 0.47 but user can change for biome or location specific
    
    Returns
    -------
    output[0] : plot of chapman richards curve with data points displayed
    output[1] : table of projected C accumulation with columns age (1-100) and tCO2e/ha
    """
    
    # fill in missing bgb, agb+bgb ------------
    for i in range(0, input_df.shape[0]):
    
        # if have agb but not bgb or agb+bgb ... use root-to-shoot to get bgb ... agb+bgb is sum of cols 2,3
        if pd.notna(input_df.at[i,'agb_t_ha']) & pd.isna(input_df.at[i,'bgb_t_ha']) & pd.isna(input_df.at[i,'agb_bgb_t_ha']):
            # if agb > rs_break then use rs_old, else use rs_young
            if input_df.at[i,'agb_t_ha'] > rs_old:
                input_df.at[i,'bgb_t_ha'] = input_df.at[i,'agb_t_ha'] * rs_old
            else:
                input_df.at[i,'bgb_t_ha'] = input_df.at[i,'agb_t_ha'] * rs_young
            input_df.at[i,'agb_bgb_t_ha'] = input_df.at[i,'agb_t_ha'] + input_df.at[i,'bgb_t_ha']

        # if have agb and bgb but not agb+bgb ... sum cols 2,3
        elif pd.notna(input_df.at[i,'agb_t_ha']) & (pd.notna(input_df.at[i,'bgb_t_ha'])) & pd.isna(input_df.at[i,'agb_bgb_t_ha']):
            input_df.at[i,'agb_bgb_t_ha'] = input_df.at[i,'agb_t_ha'] + input_df.at[i,'bgb_t_ha']
        
    # average plots of same age
    input_df = input_df.groupby(['age']).agg({'agb_t_ha':'mean',
                                            'bgb_t_ha':'mean',
                                            'agb_bgb_t_ha':'mean'})
    input_df.reset_index(drop=False, inplace=True)

    # convert biomass to CO2e -----------------
    if (d_type == 'biomass'):
        input_df['agb_tCO2e_ha'] = input_df['agb_t_ha'] * biomass_to_c * c_to_co2
        input_df['bgb_tCO2e_ha'] = input_df['bgb_t_ha'] * biomass_to_c * c_to_co2
        input_df['agb_bgb_tCO2e_ha'] = input_df['agb_bgb_t_ha'] * biomass_to_c * c_to_co2
    else:
        input_df['agb_tCO2e_ha'] = input_df['agb_t_ha'] * c_to_co2
        input_df['bgb_tCO2e_ha'] = input_df['bgb_t_ha'] * c_to_co2
        input_df['agb_bgb_tCO2e_ha'] = input_df['agb_bgb_t_ha'] * c_to_co2

    # prepare data for curve fit -----------
    age = np.array(input_df['age']).reshape((input_df['age'].shape[0],1))
    agb_bgb_tco2_ha = input_df['agb_bgb_tCO2e_ha']

    y_max_array = np.ones_like(age) * y_max_agb_bgb
    x_data = np.concatenate((age, y_max_array), axis=1)

    # curve fit
    # find parameters k and p
    if curve_type == 'sigmoid':
        params, covar = curve_fit(f=sigmoid_fun, xdata=x_data, ydata=agb_bgb_tco2_ha)
    elif curve_type == 'logistic':
        L_estimate = agb_bgb_tco2_ha.max()
        x_0_estimate = np.median(age)
        k_estimate = 1.0
        p_0 = [L_estimate, x_0_estimate, k_estimate]
        params, covar = curve_fit(logistic_fun, x_data, agb_bgb_tco2_ha, p_0, method='dogbox',
            bounds=((-np.inf,-np.inf,0.1),(np.inf,np.inf,5)))
    else:
        params, covar = curve_fit(f=curve_fun, xdata=x_data, ydata=agb_bgb_tco2_ha)

    # Generate 100 yr prediction ------------
    x_plot = np.arange(1,101,1).reshape((100,1))
    y_max_array_plot = np.ones_like(x_plot) * y_max_agb_bgb
    x_data_plot = np.concatenate((x_plot, y_max_array_plot), axis=1)

    if curve_type == 'sigmoid':
        pred_agb_bgb = sigmoid_fun(x=x_data_plot, a=params[0], b=params[1])
    elif curve_type == 'logistic':
                pred_agb_bgb = logistic_fun(x_data_plot, L=params[0], x_0=params[1], k=params[2])
    else:
        pred_agb_bgb = curve_fun(x=x_data_plot, k=params[0], p=params[1])

    # Make plot ----------------
    c_fig, ax = plt.subplots(figsize=(15,5))

    ax.set_title('AGB+BGB estimates (tCO2e/ha) over the next 100 years', fontsize=15)
    ax.set_ylabel('AGB+BGB estimate (tCO2e/ha)', fontsize=15)
    ax.set_xlabel('Age', fontsize=15)
    ax.plot(x_plot, pred_agb_bgb)
    ax.scatter(age, agb_bgb_tco2_ha, color='orange')

    # output predictions ---------------
    c_curve = pd.DataFrame({'age': x_plot.reshape(1,100).tolist()[0],
                            'tCO2/ha': pred_agb_bgb.reshape(1,100).tolist()[0]})

    return c_fig, c_curve


# chave allometry if you have height data
def chave_allometry_height(WD, DBH, H):
    """
    Function to return individual tree AGB (kg biomass) from wood density (g/cm3, DBH (cm), and height (m) data on individual stems
    NOTE: there is a different Chave allometry function for instances where you don't have tree height data
    
    Parameters
    ----------
    WD : [float]
         wood density (g/cm3) for individual stem (should be species, region specific)
    DBH : [float]
          diameter at breast height (cm) for individual stems (for multistem trees DBH = sqrt(DBH1**2 + DBH2**2 + DBH3**2))
    H : [float]
        tree height (m) for indivdual trees
    
    Return
    ------
    AGB for individual tree in kg
    """
    return WD * np.exp(-2.977 + np.log(WD * DBH**2 * H))


# chave allometry if you don't have height data
def chave_allometry_noheight(DBH, WD, ftr_collection="", lat=np.nan, lng=np.nan):
    """
    Function takes shapefile and dataframe of DBHs and WD to calculate tree-level AGB in kg
    
    Parameters
    ----------
    ftr_collection : [string]
                      GEE asset containing shapefile for project
                      default is empty string
    lat : [float]
            latitude in decimal degrees to use if don't have shapefile
    lng : [float]
            longitude in decimal degrees to use if don't have shapefile
    DBH : [float]
           list of DBHs in cm for individual stems
    WD : [float]
          matching list of wood densities in g/cm3 for individual stems, in Chave this is called rho
    
    Returns
    -------
    AGB for each stem in kg
    """
    
    if len(ftr_collection) > 0:
        roi = ee.FeatureCollection(ftr_collection)
    else:
        pt = ee.Geometry.Point(lng, lat) #x,y
        roi = pt.buffer(distance=100)

    # Environmental stress factor on the diameter-height tree allometry
    environmental_stress_factor = ee.Image("projects/ee-margauxmasson-madre-de-dios/assets/Environmental_stress_factor_chave") # E equation global gridded layer of E at 2.5 arc sec resolution 

    # Taking the mean environmental_stress_factor over the MDD region
    sample_environmental_stress_factor = environmental_stress_factor.reduceRegion(
        geometry = roi,
        reducer = ee.Reducer.mean(), 
        scale = 300)
    
    data_dict_environmental_stress_factor = sample_environmental_stress_factor.getInfo()
    E = data_dict_environmental_stress_factor['b1'] # only one band 

    # calculate biomass
    AGB_kg = np.exp(-1.803 - 0.976*E + 0.976 * np.log(WD) + 2.673 * np.log(DBH)- 0.0299*(np.log(DBH)**2))
    return AGB_kg