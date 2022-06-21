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
    elif ((pip.loc[0,'continent'] == 'North America') & (~pip.loc[0,'name_right'].isin(countries_na_notcentral)) & (pip.loc[0,'lat'] >= -23.5) & (pip.loc[0,'lat'] <= 23.5)):
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
    
    # filter species and get species mean
    #species_list = ['Abarema jupunba','Zygia latifolia'] for testing
    wood_db_here = wood_db_here[wood_db_here.binomial.isin(species_list)]
    wood_den = wood_db_here.groupby(['binomial']).mean('wd_gcm3')[['wd_gcm3']]
    return wood_den