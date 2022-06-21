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



def curve_fit_func(input_df, lat, lng, root_to_shoot=0.285, biomass_to_c=0.47):
    """
    function to take dataframe of agb+bgb biomass, location, and constants to execute Chapman Richards curve fitting
    
    Parameters
    ----------
    input_df : [pandas dataframe]
                pandas dataframe with columns 'agb_t_ha', 'bgb_t_ha', 'agb_bgb_t_ha' for the biomass data at location
    lat : [float]
            latitude of site in decimal degrees
    lng : [float]
            longitude of site in decimal degrees
    root_to_shoot : [float]
                    root to shoot ratio, should change this to extract from IPCC tier 1 tables
                    default = 0.285, value from IPCC Tier 1 table for moist tropical forest with biomass < 125
    biomass_to_c : [float]
                    fraction of biomass that is C, here default is 0.47 but user can change for biome or location specific
    
    Returns
    -------
    output[0] : plot of chapman richards curve with data points displayed
    output[1] : table of projected C accumulation with columns age (1-100) and tCO2e/ha
    """

    # constants
    c_to_co2 = (44/12) #conversion factor c to co2 equivalent
    
    # fill in missing bgb, agb+bgb ------------
    for i in range(0, input_df.shape[0]):
    
        # if have agb but not bgb or agb+bgb ... use root-to-shoot to get bgb ... agb+bgb is sum of cols 2,3
        if pd.notna(input_df.at[i,'agb_t_ha']) & pd.isna(input_df.at[i,'bgb_t_ha']) & pd.isna(input_df.at[i,'agb_bgb_t_ha']):
            input_df.at[i,'bgb_t_ha'] = input_df.at[i,'agb_t_ha'] * root_to_shoot
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
    input_df['agb_tCO2e_ha'] = input_df['agb_t_ha'] * biomass_to_c * c_to_co2
    input_df['bgb_tCO2e_ha'] = input_df['bgb_t_ha'] * biomass_to_c * c_to_co2
    input_df['agb_bgb_tCO2e_ha'] = input_df['agb_bgb_t_ha'] * biomass_to_c * c_to_co2

    # get max from carbon density GEE -----------------
    pt = ee.Geometry.Point(lat, lng)
    buffered_pt = pt.buffer(distance=20000)
    biomass_image = ee.ImageCollection("NASA/ORNL/biomass_carbon_density/v1").first()

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

    # prepare data for curve fit -----------
    age = np.array(input_df['age']).reshape((input_df['age'].shape[0],1))
    agb_bgb_tco2_ha = input_df['agb_bgb_tCO2e_ha']

    y_max_array = np.ones_like(age) * y_max_agb_bgb
    x_data = np.concatenate((age, y_max_array), axis=1)

    # curve fit
    # find parameters k and p
    params, covar = curve_fit(f=curve_fun, xdata=x_data, ydata=agb_bgb_tco2_ha)

    # Generate 100 yr prediction ------------
    x_plot = np.arange(1,101,1).reshape((100,1))
    y_max_array_plot = np.ones_like(x_plot) * y_max_agb_bgb
    x_data_plot = np.concatenate((x_plot, y_max_array_plot), axis=1)

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