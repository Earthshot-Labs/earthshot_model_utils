import pandas as pd
import numpy as np
import geopandas as gpd

import ee

try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

# constants
c_to_co2 = (44/12) #conversion factor c to co2 equivalent

def clean_biomass_data(input_df, d_type, rs_break=125, rs_young=0.285, rs_old=0.285, biomass_to_c=0.47):
    """
    Function to take dataframe of agb and possibly bgb. Fills in missing bgb according to supplied parameters.
    Averages measurements of the same age (how does this affect results?). Convert units to tCO2e/ha, based
    on units of input data.
    Parameters
    ----------
    input_df : [pandas dataframe]
                pandas dataframe with columns 'agb_t_ha', 'bgb_t_ha', 'agb_bgb_t_ha' for the biomass data at location
    d_type : [string]
            'biomass' if input_df contains biomass in t/ha
            'carbon' if input_df contains carbon in t/ha
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
    ---------
    input_df : the dataframe that was cleaned inplace
    """
    # fill in missing bgb, agb+bgb ------------
    for i in range(0, input_df.shape[0]):

        # if have agb but not bgb or agb+bgb ... use root-to-shoot to get bgb ... agb+bgb is sum of cols 2,3
        if pd.notna(input_df.at[i, 'agb_t_ha']) & pd.isna(input_df.at[i, 'bgb_t_ha']) & pd.isna(
                input_df.at[i, 'agb_bgb_t_ha']):
            # if agb > rs_break then use rs_old, else use rs_young
            if input_df.at[i, 'agb_t_ha'] > rs_break:
                input_df.at[i, 'bgb_t_ha'] = input_df.at[i, 'agb_t_ha'] * rs_old
            else:
                input_df.at[i, 'bgb_t_ha'] = input_df.at[i, 'agb_t_ha'] * rs_young
            input_df.at[i, 'agb_bgb_t_ha'] = input_df.at[i, 'agb_t_ha'] + input_df.at[i, 'bgb_t_ha']

        # if have agb and bgb but not agb+bgb ... sum cols 2,3
        elif pd.notna(input_df.at[i, 'agb_t_ha']) & (pd.notna(input_df.at[i, 'bgb_t_ha'])) & pd.isna(
                input_df.at[i, 'agb_bgb_t_ha']):
            input_df.at[i, 'agb_bgb_t_ha'] = input_df.at[i, 'agb_t_ha'] + input_df.at[i, 'bgb_t_ha']

    # average plots of same age
    #input_df = input_df.groupby(['age']).agg({'agb_t_ha': 'mean',
    #                                          'bgb_t_ha': 'mean',
    #                                          'agb_bgb_t_ha': 'mean'})
    #input_df.reset_index(drop=False, inplace=True)

    # convert biomass to CO2e -----------------
    if (d_type == 'biomass'):
        input_df['agb_tCO2e_ha'] = input_df['agb_t_ha'] * biomass_to_c * c_to_co2
        input_df['bgb_tCO2e_ha'] = input_df['bgb_t_ha'] * biomass_to_c * c_to_co2
        input_df['agb_bgb_tCO2e_ha'] = input_df['agb_bgb_t_ha'] * biomass_to_c * c_to_co2
    else:
        input_df['agb_tCO2e_ha'] = input_df['agb_t_ha'] * c_to_co2
        input_df['bgb_tCO2e_ha'] = input_df['bgb_t_ha'] * c_to_co2
        input_df['agb_bgb_tCO2e_ha'] = input_df['agb_bgb_t_ha'] * c_to_co2

    return input_df


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
    wood_db.columns = ['family', 'binomial', 'wd_gcm3', 'region', 'ref_num']
    wood_db.region.unique()  # regions are tricky, maybe input lat, lng and it spits out correct region

    # get geometries of world countries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # get coordinates of specific site
    # lng, lat = -42.769961, -22.453611 test values, actually get from func input
    df = pd.DataFrame({'name': ['point1'],
                       'lat': lat,
                       'lng': lng})
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat))
    gdf = gdf.set_crs(world.crs)

    pip = gpd.sjoin(gdf, world)

    # get list of north american countries that are in central america (minus mexico b/c tracked separately)
    countries_na_notcentral = ['United States of America', 'Canada', 'Mexico']

    # filter wood density by region based on lat lon
    # tropics are lat between -23.5 and 23.5
    if pip.loc[0, 'name_right'] == 'China':
        # filter wood den for region = 'China'
        wood_db_here = wood_db[wood_db.region == 'China']
    elif pip.loc[0, 'name_right'] == 'India':
        # filter wood den for region = 'India'
        wood_db_here = wood_db[wood_db.region == 'India']
    elif pip.loc[0, 'name_right'] == 'Mexico':
        # filter wood den for region = 'Mexico'
        wood_db_here = wood_db[wood_db.region == 'Mexico']
    elif ((pip.loc[0, 'name_right'] == 'Australia') & (pip.loc[0, 'lat'] < -23.5)):
        # filter wood den for region = 'Australia'
        wood_db_here = wood_db[wood_db.region == 'Australia']
    elif ((pip.loc[0, 'name_right'] == 'Australia') & (pip.loc[0, 'lat'] >= -23.5) & (pip.loc[0, 'lat'] <= 23.5)):
        # filter wood den for region = 'Australia/PNG (tropical)'
        wood_db_here = wood_db[wood_db.region == 'Australia/PNG (tropical)']
    elif (pip.loc[0, 'name_right'] == 'Madagascar'):
        # filter wood den for region = 'Madagascar'
        wood_db_here = wood_db[wood_db.region == 'Madagascar']
    elif ((pip.loc[0, 'continent'] == 'South America') & (pip.loc[0, 'lat'] >= -23.5) & (pip.loc[0, 'lat'] <= 23.5)):
        # filter wood den for region = South America (tropical) & South America (Tropical)
        wood_db_here = wood_db[
            (wood_db.region == 'South America (tropical)') | (wood_db.region == 'South America (Tropical)')]
    elif ((pip.loc[0, 'continent'] == 'South America') & ((pip.loc[0, 'lat'] < -23.5) | (pip.loc[0, 'lat'] > 23.5))):
        # filter wood den for region = South America (extratropical)
        wood_db_here = wood_db[wood_db.region == 'South America (extratropical)']
    elif ((pip.loc[0, 'continent'] == 'North America') & (pip.loc[0, 'name_right'] not in countries_na_notcentral) & (
            pip.loc[0, 'lat'] >= -23.5) & (pip.loc[0, 'lat'] <= 23.5)):
        # filter wood den for region = Central America (tropical)
        wood_db_here = wood_db[wood_db.region == 'Central America (tropical)']
    elif (pip.loc[0, 'continent'] == 'Europe'):
        # filter wood den for region = Europe
        wood_db_here = wood_db[wood_db.region == 'Europe']
    elif ((pip.loc[0, 'name_right'] == 'United States of America') | (pip.loc[0, 'name_right'] == 'Canada')):
        # filter wood den for region = NorthAmerica -- assuming that's USA + Canada b/c other countries in different categories
        wood_db_here = wood_db[wood_db.region == 'NorthAmerica']
    elif ((pip.loc[0, 'continent'] == 'Africa') & (pip.loc[0, 'lat'] >= -23.5) & (pip.loc[0, 'lat'] <= 23.5)):
        # filter wood den for region = Africa (tropical)
        wood_db_here = wood_db[wood_db.region == 'Africa (tropical)']
    elif ((pip.loc[0, 'continent'] == 'Africa') & ((pip.loc[0, 'lat'] < -23.5) | (pip.loc[0, 'lat'] > 23.5))):
        # filter wood den for region = Africa (extratropical)
        wood_db_here = wood_db[wood_db.region == 'Africa (extratropical)']
    elif ((pip.loc[0, 'continent'] == 'Asia') & (pip.loc[0, 'lat'] >= -23.5) & (pip.loc[0, 'lat'] <= 23.5)):
        # filter wood den for region = South-East Asia (tropical)
        wood_db_here = wood_db[wood_db.region == 'South-East Asia (tropical)']
    elif ((pip.loc[0, 'continent'] == 'Asia') & ((pip.loc[0, 'lat'] > 23.5) | (pip.loc[0, 'lat'] < -23.5))):
        # filter wood den for region = South-East Asia -- assuming that's extratropical
        wood_db_here = wood_db[wood_db.region == 'South-East Asia']
    elif (pip.loc[0, 'continent'] == 'Oceana'):
        # filter wood den for region = Oceania
        wood_db_here = wood_db[wood_db.region == 'Oceania']

    # get standard variance for species that only have 1 measurement
    standard_std = wood_db_here.groupby(['binomial'])['wd_gcm3'].agg('std').mean()

    # filter species and get species mean
    # species_list = ['Abarema jupunba','Zygia latifolia'] for testing
    wood_db_here = wood_db_here[wood_db_here.binomial.isin(species_list)]
    # wood_den = wood_db_here.groupby(['binomial']).mean('wd_gcm3')[['wd_gcm3']] #used when only got mean
    wood_den = wood_db_here.groupby(['binomial'])['wd_gcm3'].agg(['mean', 'std', 'count'])
    # replace missing STD with the mean STD for that region
    wood_den['std'] = wood_den['std'].replace(np.nan, standard_std)
    return wood_den

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
    return WD * np.exp(-2.977 + np.log(WD * DBH ** 2 * H))


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
        pt = ee.Geometry.Point(lng, lat)  # x,y
        roi = pt.buffer(distance=100)

    # Environmental stress factor on the diameter-height tree allometry
    environmental_stress_factor = ee.Image(
        "projects/ee-margauxmasson-madre-de-dios/assets/Environmental_stress_factor_chave")  # E equation global gridded layer of E at 2.5 arc sec resolution

    # Taking the mean environmental_stress_factor over the MDD region
    sample_environmental_stress_factor = environmental_stress_factor.reduceRegion(
        geometry=roi,
        reducer=ee.Reducer.mean(),
        scale=300)

    data_dict_environmental_stress_factor = sample_environmental_stress_factor.getInfo()
    E = data_dict_environmental_stress_factor['b1']  # only one band

    # calculate biomass
    AGB_kg = np.exp(-1.803 - 0.976 * E + 0.976 * np.log(WD) + 2.673 * np.log(DBH) - 0.0299 * (np.log(DBH) ** 2))
    return AGB_kg