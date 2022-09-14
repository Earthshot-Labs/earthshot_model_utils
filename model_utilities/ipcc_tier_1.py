import pandas as pd

def _expandIPCC(years, r0, r20, k20, kmax, root_shoot_break, rs_young, rs_old):
    # Aboveground Biomass (proposed fix above)
    ## Biomass = np.minimum(k20, years*r0) + np.maximum(0, np.minimum(kmax, (years-20)*r20))
    year20cap = np.minimum(k20, 20 * r0)
    yearsPast20 = np.maximum(0, years - 20)
    Biomass = np.minimum(year20cap, years * r0) + yearsPast20 * r20
    Biomass = np.minimum(kmax, Biomass)

    # Add belowground biomass
    Biomass *= 1 + rs_old + (rs_young - rs_old) * (Biomass < root_shoot_break)

    # Convert from dry biomass -> Carbon -> CO2e
    Biomass *= 0.47 * 3.67
    Biomass = np.round(Biomass, 2)

    return Biomass


def PredictIPCC(polygonData=None, ecozone=None, continent=None, forest_type=None, yearStart=0, yearEnd=30):
    # Determine ecozone and continent, either from boundaries or passed to function
    if not ecozone or not continent:
        if not polygonData:
            ret['valid'] = 0
            ret['msg'] = "Parcel boundaries or both ecozone and continent must be specified"
            return ret

        zone = _getEcozoneFromPolygons(polygonData)
        ecozone = zone['ecozone'].lower()
        continent = zone['continent'].lower()

    # Read datafile
    ipcc = pd.read_csv('./data/IPCC_Tier1_parameters.csv')
    params = ipcc.loc[(ipcc['ecozone'] == ecozone.lower()) &
                      (ipcc['continent'] == continent.lower())]

    # Return if Invalid Ecozone / Contintent combination
    if params.empty:
        ret['valid'] = 0
        ret['msg'] = "Ecozone and Continent combination not found"
        return ret

    # Select forest type if provided
    if forest_type:
        params = params.loc[params['forest_type'] == forest_type.lower()]

    # Return if forest type not found
    if params.empty:
        ret['valid'] = 0
        ret['msg'] = f"No forest type '{forest_type}' available for {ecozone} and {continent}"
        return ret

    # Calculate low, median, and high biomass for specified year range for each forest type
    ret = {'valid': 1,
           'msg': '',
           'ecozone': ecozone,
           'continent': continent,
           'start_year': yearStart,
           'end_year': yearEnd,
           'units': 'tCO2e per reforestable ha',
           'predictions': {},
           }

    years = np.array(range(yearStart, yearEnd + 1))
    for r in params.to_dict('records'):
        Med = _expandIPCC(years, r['r0'], r['r20'], r['K20'], r['Kmax'],
                          r['root_shoot_break'], r['rs_young'], r['rs_old'])
        Low = _expandIPCC(years, r['r0_low'], r['r20_low'], r['K20_low'], r['Kmax_low'],
                          r['root_shoot_break'], r['rs_young'], r['rs_old'])
        High = _expandIPCC(years, r['r0_high'], r['r20_high'], r['K20_high'], r['Kmax_high'],
                           r['root_shoot_break'], r['rs_young'], r['rs_old'])

        ret['predictions'][r['forest_type']] = {
            'biomass': Med,
            'biomassLow': Low,
            'biomassHigh': High,
            'uncertainty_method_cap': r['cap_uncertainty_type'],
            'uncertainty_method_rate': r['rate_uncertainty_type']
        }

    return ret

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
    root_shoot_ipcc = pd.read_csv(
        'https://raw.githubusercontent.com/Earthshot-Labs/science/master/IPCC_tier_1/prediction/ipcc_table_intermediate_files/ipcc_tier1_all.csv?token=GHSAT0AAAAAABQWL3QREMVXR567IWPOZF22YV5W4YQ')

    # open gez2010 shapefile for Global Ecoological Zones
    dir_here = os.getcwd()
    gez_shp = dir_here + '/deepdive_automation/gez2010/gez_2010_wgs84.shp'
    gez_gdf = gpd.read_file(gez_shp)

    # Get GEZ
    point_xy = [[lng], [lat]]
    poi = gpd.points_from_xy(x=point_xy[0], y=point_xy[1])

    idx = gez_gdf['geometry'].contains(poi[0])
    eco_type = gez_gdf.loc[idx, 'gez_name'].values[0].lower()

    # get geometries of world countries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    idx = world['geometry'].contains(poi[0])
    continent = world.loc[idx, 'continent'].values[0].lower()

    # get relevant rows
    rs_rows = root_shoot_ipcc[(root_shoot_ipcc.continent == continent) &
                              (root_shoot_ipcc.nump == eco_type) &
                              (root_shoot_ipcc.forest_type == for_type)]

    # if multiple rows then average values???
    root_shoot_break = rs_rows['root_shoot_break'].mean()
    rs_young = rs_rows['rs_young'].mean()
    rs_old = rs_rows['rs_old'].mean()

    return root_shoot_break, rs_young, rs_old