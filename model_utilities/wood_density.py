import geopandas as gpd
import pandas as pd
import shapely
import numpy as np


def readWoodDensityDatabase():
    try:
        # Try to read the local cache
        wood_db = pd.read_csv('data/GlobalWoodDensityDatabase.csv.zip', index_col=0)
    except:
        # Read Data from DRYAD
        wood_db_url = 'https://datadryad.org/stash/downloads/file_stream/94836'
        wood_db = pd.read_excel(wood_db_url, sheet_name='Data', index_col=0)
        
        # Simplify Headers
        wood_db.columns = ['family', 'binomial', 'wd_gcm3', 'region', 'ref_num']
        
        #Standardize Regions
        wood_db.loc[wood_db['region']=='NorthAmerica','region'] = 'North America'
        wood_db.loc[wood_db['region']=='South America (Tropical)','region'] = 'South America (tropical)'
        wood_db.loc[wood_db['region']=='South America (extratropical)','region'] = 'South America'
        wood_db.loc[wood_db['region']=='Africa (extratropical)','region'] = 'Africa'
        
        # Persist local file
        wood_db.to_csv('data/GlobalWoodDensityDatabase.csv.zip', index=False)
        
    return wood_db
        


def getWDDBRegion(lat, lng):
    # Read Countries and Continents shapefile
    world = gpd.read_file('data/naturalearth_lowres.shp.zip')
    
    # Get country and continent at point
    point = shapely.Point(lng,lat)
    location = world.loc[world.contains(point)].to_dict('list')
    country = location['name'][0]
    continent = location['continent'][0]

    if country in ('China', 'India', 'Madagascar', 'Mexico'):
        region = country
    elif country in ('Australia', 'Papua New Guinea') and isTropical:
        region = 'Australia/PNG (tropical)'
    else:
        # Continent-Based Split on Tropical / Not Tropical
        region = continent
        if abs(lat) < 23.5:
            region += ' (tropical)'
        
        # Special renaming cases
        if region == 'North America (tropical)':
            region = 'Central America (tropical)'
        elif region == 'Asia (tropical)':
            region = 'South-East Asia (tropical)'
        elif region == 'Asia':
            if lat > 30:
                region = 'China'
            else:
                region = 'South-East Asia'
        
    return region



# look up wood density from Zanne et al 2009: https://datadryad.org/stash/dataset/doi:10.5061/dryad.234
def getWoodDensity(species_list, lat, lng, default=None):
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

    wd = readWoodDensityDatabase()
    region = getWDDBRegion(lat, lng)
    regionMatch = (wd['region']==region)
    
    # Calculate default from all entries in region or use supplied value
    regionDensities = wd.loc[regionMatch, 'wd_gcm3']
    regionalStd = regionDensities.std()
    regional = {'mean': regionDensities.mean(), 'std': regionalStd, 'flag':5}
    
    if default:
        default = {'mean': default, 'std': 0, 'flag':0}
    else:
        default = regional
    
    
    flag = 0
    wood_den = [dict(default, binomial="default")]
    wood_den += [dict(regional, binomial="regional average")]
    
    for binomial in species_list:
        try:
            genus, species = binomial.split()
        except:
            genus = binomial.strip()
            
        speciesMatch = (wd['binomial']==binomial)
        if np.any(speciesMatch):
            if np.any(speciesMatch & regionMatch):
                selected = speciesMatch & regionMatch
                flag = 1
            else:
                selected = speciesMatch
                flag = 2
        else:
            genusMatch = wd['binomial'].str.startswith(genus)
            if np.any(genusMatch & regionMatch):
                selected = genusMatch & regionMatch
                flag = 3
            elif np.any(genusMatch):
                selected = genusMatch
                flag = 4
            else:
                flag = 5            
            
        if flag==0:
            wood_den += [dict(default, binomial=binomial)]
        if flag==5:
            wood_den += [dict(regional, binomial=binomial)]
        else:
            matches = wd.loc[selected, 'wd_gcm3']
            wood_den += [{'binomial':binomial, 'mean': matches.mean(), 'std': matches.std(), 'flag':flag}]
            

    wood_den = pd.DataFrame(wood_den)
    wood_den['std'] = wood_den['std'].replace(np.nan, 0)

    return wood_den
    

    
if __name__ == '__main__':
    species = ['Lagerstroemia calyculata', 'Lagerstroemia crispa', 'Vitex ajugaeflora',
    'Xylia xylocarpa', 'Dalbergia oliveri', 'Shorea siamensis', 'Diospyros maritima',
    'Terminalia chebula', 'Terminalia triptera', 'Microcos paniculata', 'Dipterocarpus alatus',
    'Dipterocarpus obtusifolius', 'Adina polycephala', 'Adina pilulifra', 'Pterocarpus macrocarpus',
    'Nephelium sp.', 'Irvingia malayana', 'Peltophorum pterocarpum', 'Bauhinia sp.',
    'Antheroporum pierrei', 'Hopea recopei', 'Shorea roxburghii', 'Dillenia scabrella',
    'Cratoxylon pruniflorum', 'Diospyros sp.', 'Syzygium sp.', 'Syzygium oblatum',
    'Cephalanthus tetrandra', 'Anisoptera costata', 'Careya arborea', 'Mangifera minitifolia',
    'Sapindus saponaria', 'Xylia xylocarpa', 'Shorea obtusa', 'Cryptocarya petelotii',
    'Terminalia alata', 'Dipterocarpus intricatus', 'Adina pilulifra', 'Irvingia malayana',
    'Morinda citrifolia', 'Cratoxylon pruniflorum', 'Dalbergia cochinchinensis',
    'Syzygium sp.', 'Careya arborea']
    WD = getWoodDensity(species, 13.4, 107, default=0.565)
    print(WD)

