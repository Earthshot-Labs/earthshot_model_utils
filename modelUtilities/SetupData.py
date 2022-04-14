###############################################################################
###############################################################################
## 
## 03/31/2022
##
## Written by: Meghan Blumstein (meghan@earthshot.eco) 
##
## Objective: Take raw poorter et al. 2016 data and put into pandas dataframe
##
## Inputs: raw poorter et al. 2016 data saved in Science goolgle drive
##
## Outputs: formatted pandas dataframe
##
###############################################################################
###############################################################################

def Format_Poorter(file_location):
  
  ## Get packages
  import pandas as pd
  import numpy as np

  ## Load Pooter data (with lat/lon coordinates)
  df = pd.read_csv(file_location)
  
  #---------------------------------------#
  ## Coerece age to numeric 
  ## [need to reclass "OG" codes]
  ## ** OG > 100 Years
  #---------------------------------------#
  
  ## Start by dropping old growth as we are not using it in these models
  df = df[df['Age']!= "OG"]
      
  ## Then coerce age a numeric
  df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
  
  #---------------------------------------#
  ## Create log column of AGB & Age 
  #---------------------------------------#
  
  ## First set anything that is 0 to 1 to avoid log(0) == -inf
  df.loc[df['AGB (Mg/ha)'] < 1, 'AGB (Mg/ha)'] = 0.01
  df.loc[df['Age'] < 1, 'Age'] = 1
     
  ## Log column
  df['log_AGB'] = np.log(df['AGB (Mg/ha)'])
  df['log_Age'] = np.log(df['Age'])
  
  #---------------------------------------#
  ## Remove Columbian Islands
  #---------------------------------------#
  
  ## Remove the Columbian island as can't get predictors for it and it looks
  ## like Poorter et al. 2016 also dropped this data 
  df = df[df['Chronosequence'] != 'Providencia Island']
  
  #---------------------------------------#
  ## Rename columns to remove spaces & characters
  ## and sort by age
  #---------------------------------------#
  
  df = df.rename(columns={'AGB (Mg/ha)':'AGB_Mg_ha'})
  df.sort_values(by=['Age'])
  
  return df













