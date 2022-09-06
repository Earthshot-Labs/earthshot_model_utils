import pandas as pd

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