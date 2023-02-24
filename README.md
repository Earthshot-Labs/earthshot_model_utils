# model_utilities
Utility functions for geospatial model development

This repository contains an installable Python package called `model_utilities` with a wide range of functionality for:
- building training and testing datasets for geospatial machine learning models, from assets in google earth engine
- automated model training
- performing inference with trained models to create wall-to-wall maps

and more described below related to forest mensuration, that we plan to split into a separate repository in the future.

Improved documentation is coming soon. For now check out the `tests` directory for demonstrations. Please reach out to steve@earthshot.eco/margaux@earthshot.eco with questions, feedback, or ideas.

# Installation:
Download folder to your local workspace. In Terminal activate the virtual environment where you want to install the `model_utilities` package. Navigate to the `model_utilities` folder (in terminal). Pip install the package using `pip install .`. This should install the package into the activated virutal environment. Then in Python (jupyter notebook or elsewhere using the same virtual environment) you will be able to access the package using `import model_utilities`. Functions can be used as `model_utilities.forest_mensuration.clean_biomass_data(...)`.

You need to have the `ee` package working on python. If you are using a Jupyter notebook that means you need to have notebooks authenticated for GEE (more detailed instructions coming soon).

**NOTE:** you will need to download the GEZ2010 shapefile and IPCC parameters csv into a data folder as this is part of the gitignore file. Specific instructions on this are coming soon. Link to download shapefile is: https://storage.googleapis.com/fao-maps-catalog-data/uuid/2fb209d0-fd34-4e5e-a3d8-a13c241eb61b/resources/gez2010.zip and you will also need to download this file from Joe's IPCC Tier 1 work: https://github.com/Earthshot-Labs/science/blob/master/IPCC_tier_1/prediction/ipcc_table_intermediate_files/ipcc_tier1_all.csv (since it's a private repo I'm having issues automatically pulling the file from the function)

# Update Package:
If you have a new version of the package ready to install, navigate to the folder as with the install and type `pip install --upgrade .`. If you expect lots of changes install in editable mode using `pip install -e .`.

# Functions: 
(need to update this list). 
`wood_density_lookup`
`curve_fun`
`logistic_fun`
`curve_fit_func`
`chave_allometry_height`
`mature_biomass_spawn`
`root_shoot_ipcc`
`chave_allometry_noheight`
`getNearbyMatureForestPercentiles`

# Examples:
There is a notebook in the science repo with examples of usage for each function in the testing folder


# Issues:
- GEZ shapefile too large to host on github (need to figure out how to access it without uploading) -- add to `.gitignore`.
- Root to shoot ratio table from Joe is in private repo so can't access it through pandas url call -- download and add to `.gitignore`?? Or copy into this repo? But then we're not tracking changes ...


# Future Improvements:

ipcc:
- need a function get ecozone? Right now we're just figuring that out and manually entering into function

mature biomass spawn:
- make sure that buffer only extends within same ecozone (talk to Joe)
- figure out which percentiles to keep (eg 10%ile is often unrealistically low, maybe only keep 50%ile and above? same for Africa and Latin America?)

curve fit:
- uncertainty (effect of allometries)
- average plots of same age?? (remove this from cleaning step)
- function to show 'best fit' curve with each maximum as a plot (maybe curve fit returns several figures)
- don't show MC simulations where row 2 < row 1 (decreasing at the start)

wood density:
- consider more carefully how default SD of WD for n=1 is returned (now it's the mean SD of WD across the region). We don't want it to be 0 because it could be used to track uncertainty in the future. But the current default SD(WD) could be misleading because it's biased to what appeared in the literature?
- what other wood density databases are there? Include wood density in our custom DB?

allometry:
- allometry for palms (maybe function chooses right allometry based on whether it's a palm and what data columns you have)? Are these factored into Chave allometry or should they be separate?
- uncertainty from allometry? using Chave vs local allometry?

research (quality flags):
- when do UNR and ANR converge?
- what is the reasonable time to 90% of max?
- what is the maximum growth rate (slope) for any given location?

agroforestry:
- models specific to agroforestry systems

intervention-specific:
- effect of different species mixes
- effect of interventions (e.g. irrigation, weeding, fertilizer, soil ammendments, soil tilling, ...)
- effect of planting density (when does high planting density converge with other curves? max biomass would be the same regardless)

