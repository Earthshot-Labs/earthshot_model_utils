# model_utilities
Utility functions for model development and deployment

## deepdive_automation
This folder contains an installable Python package called `deepdive_automation` with a number of functions for the science team to conduct deep dives to do carbon projects for specific projects.

**Installation:**
Download folder to your local workspace. In Terminal activate the virtual environment where you want to install the `deepdive_automation` package. Navigate to the `deepdive_automation` folder (in terminal). Pip install the package using `pip install deepdive_automation`. This should install the package into the activated virutal environment. Then in Python (jupyter notebook or elsewhere using the same virtual environment) you will be able to access the package using `import deepdive_automation`. Functions can be used as `deepdive_automation.wood_density_lookup(...)`.

You need to have the `ee` package working on python. If you are using a Jupyter notebook that means you need to have notebooks authenticated for GEE (more detailed instructions coming soon).

NOTE: you will need to download the GEZ2010 shapefile into a data folder as this is part of the gitignore file. Specific instructions on this are coming soon. Link to download shapefile is: https://storage.googleapis.com/fao-maps-catalog-data/uuid/2fb209d0-fd34-4e5e-a3d8-a13c241eb61b/resources/gez2010.zip and you will also need to download this file from Joe's IPCC Tier 1 work: https://github.com/Earthshot-Labs/science/blob/master/IPCC_tier_1/prediction/ipcc_table_intermediate_files/ipcc_tier1_all.csv (since it's a private repo I'm having issues automatically pulling the file from the function)

NOTE: when using the latest version the package is now called `model_utilities`. To install download and navigate to the folder `model_utilities` and `pip setup.py install`.

**Update:**
If you have a new version of the package ready to install, navigate to the folder as with the install and type `pip install --upgrade .`.

**Functions:**
`wood_density_lookup`
`curve_fun`
`logistic_fun`
`curve_fit_func`
`chave_allometry_height`
`mature_biomass_spawn`
`root_shoot_ipcc`
`chave_allometry_noheight`
`getNearbyMatureForestPercentiles`

**Examples:**
There is a notebook in the science repo with examples of usage for each function: https://github.com/Earthshot-Labs/science/blob/master/Ad-hoc_project_analyses/Deepdive_automation_examples.ipynb 

**Issues:**
- GEZ shapefile too large to host on github (need to figure out how to access it without uploading) -- add to `.gitignore`.
- Root to shoot ratio table from Joe is in private repo so can't access it through pandas url call -- download and add to `.gitignore`?? Or copy into this repo? But then we're not tracking changes ...

**Functions want to make:**
- allometry for palms (maybe function chooses right allometry based on whether it's a palm and what data columns you have)? Are these factored into Chave allometry or should they be separate?

**Future Improvements:**

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

