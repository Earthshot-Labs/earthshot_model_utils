# model_utilities
Utility functions for model development and deployment

## deepdive_automation
This folder contains an installable Python package called `deepdive_automation` with a number of functions for the science team to conduct deep dives to do carbon projects for specific projects.

**Installation:**
Download folder to your local workspace. In Terminal activate the virtual environment where you want to install the `deepdive_automation` package. Navigate to the `deepdive_automation` folder (in terminal). Pip install the package using `pip install deepdive_automation`. This should install the package into the activated virutal environment. Then in Python (jupyter notebook or elsewhere using the same virtual environment) you will be able to access the package using `import deepdive_automation`. Functions can be used as `deepdive_automation.wood_density_lookup(...)`.

NOTE: you will need to download the GEZ2010 shapefile into a data folder as this is part of the gitignore file. Specific instructions on this are coming soon.

**Functions:**
`wood_density_lookup`
`curve_fun`
`curve_fit_func`
`chave_allometry_height`
`mature_biomass_spawn`
`root_shoot_ipcc`

**Issues:**
- GEZ shapefile too large to host on github (need to figure out how to access it without uploading) -- add to `.gitignore`.

**Functions want to make:**
- Chave allometry (without height) -- need E layer from Margaux (or how to access it on GEE)
- allometry for palms (maybe function chooses right allometry based on whether it's a palm and what data columns you have)? Are these factored into Chave allometry or should they be separate?
- ymax from Margaux's mature biomass code (coming later this summer)

**To consider:**

mature biomass spawn:
- make sure that buffer only extends within same ecozone (talk to Joe)

curve fit:
- uncertainty (MC simulation, picking ymax, effect of allometries)
- average plots of same age??
- replace code for ymax with Margaux's piece
- better way to pick forest type for IPCC root to shoot ratio

wood density:
- could also return n values that were averaged from and SD of WD values (figure out default WD to return if n=1)
