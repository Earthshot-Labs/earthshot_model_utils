# model_utilities
Utility functions for model development and deployment

## deepdive_automation
This folder contains an installable Python package called `deepdive_automation` with a number of functions for the science team to conduct deep dives to do carbon projects for specific projects.

**Functions:**
`wood_density_lookup`
`curve_fun`
`curve_fit_func`
`chave_allometry_height`
`mature_biomass_spawn`
`root_shoot_ipcc`

**Issues:**
- GEZ shapefile too large to host on github (need to figure out how to access it without uploading)

**Functions want to make:**
- Chave allometry (without height) -- need E layer from Margaux (or how to access it on GEE)
- allometry for palms (maybe function chooses right allometry based on whether it's a palm and what data columns you have)
- ymax from Margaux's code

**To consider:**

mature biomass spawn:
- make sure that buffer only extends within same ecozone

curve fit:
- uncertainty (MC simulation, picking ymax, effect of allometries)
- average plots of same age??
- replace code for ymax with Margaux's piece
- better way to pick forest type for IPCC root to shoot ratio

wood density:
- choose correct root-to-shoot ratio from lat/lng (or enter biome in func) and above/below threshold in IPCC Tier 1
- could also return n values that were averaged from 
