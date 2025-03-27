# SystemFlow

This repository contains the code and data necessary to reproduce the data and plots for the paper "[Modeling Performance of Data Collection Systems for  High-Energy Physics](https://pubs.aip.org/aip/aml/article/2/4/046113/3326300)."

The required Python environment can be constructed in Conda using the included environment file. 

The illustrative figures 2, 3, and 4 were created in Adobe Illustrator and are not included as part of this repository. 

# Plots
To construct specific figures from the paper, run the following notebooks after constructing and activating the environment:

## cpu_scaling.ipynb
Includes figure 1 (Cell 52)

Note: ensure that the submodule containing historical processor data is downloaded. If necessary, run ``git clone --recursive git@github.com:wilkieolin/system_flow.git`` to clone the submodule.

## pileup_vs_l1t.ipynb
Includes figure 5 (Cell 43)

## vary_system.ipynb
Includes figure 6 (Cell 66)

# Tables
To reproduce the results summarized in tables 1 and 2, run **results_table.ipynb**. Table 1 is located in Cell 54 and table 2 is in Cell 70.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/wilkieolin/system_flow/HEAD?labpath=vary_hlt.ipynb)