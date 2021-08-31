# sky-estimation-WP3.7/measureMetrics
Repository for software used to do photometry on model images and output sky subtraction metrics.


## PYTHON CODE

\_\_init\_\_.py : blank file to turn directory into Python module

image_retrieval.py : functions for retrieving image cutouts at specific coordinates using the Butler

plot_metrics.py : functions for reading in and parsing photometry output tables and measuring/plotting metrics

surface_photometry.py : functions for doing surface photometry on images

utility.py : miscellaneous useful functions


## JUPYTER NOTEBOOKS

CataloguePhotometry.ipynb : demonstration of how to make photometry pickle tables for the full catalogues

Overview.ipynb : broad overview of injected models and analysis thereof


## TABLES

mags*.p : output from CataloguePhotometry.ipynb, photometry tables of models pre-sky-subtraction

coadd_mags*.p : output from CataloguePhotometry.ipynb, photometry tables of models post-sky-subtraction

lsstuk_icl_dwarfs.fits : table of parameters for the ICL + New Horizon dwarf model input catalogue

lsstuk_lsb_sersic.fits : table of parameters for the single-Sersic profile model input catalogue
