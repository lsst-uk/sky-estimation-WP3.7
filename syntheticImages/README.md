# sky-estimation-WP3.7/syntheticImages
Repository for software used to create and process fully synthetic images, for testing different kinds of sky-subtraction methods


## PYTHON CODE

fakes/insert_fakes.py: classes and functions for creating mock images, with mock galaxies, stars (including functions to download real catalogue data), and to create mock sky patterns.

fakes/reduction_pipeline.py: functions to "reduce" the mock images in various ways, including modeling and removing skies, masking of sources, etc.

fakes/sersic.py: functions for manipulating Sersic profiles, including deriving useful quantities given different parameters (e.g. deriving effective SB given magnitude, Sersic index, and effective radius)

Unit tests are found in the tests directory.


## INSTALLATION

Install necessary Python packages using the following:

`pip3 install -r syntheticImages/requirements.txt`

