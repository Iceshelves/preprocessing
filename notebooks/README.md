# notebooks

This folder contains all the notebooks that have been used to develop and test the various steps of the preprocessing 
pipeline:

* [`copy_tiles_GCS-to-dCache.ipynb`](./copy_tiles_GCS-to-dCache.ipynb) contains a prototype of the script employed to 
download the tiles from bucket on Google Cloud Storage (where they have been exported from the Google Earth Engine) to 
the SURF dCache storage. The scripts actually employed on Spider are in the [`scripts`](../scripts) folder.
  
* [`create_STAC_catalog.ipynb`](./create_STAC_catalog.ipynb) is the notebook that has been employed to generate the STAC
catalog for the tiles downloaded on dCache ([`create_STAC_catalog_tutorial.ipynb`](./create_STAC_catalog_tutorial.ipynb)
is the prototype and contains more description and documentation of each steps).

* The following notebooks have been used to test different approaches for generating cutouts from the tiles:
    * [`window_loading_rolling.ipynb`](./window_loading_rolling.ipynb) shows how to achieve this with the `xarray`'s 
    `rolling` method.
    * [`window_loading_list.ipynb`](./window_loading_list.ipynb) shows how to achieve this with creating a list of `rasterio`'s `Window` objects.

* [`data_manipulation_sentinel1-2_example.ipynb`](./data_manipulation_sentinel1-2_example.ipynb) shows how to manipulate S1 data and/with S2 data with `rioxarray`.
    
 
     
  