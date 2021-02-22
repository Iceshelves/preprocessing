#!/usr/bin/env python
# coding: utf-8

# import required packages

import pathlib
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import rioxarray as rioxr
import pandas as pd
import geopandas as gpd
import tensorflow
from tensorflow import keras
import VAE
import datetime
import geopandas as gpd
import pystac
from stac2webdav.utils import catalog2geopandas


"""
Data ingestion routines.
Read catalog of tiles/assets
Read tile paths from STAC compliant catalog.
Read label geojson files to produce labeled test data.

"""

def read_config(config):
    if pathlib.Path(config).is_file():
        raise NotImplementedError()
        
    else :
        catPath = config['catalogPath']
        labPath = config['labelsPath']
        outputDir = config['outputDirectory']
        sizeTestSet = config['sizeTestSet']
        valSplitFrac = config['validationSplitFraction']
        roiFile = config['ROIFile']
        bands = config['bands']
        sizeCutOut = config['sizeCutOut']
        nEpochMax = config['nEpochMax']
        sizeStep = config['sizeStep']
        
    return catPath, labPath, outputDir, sizeTestSet, valSplitFrac, roiFile, bands, sizeCutOut, nEpochmax, sizeStep


def read_tile_catalog(catalog_path):
    """ Read tile catalog """
    catalog_path = pathlib.Path(catalog_path)
    catalog_path = catalog_path / "catalog.json"
    return pystac.Catalog.from_file(catalog_path.as_posix())


def read_labels(labels_path):
    """ Read all labels, and merge them in a single GeoDataFrame """
    labels_path = pathlib.Path(labels_path)
    labels = [gpd.read_file(p) for p in labels_path.glob("*.geojson")]
    crs = labels[0].crs
    assert all([l.crs == crs for l in labels])
    labels = pd.concat(labels).pipe(gpd.GeoDataFrame)
    return labels.set_crs(crs)


def get_asset_paths(catalog, item_ids, asset_key):
    """ Extract the asset paths from the catalog """
    items = (catalog.get_item(id, recursive=True) for id in item_ids)
    assets = (item.assets[asset_key] for item in items)
    return [asset.get_absolute_href() for asset in assets]



"""
defintition of Dataset object providing data for ingestion to VAE
"""

class Dataset:
    
    def __init__(self, tile_list, cutout_size, bands, offset=0, stride=None, num_tiles=None, shuffle_tiles=False):
        self.cutout_size = cutout_size
        self.bands = bands
        self.stride = stride if stride is not None else self.cutout_size
        _num_tiles = num_tiles if num_tiles is not None else len(tile_list)
        self.tiles = random.sample(tile_list, _num_tiles) if shuffle_tiles else tile_list[:_num_tiles]
        if offset >= cutout_size:
            raise ValueError(
                "offset larger than window size - set "
                "offset to {}".format(sizeCutOut%offset)
            )
        self.offset = offset
        
        self.mask = None
        self.buffer = None
        self.invert = None
        self.all_touched = None
    
    def set_mask(self, geometry, crs, buffer=None, invert=False, all_touched=False):
        """ Mask a selection of the pixels using a geometry."""
        self.mask = gpd.GeoSeries({"geometry": geometry}, crs=crs)
        self.buffer = buffer if buffer is not None else 0
        self.invert = invert
        self.all_touched = all_touched
    
    def to_tf(self):
        """ Obtain dataset as a tensorflow `Dataset` object. """
        ds = tf.data.Dataset.from_generator(
            self._generate_cutouts,  
            output_types=(tf.float64, tf.float64, tf.float32), 
            output_shapes=(
                None,  # x
                None,  # y
                (None, None, self.cutout_size, self.cutout_size)  # samples, bands, x_win, y_win
            )
        )
        return ds.flat_map(lambda x,y,z: tf.data.Dataset.from_tensor_slices((x,y,z)))

    def _generate_cutouts(self):
        """ 
        Iterate over (a selection of) the tiles yielding all 
        cutouts for each of them.
        """
        for tile in self.tiles:
            
            print(f"Reading tile {tile}!")
            
            # read tile
            da = rioxr.open_rasterio(tile).astype("float32")  # needed to mask with NaN's
            
            #select bands
            if self.bands is not None:
                if type(self.bands) is not list:
                    da = da.sel(band=[self.bands])
                else:
                    da = da.sel(band=self.bands)
                    
                
            
            if self.mask is not None:
                mask = self.mask.to_crs(da.spatial_ref.crs_wkt)
                geometry = mask.unary_union.buffer(self.buffer)
                da = da.rio.clip([geometry], drop=True, invert=self.invert, all_touched=self.all_touched)
            
            # apply offset
            da = da.shift(x=self.offset, y=self.offset)  # only shift data, not coords
            da['x'] = da.x.shift(x=self.offset)
            da['y'] = da.y.shift(y=self.offset)

            # generate windows
            da = da.rolling(x=self.cutout_size, y=self.cutout_size)
            da = da.construct({'x': 'x_win', 'y': 'y_win'}, stride=self.stride)

            # drop NaN-containing windows
            da = da.stack(sample=('x', 'y'))
            da = da.dropna(dim='sample', how='any')
            yield (da.sample.coords['x'], 
                   da.sample.coords['y'], 
                   da.data.transpose(3, 1, 2, 0))  # samples, x_win, y_win, bands


"""
Begin data ingestion and initial processing

"""

"""
1. Input config
"""
config = {'catalogPath':"./S2_composite_catalog",
		  'labelsPath':"./labels",
		  'outpuDir':'~/output',
		  'sizeTestSet':12,
		  'valSplitFrac':0.3,
		  'roiFile':"./ne_10m_antarctic_ice_shelves_polys/ne_10m_antarctic_ice_shelves_polys.shp",
		  'bands':[1,2,3],
		  'sizeCutOut':20,
		  'nEpochMax':2,
		  'sizeStep':5}
	

catPath, labPath, outputDir, sizeTestSet, valSplitFrac, roiFile, _, sizeCutOut, nEpochmax, sizeStep = read_config(config)



"""
2. Asset ingestion
"""

"""
2.1 Tiles
"""
# read tile catalog
catalog = read_tile_catalog(catPath)
tiles = catalog2geopandas(catalog)


"""
2.2 lebels and construction of labeled data
"""
# read labels 
labels = read_labels(labPath)

label_dates = pd.to_datetime(labels.Date)
start_datetime = pd.to_datetime(tiles.start_datetime).min() 
end_datetime = pd.to_datetime(tiles.end_datetime).max() 
mask = (label_dates >= start_datetime) & (label_dates <= end_datetime)
labels = labels[mask]


"""
3. Construction of test, train, validate datasets
"""
"""
3.1 test. This includes the labeled data
"""
# reserve the labeled tiles for the test set
mask = tiles.intersects(labels.unary_union)
test_set_labeled = tiles[mask]

# pick additional unlabeled tiles for the test set
n_tiles_labeled = len(test_set_labeled)
n_tiles_unlabeled = sizeTestSet - n_tiles_labeled
test_set_unlabeled = tiles[~mask].sample(n_tiles_unlabeled)

# split test set and training/validation set
test_set = pd.concat([test_set_labeled, test_set_unlabeled])

"""
3.2 train/validation set
"""
train_set = tiles.index.difference(test_set.index) # tiles not in test set
train_set = tiles.loc[train_set]

#split off validation set
# split training set and validation set
val_set_size = round(valSplitFrac*len(train_set))
val_set = train_set.sample(val_set_size)
train_set = train_set.index.difference(val_set.index)
train_set = tiles.loc[train_set]


"""
4. Construction of Dataset objects for all three subsets
""" 
# extract tile paths from the catalog
test_set_paths = get_asset_paths(catalog, test_set.index, "B2-B3-B4-B11")
train_set_paths = get_asset_paths(catalog, train_set.index, "B2-B3-B4-B11")
val_set_paths = get_asset_paths(catalog, val_set.index, "B2-B3-B4-B11")

#Iceshelf extent as intital balancing

mask = gpd.read_file(roiFile)

"""
I think we could also balance the test set using the mask,
t.b.h. as it is a preselection step across all data
"""

# no balancing in the test set (i.e. don't apply mask)
test_set = Dataset(test_set_paths, sizeCutOut, bands, shuffle_tiles=True)
test_set_tf = test_set.to_tf()

# balanced validation set (i.e. apply mask)
val_set = Dataset(val_set_paths, sizeCutOut, bands, shuffle_tiles=True)
val_set.set_mask(mask.unary_union, crs=mask.crs)
val_set_tf = val_set.to_tf()

#batch data sets
# val_set_tf = val_set_tf.shuffle(buffer) # if we use a subset, we should probably shuffle this
val_set_tf = val_set_tf.batch(64, drop_remainder=True)


# test_set_tf = test_set_tf.shuffle(buffer) # if we use a subset, we should probably shuffle this
test_set_tf = test_set_tf.batch(64, drop_remainder=True)


"""
5. Loop and feed to VAE
"""

epochCounter = 1 # start at 0 or adjust offset calculation

# using datetime module for naming the current model, so that old models do not get overwritten
import datetime;   
# ct stores current time 
ct = datetime.datetime.now() 
# ts store timestamp of current time 
ts = ct.timestamp()

#make vae 

encoder_inputs, encoder, z , z_mean, z_log_var = VAE.make_encoder()
decoder  = VAE.make_decoder()
vae = VAE.make_vae(encoder_inputs, z, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
path = os.path.join(outputDir, '/model_' + str(int(ts))) 
vae.save(os.path.join(path ,'_epoch_' + str(epochCounter -1) ))

"""
begin loop
"""

while epochcounter < nEpochsMax:
    offset = (epochcounter -1)*sizeStep
 
    train_set = Dataset(train_set_paths, sizeCutOut, offset=offset,
                        shuffle_tiles=True)
    train_set.set_mask(mask.unary_union, crs=mask.crs)
    train_set_tf = train_set.to_tf()

    """
    this buffersize sould correpond to 6 full tiles, 
    so 7 in total in memory
    """ 
    train_set_tf = train_set_tf.shuffle(buffersize=3000000).batch(64, drop_remainder=True)
    #inData = Dataset(inputList,sizeCutOut,offset=offset,shuffle_tiles=True)
    #inData.set_mask(gdf.unary_union, crs=gdf.crs)
    #dataSet = inData.to_tf()

    """
    have to rethink normalization
    """
    #normalization
    #data = dataSet # should be split in train and test
    #imax = np.max(data) # 15.000-ish
    #data = (data+0.1)/ (imax+1)

    
    vae =  keras.models.load_model(path + '_epoch_' + str(epochCounter -1))# change it: make a call to os to create a path
    
    vae.fit(train_set_tf, epochs=1, validation_data = val_set_tf)
    vae.save(os.path.join(path,'_epoch_' + str(epochCounter))   # change it: make a call to os to create a path
         
             
    #repeat from here        
    epochcounter = epochcounter + 1
