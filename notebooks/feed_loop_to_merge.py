#!/usr/bin/env python
# coding: utf-8

# import required packages

# In[1]:


import pathlib
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import rioxarray as rioxr
import geopandas as gpd
import tensorflow
from tensorflow import keras
import VAE


# In[ ]:


config = {'inputDirectory':'~/input',
 'outputDirectory':'~/output',         
 'bands':[1,2,3],
 'sizeCutOut':20,
 'nEpochMax':2,
 'sizeStep':5}


# In[ ]:


def create_input_list(inputDir):
    inputList = [t for t in pathlib.Path(inputDir).glob('*.tif')]
    return inputList


# In[ ]:


def read_config(config):
    if pathlib.Path(config).is_file():
        raise NotImplementedError()
        
    else :
        inputDir = config['inputDirectory']
        outputDir = config['outputDirectory']
        bands = config['bands']
        sizeCutOut = config['sizeCutOut']
        nEpochMax = config['nEpochMax']
        sizeStep = config['sizeStep']
        
    return inputDir, outputDir, bands, sizeCutOut, nEpochmax, sizeStep


# define Dataset class and methods

# In[ ]:


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


# ### Begin feedloop

# In[ ]:


inputDir, outputDir, _, sizeCutOut, nEpochmax, sizeStep = read_config(config)


# In[ ]:


inputList = create_input_list(inputDir)


# In[ ]:


epochCounter = 1


# In[ ]:


gdf = gpd.read_file("ne_10m_antarctic_ice_shelves_polys/ne_10m_antarctic_ice_shelves_polys.shp")


# In[ ]:



# using datetime module for naming the current model, so that old models do not get overwritten
import datetime;   
# ct stores current time 
ct = datetime.datetime.now() 
# ts store timestamp of current time 
ts = ct.timestamp() 


# In[ ]:



#make vae 

encoder_inputs, encoder, z , z_mean, z_log_var = VAE.make_encoder()
decoder  = VAE.make_decoder()
vae = VAE.make_vae(encoder_inputs, z, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
path = os.path.join(outputDir, '/model_' + str(int(ts))) 
vae.save(path + '_epoch_' + str(epochCounter -1) )# change it: make a call to os to create a path


# In[ ]:


#test_data # should NOT be balanced
#validation_data = .....# should be balanced 

while epochcounter < nEpochsMax:
    offset = epochcounter*sizeStep
    #randomly permutate inputList for each epoch
    #inputList = list(np.random.permutation(inputList))
    #construct Dataset
    inData = Dataset(inputList,sizeCutOut,offset=offset,shuffle_tiles=True)
    inData.set_mask(gdf.unary_union, crs=gdf.crs)
    dataSet = inData.to_tf()
    
    """
    this buffersize sould correpond to 6 full tiles, 
    so 7 in total in memory
    """ 
    dataSet.shuffle(buffersize=3000000).batch(64,drop_remainder=True)
    
    #normalization
    data = dataSet # should be split in train and test
    imax = np.max(data) # 15.000-ish
    data = (data+0.1)/ (imax+1)
    """
    network to be addded below
    """
    vae =  keras.models.load_model(path + '_epoch_' + str(epochCounter -1))# change it: make a call to os to create a path
    
    vae.fit(training_data, epochs=1, validation_data = validation_data)
    vae.save(path + '_epoch_' + str(epochCounter))   # change it: make a call to os to create a path
         
             
    #repeat from here        
    epochcounter = epochcounter + 1
  


# In[ ]:


# embedding with tSNE here (for all models?) or, if you want to choose the model, in another script

