import gc
import random

import geopandas as gpd
import rioxarray as rioxr
import tensorflow as tf


class Dataset:
    """
    Split tiles into square cutouts and return these as a tensorflow Dataset.
    """
    def __init__(self, tile_list, cutout_size, bands, offset=0, stride=None,
                 num_tiles=None, shuffle_tiles=False, norm_threshold=None,
                 return_coordinates=False):
        """
        :param tile_list: list of tile paths
        :param cutout_size: length of the cutout side (number of pixels)
        :param bands: list of band indices to be considered
        :param offset: lateral (X and Y) cutout offset (number of pixels)
        :param stride: cutout stride (number of pixels)
        :param num_tiles: only use a selection of the provided tile list
        :param shuffle_tiles: if True, shuffle tiles
        :param norm_threshold: normalize image intensities using the provided
            threshold value
        :param return_coordinates: if True, return the (X, Y) coordinates
            together with the cutouts
        """
        self.cutout_size = cutout_size
        self.bands = bands
        self.stride = stride if stride is not None else self.cutout_size
        _num_tiles = num_tiles if num_tiles is not None else len(tile_list)
        self.tiles = random.sample(tile_list, _num_tiles) \
            if shuffle_tiles else tile_list[:_num_tiles]
        if offset >= cutout_size:
            raise ValueError(
                "offset larger than window size - set "
                "offset to {}".format(cutout_size % offset)
            )
        self.offset = offset

        self.mask = None
        self.buffer = None
        self.invert = None
        self.all_touched = None
        self.norm_threshold = norm_threshold
        self.return_coordinates = return_coordinates

    def set_mask(self, geometry, crs, buffer=None, invert=False,
                 all_touched=False):
        """
        Clip the tiles with the provided geometry before generating the
        cutouts.

        :param geometry: geometry
        :param crs: coordinate reference system (geopandas compatible)
        :param buffer: apply provided buffer to the geometry
        :param invert: clip the tiles with the inverted mask
        :param all_touched: see `rioxarray.clip`
        """
        self.mask = gpd.GeoSeries({"geometry": geometry}, crs=crs)
        self.buffer = buffer if buffer is not None else 0
        self.invert = invert
        self.all_touched = all_touched

    def to_tf(self):
        """ Return dataset as a tensorflow `Dataset` object. """
        ds = tf.data.Dataset.from_generator(
            self._generate_cutouts,
            output_types=(tf.float64, tf.float64, tf.float32),
            output_shapes=(
                None,  # x
                None,  # y
                # samples, x_win, y_win, bands
                (None, self.cutout_size, self.cutout_size, None)
            )
        )
        if not self.return_coordinates:
            ds = ds.map(lambda x, y, cutout: cutout)  # only return cutout
        # remove the outer dimension of the array if not return_coordinates
        return ds.flat_map(
            lambda *x: tf.data.Dataset.from_tensor_slices(
                x if len(x) > 1 else x[0]
            )
        )

    def _generate_cutouts(self):
        """
        Iterate over (a selection of) the tiles yielding all cutouts for each
        of them.
        """
        for tile in self.tiles:
            gc.collect()  # solves memory leak when dataset is used within fit

            # read tile - floats are required to mask with NaN's
            da = rioxr.open_rasterio(tile).astype("float32")

            # select bands
            if self.bands is not None:
                if type(self.bands) is not list:
                    da = da.sel(band=[self.bands])
                else:
                    da = da.sel(band=self.bands)

            if self.mask is not None:
                mask = self.mask.to_crs(da.spatial_ref.crs_wkt)
                geometry = mask.unary_union.buffer(self.buffer)
                da = da.rio.clip([geometry], drop=True, invert=self.invert,
                                 all_touched=self.all_touched)

            # apply offset
            da = da.shift(x=self.offset, y=self.offset)
            da['x'] = da.x.shift(x=self.offset)
            da['y'] = da.y.shift(y=self.offset)

            # generate windows
            da = da.rolling(x=self.cutout_size, y=self.cutout_size)
            da = da.construct({'x': 'x_win', 'y': 'y_win'}, stride=self.stride)

            # drop NaN-containing windows
            da = da.stack(sample=('x', 'y'))
            da = da.dropna(dim='sample', how='any')

            # normalize
            if self.norm_threshold is not None:
                da = (da + 0.1) / (self.norm_threshold + 1)
                da = da.clip(max=1)

            yield (
                da.sample.coords['x'],
                da.sample.coords['y'],
                da.data.transpose(3, 1, 2, 0)  # samples, x_win, y_win, bands
            )
