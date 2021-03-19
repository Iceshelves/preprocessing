import pathlib

import geopandas as gpd
import pandas as pd
import pystac


def _read_tile_catalog(catalog_path):
    """ Read the tile catalog """
    catalog_path = pathlib.Path(catalog_path)
    catalog_path = catalog_path / "catalog.json"
    return pystac.Catalog.from_file(catalog_path.as_posix())


def _catalog_to_geodataframe(catalog, crs="WGS84"):
    """ Convert STAC catalog to a GeoDataFrame object """
    features = {item.id: item.to_dict() for item in catalog.get_all_items()}
    gdf = gpd.GeoDataFrame.from_features(features.values())
    gdf.index = features.keys()
    for column in gdf.columns:
        if 'datetime' in column:
            gdf[column] = pd.to_datetime(gdf[column])
    gdf = gdf.set_crs(crs)
    return gdf


def _read_labels(labels_path, verbose=True):
    """ Read all labels, and merge them in a single GeoDataFrame """
    labels_path = pathlib.Path(labels_path)
    labels = [gpd.read_file(p) for p in labels_path.glob("*.geojson")]
    if verbose:
        print("Labels successfully read from {} files".format(len(labels)))

    crs = labels[0].crs
    assert all([label.crs == crs for label in labels])
    labels = pd.concat(labels)

    # fix datetimes' type
    labels.Date = pd.to_datetime(labels.Date)
    return labels


def _get_tile_paths(catalog, item_ids, asset_key):
    """ Extract the asset paths from the catalog """
    items = (catalog.get_item(item_id, recursive=True) for item_id in item_ids)
    assets = (item.assets[asset_key] for item in items)
    return [asset.get_absolute_href() for asset in assets]


def _filter_labels(labels, start_datetime, end_datetime, verbose=True):
    """ Select the labels whose date in the provided datetime range """
    mask = (labels.Date >= start_datetime) & (labels.Date <= end_datetime)
    if verbose:
        print("Selecting {} out of {} labels".format(mask.sum(), len(labels)))
    return labels[mask]


def split_train_and_test(catalog_path, test_set_size, labels_path=None,
                         validation_set_size=None, random_state=None,
                         verbose=True):
    """
    The tiles in the provided STAC catalog are split in test, validation and
    training sets.

    :param catalog_path: STAC catalog path
    :param test_set_size: size of the test set
    :param labels_path: path to the labels. If provided, all tiles overlapping
        with the labels will be included in the test set
    :param validation_set_size: size of the validation set
    :param random_state: random state for the data set sampling
    :param verbose: if True, print info to stdout
    """

    # read tile catalog
    catalog = _read_tile_catalog(catalog_path)
    tiles = _catalog_to_geodataframe(catalog)

    # read labels and reserve the labeled tiles for the test set
    test_set = gpd.GeoDataFrame()
    if labels_path is not None:
        labels = _read_labels(labels_path, verbose)
        labels = labels.to_crs(tiles.crs)  # make sure same CRS is used

        # select the only labels matching the tiles timespan
        labels = _filter_labels(labels,
                                tiles.start_datetime.min(),
                                tiles.end_datetime.max())

        # add the tiles overlapping with the labels to the test set
        mask = tiles.intersects(labels.unary_union)
        test_set = test_set.append(tiles[mask])

        if verbose:
            print("{} tiles overlap with labels: ".format(len(test_set)))
            for tile in test_set.index:
                print(tile)

        if len(test_set) > test_set_size:
            raise ValueError(
                "Labels overlap with {} tiles while a test set size of {} was "
                "selected - please increase `test_set_size` to >= {}.".format(
                    len(test_set),
                    test_set_size,
                    len(test_set)
                )
            )

        tiles = tiles[~mask]

    # pick additional unlabeled tiles for the test set
    test_set_unlabeled = tiles.sample(test_set_size - len(test_set),
                                      random_state=random_state)
    test_set = test_set.append(test_set_unlabeled)
    test_set_paths = _get_tile_paths(catalog, test_set.index, "B2-B3-B4-B11")

    train_set = tiles.index.difference(test_set_unlabeled.index)
    train_set = tiles.loc[train_set]

    # split validation set and training set
    val_set = gpd.GeoDataFrame()
    if validation_set_size is not None:
        val_set = val_set.append(
            train_set.sample(validation_set_size, random_state=random_state)
        )
        train_set = train_set.index.difference(val_set.index)
        train_set = tiles.loc[train_set]

    train_set_paths = _get_tile_paths(catalog, train_set.index, "B2-B3-B4-B11")
    val_set_paths = _get_tile_paths(catalog, val_set.index, "B2-B3-B4-B11")

    return train_set_paths, val_set_paths, test_set_paths
