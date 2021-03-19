"""
Download tiles from GCS bucket to SURF dCache storage

"""

import dcachefs
import inspect
import gcsfs
import os
import sys

from configparser import ConfigParser
from concurrent.futures import ProcessPoolExecutor, as_completed


dcache_api_url = "https://dcacheview.grid.surfsara.nl:22880/api/v1/"
dcache_webdav_url = "https://webdav.grid.surfsara.nl:2880"


def _get_token(rclone_config_file: str):
    """ Extract token from file (rclone config or plain file) """
    with open(rclone_config_file) as f:
        content = f.read()

    token = None

    for line in content.splitlines():
        # try rclone config file
        if line.startswith("bearer_token"):
            token = line.split()[-1]

    if token is None:
        # assume plain text file
        token = content.strip()
    return token


class Downloader:
    """
    Provide functionality to download a file from GCS to dCache
    """
    def __init__(self, gcs_token_file, dcache_token_file):
        """
        :param gcs_token_file: file with GCS credentials
        :param dcache_token_file: file with dCache token (rclone config file
            or plain text file)
        """

        # read authentication credentials created by `gcloud`
        self.gcs_fs = gcsfs.GCSFileSystem(token=gcs_token_file)

        # configure access to dCache file system
        self.dcache_fs = dcachefs.dCacheFileSystem(
            token=_get_token(dcache_token_file),
            api_url=dcache_api_url,
            webdav_url=dcache_webdav_url,
            block_size=0  # will open file in stream mode
        )

    def download(self, from_uri, to_uri, from_kwargs=None, to_kwargs=None):
        """
        Download file

        :param from_uri: source GCS path
        :param to_uri: destination dCache path
        :param from_kwargs: additional kwargs for GCSFileSystem `open` method
        :param to_kwargs: additional kwargs for dCacheFileSystem `open` method
        :return:
        """

        from_kwargs = from_kwargs or {}
        to_kwargs = to_kwargs or {}

        already_exists = self.dcache_fs.exists(to_uri)
        same_size = False if not already_exists else \
            self.gcs_fs.size(from_uri) == self.dcache_fs.size(to_uri)

        # download missing/incomplete tiles
        if not already_exists or not same_size:
            with self.gcs_fs.open(from_uri, **from_kwargs) as f_read:
                with self.dcache_fs.open(to_uri, "wb", **to_kwargs) as f:
                    f.write(f_read)


def main(gcs_token_file: str, dcache_token_file: str, gcs_path: str,
         dcache_path: str, n_workers: int):
    """
    Download a directory from GCS to dCache, running the download of each file
    in parallel using multiprocessing.

    :param gcs_token_file: GCS token file
    :param dcache_token_file: dCache token file
    :param gcs_path: GCS bucket path
    :param dcache_path: dCache path
    :param n_workers: number of processes
    """
    downloader = Downloader(gcs_token_file=gcs_token_file,
                            dcache_token_file=dcache_token_file)

    files = downloader.gcs_fs.glob(gcs_path)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        for file in files:
            from_uri = f"gs://{file}"
            _, filename = os.path.split(file)
            to_uri = f"{dcache_path}/{filename}"
            future = executor.submit(downloader.download, from_uri, to_uri,
                                     to_kwargs={"timeout": 1200})
            futures.append(future)

    for future in as_completed(futures):
        future.result()


def _get_input_args(config_file: str):
    """
    Read input arguments of `main` from config file

    :param config_file: (str) config file path
    """
    print(f"Reading {config_file}..")
    config = ConfigParser()
    config.read(config_file)
    section = config.sections().pop()
    sign = inspect.signature(main)
    return {key: val.annotation(config[section].get(key, None))
            for key, val in sign.parameters.items()}


if __name__ == "__main__":
    args = _get_input_args(sys.argv[-1])
    main(**args)
