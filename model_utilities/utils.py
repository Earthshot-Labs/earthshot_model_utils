####### Imports ######
from osgeo import gdal, ogr, gdal_array
import os
from google.cloud import storage


####### Functions ######
def upload_to_bucket(gcp_bucket, folder_name, file_name, file_local_path):
    """
    Upload a file to a GCP bucket

    Parameters
    ----------
    - gcp_bucket: (string) name of the GCP bucket
    - folder_name: (string) name of the folder in the GCP bucket
    - file_name: (string) name of the file that will be uploaded to GCP
    - file_local_path: (string) local path of the file
    -------
    """
    client = storage.Client()
    storage.blob._DEFAULT_CHUNKSIZE = 5 * 1024 * 1024 # https://github.com/GoogleCloudPlatform/python-docs-samples/issues/2488#issuecomment-1170362655
    bucket = client.bucket(gcp_bucket)
    blob_path = f"{folder_name}/{file_name}"
    print(f"\nUpload to bucket: {blob_path} from {file_local_path}")
    blob = bucket.blob(blob_path)
    if os.path.exists(file_local_path):
        blob.upload_from_filename(file_local_path)
    else:
        print(f'ERROR in upload_to_bucket: file {file_local_path} does not exist.')
    return blob_path



def glob_blob_in_GCP(gcp_bucket, gcp_folder_name, extension='.tif'):
    """
    Finds all the pathnames in GCP bucket folder matching the extension provided.

    Parameters
    ----------
    - gcp_bucket: (string) name of the GCP bucket
    - gcp_folder_name: (string) name of the folder in the GCP bucket
    - extension: (string) extension of the files to find the pathnames. default='.tif'

    -------
    Return
    ----------
    - list_paths: (list of strings) of the matching GCP pathnames
    -------
    """
    client = storage.Client()
    list_paths = []
    for blob in client.list_blobs(gcp_bucket, prefix=f'{gcp_folder_name}'):
        file = str(blob).split('/')[-1].split(',')[0]
        if extension in file:
            # see https://gis.stackexchange.com/questions/428298/open-raster-file-with-gdal-on-google-cloud-storage-bucket-returns-none
            list_paths.append(f'/vsigs/{gcp_bucket}/{gcp_folder_name}/{file}')
    return list_paths