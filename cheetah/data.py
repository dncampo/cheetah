from cheetah.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, BUCKET_IMAGE_FOLDER
import numpy as np
import pandas as pd
import os
from google.cloud import storage
from glob import glob

ENV = os.environ.get('ENV', default="gcp")

def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}" #gcp path by default
    if ENV == 'local':
        path = 'raw_data/HAM10000_metadata.csv'
    return pd.read_csv(path)

def path_to_metadata(df):
    '''Adds a column to the dataframe pointing to the path of the image on the
    file system (either local or a cloud bucket'''
    if ENV == 'local':
        base_skin_dir = 'raw_data/'
        imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                            for x in glob(os.path.join(base_skin_dir,'**', '*.jpg'),recursive=True)}

    else:
        # gs://{BUCKET_NAME}/data/raw_data/HAM10000_images_part_1/
        # https://cloud.google.com/storage/docs/samples/storage-list-files-with-prefix
        prefix ='data/raw_data/'
        delimiter=None
        storage_client = storage.Client()
        # Note: Client.listl_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs(BUCKET_NAME, prefix=prefix, delimiter=delimiter)

        imageid_path_dict = {os.path.splitext(os.path.basename(blob.name))[0]: f"gs://{BUCKET_NAME}/{blob.name}"
                            for blob in blobs}


    df['path'] = df['image_id'].map(imageid_path_dict.get)
    return df

def data_balancer(df):
    '''Returns a metadata set with equal number of melanoma and nevi observations'''
    mel_df = df.query('dx == "mel"')
    rest_df = df.query('dx == "nv"')
    #randomly takes indexes equal number of melanoma observations
    idx_nonmel = np.random.choice(list(rest_df.index), len(mel_df))
    non_mel_balanced = df.iloc[idx_nonmel]
    balanced_meta = pd.concat([non_mel_balanced,mel_df], axis=0)
    return balanced_meta

def reduce_set(df, reduction_factor = 10):
    '''Reduces dataset size by the reduction factor'''
    # Choosing the random indices of small train set and small test set
    idx =  np.random.choice(len(df), round(len(df)/reduction_factor))
    # Collecting the two subsamples images_train_small and images_test_small from images_train and images_test
    reduced_df = df.iloc[idx]
    return reduced_df
