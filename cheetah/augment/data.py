import numpy as np
import pandas as pd
from cheetah.params import BUCKET_AUGMENT_DATA_PATH, BUCKET_NAME
import os
from google.cloud import storage
from glob import glob


ENV = os.environ.get('ENV', default="gcp")


def get_augment_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    path = f"gs://{BUCKET_NAME}/{BUCKET_AUGMENT_DATA_PATH}" #gcp path by default
    if ENV == 'local':
        path = 'raw_data/augment/mel_augmented.csv'
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

def data_augment_balancer(df, aug_df, n_images=1113):
    '''Returns a metadata set with equal number of melanoma and nevi observations'''
    # Retrieve Nevi
    rest_df = df.query('dx == "nv"')
    #randomly takes indexes equal number of melanoma observations
    idx_nonmel = np.random.choice(list(rest_df.index), n_images)
    non_mel_balanced = df.iloc[idx_nonmel]
    # add path to non_mel df
    non_mel_balanced = path_to_metadata(non_mel_balanced)

    # Retrieve Melanoma augmented images
    idx_mel = np.random.choice(list(aug_df.index), n_images)
    mel_augmented = df.iloc[idx_mel]

    print('********** Taking augmented data **********')
    balanced_meta = pd.concat([non_mel_balanced,mel_augmented], axis=0)

    return balanced_meta
