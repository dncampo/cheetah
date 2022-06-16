from cheetah.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH
import numpy as np
import pandas as pd
import os
from glob import glob


def get_data(source='local'):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    if source == 'gcp':
        df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
    elif source == 'local':
        df = pd.read_csv('raw_data/HAM10000_metadata.csv')
    return df

def path_to_metadata(df):
    base_skin_dir = 'raw_data/'
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                        for x in glob(os.path.join(base_skin_dir,'**', '*.jpg'),recursive=True)}

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
