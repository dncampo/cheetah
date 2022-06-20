import numpy as np
import pandas as pd
from cheetah.params import BUCKET_AUGMENT_DATA_PATH, BUCKET_NAME, BUCKET_TEST_DATA_PATH
import os
from google.cloud import storage
from glob import glob


ENV = os.environ.get('ENV', default="gcp")


def get_augment_data():
    """Gets the augmented data and the the test metadata prepared with
    cheetah.augment._generate_test_csv().

    Returns two pd.Dataframes for augmented and test images.
    """
    path = f"gs://{BUCKET_NAME}/{BUCKET_AUGMENT_DATA_PATH}" #gcp path by default
    test_path = f"gs://{BUCKET_NAME}/{BUCKET_TEST_DATA_PATH}"
    if ENV == 'local':
        path = 'raw_data/augment/mel_augmented.csv'
        test_path = 'raw_data/augment/mel_test.csv'
    augment_df = pd.read_csv(path)
    test_mel_df = pd.read_csv(test_path)
    return augment_df, test_mel_df

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

def data_augment_balancer(df, aug_df, class_size=2000):
    '''Prepare a dataframe with balanced number of classes of set_size
    number of images with balance ratio of non_mel to mel. The balance
    takes (set_size/2 - n) augmented images, where n is the number of
    original images.

    Args:
        df: pd.Dataframe with original data.
        aug_df: pd.Dataframe with augmented data.
        set_size: Size of the final dataset

    Returns:
        balanced_meta: pd.Dataframe with balanced images via data augmentation.
    '''
    # Retrieve Nevi and Melanoma from original dataset
    rest_df = df.query('dx == "nv"')
    mel_df = df.query('dx == "mel"')

    #randomly takes indexes equal number of melanoma observations
    train_test_rest = rest_df.sample(0.1 * len(rest_df) + class_size)

    # add path to non_melanoma df
    train_test_rest = path_to_metadata(train_test_rest)
    # holdout
    train_rest = train_test_rest[:class_size]
    test_rest = train_test_rest[class_size:]

    # add path to melanoma df
    mel_df = path_to_metadata(mel_df)

    # Retrieve Melanoma augmented images
    idx_mel = np.random.choice(list(aug_df.index), class_size - len(mel_df))
    mel_augmented = df.iloc[idx_mel]

    print('********** Taking augmented data **********')
    balanced_meta = pd.concat([train_rest,mel_df,mel_augmented], axis=0)

    return balanced_meta, test_rest
