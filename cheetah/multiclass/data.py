import numpy as np
import pandas as pd
from cheetah.params import BUCKET_AUGMENT_DATA_PATH,BUCKET_AUGMENT_PATH, BUCKET_NAME, BUCKET_TEST_DATA_PATH
import os
from google.cloud import storage
from glob import glob


ENV = os.environ.get('ENV', default="gcp")


def get_augmented_and_test():
    """Gets the augmented data and the the test metadata prepared with
    cheetah.augment._generate_test_csv() for multiclass labels.

    Returns:
        multiclass_dict: a dictionary of dataframes, augmented and test
                         dataframes per class.
    Example: for
        multiclass_dict = {
            'augmented_<label>_df': pd.Dataframe of augmented images
            'test_<label>_df': pd.Dataframe of test images
        }
    """
    augmented_classes = [
        'akiec',
        'bcc',
        'bkl',
        # 'df',
        'mel',
        'nv',
        # 'vasc'
    ]
    augmented_test_dict = {}

    for label in augmented_classes:
        prefix_path = f"gs://{BUCKET_NAME}/data"
        path = f"gs://{BUCKET_NAME}/{BUCKET_AUGMENT_PATH}"
        test_path = os.path.join(path,label,f'{label}_test.csv')
        if ENV == 'local':
            prefix_path = ''
            path = f'raw_data/augment/{label}/{label}_augmented.csv'
            test_path = f'raw_data/augment/{label}/{label}_test.csv'
        df = pd.read_csv(os.path.join(path,label,f'{label}_augmented.csv'))
        test_df = pd.read_csv(test_path)
        if prefix_path != '':
            df['path'] = df['path'].map(lambda x: \
                                        os.path.join(prefix_path, x))
            test_df['path'] = test_df['path'].map(lambda x: \
                                        os.path.join(prefix_path, x))
        augmented_test_dict[f'augmented_{label}_df'] = df
        augmented_test_dict[f'test_{label}_df'] = test_df

    return augmented_test_dict


def multiclass_balancer(df, aug_test_dict, class_size=2000):
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
    test_df = pd.DataFrame()
    train_df = pd.DataFrame()
    for key, category_df in aug_test_dict:
        # stack test dataframes for all categories
        if 'test' in key:
            test_df = pd.concat([test_df, category_df], axis=0)
        if 'augmented' in key:
            category = key[10:-3]
            n_original = len(df.query(f'dx == "{category}"'))
            n_augmented = class_size - n_original
            if n_augmented > 0 and n_augmented <= len(category_df):
                train_df = pd.concat([train_df,
                                      category_df.sample(n_augmented)],
                                      axis=0)
            elif n_augmented > len(category_df):
                print('********* Multiclass balancer Error *************')
                print(f'class_size={class_size}, n_augmented={n_augmented}')
                exit(-1)
    print('********** Taking augmented MULTICLASS data **********')
    return train_df, test_df


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
