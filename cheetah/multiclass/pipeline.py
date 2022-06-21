import tensorflow as tf
from cheetah.params import IMAGE_HEIGHT, IMAGE_WIDTH
from tensorflow.io import decode_jpeg, read_file
from tensorflow.image import resize
from tensorflow.data import AUTOTUNE, Dataset

def tf_train_val_split(ds, val_ratio=0.2):
    '''Takes a ds tensorflow.dataset and returns train and validation
    datasets with a validation ratio val_ratio.'''
    val_size = int(len(ds) * val_ratio)
    train_ds = ds.skip(val_size)
    val_ds = ds.take(val_size)
    return train_ds, val_ds

def decode_resize(img,img_height=IMAGE_HEIGHT, img_width=IMAGE_WIDTH):
    '''Decodes JPG img=image and resizes to img_height, img_width.
    Returns a uint8 resize image'''
    # Convert the compressed string to a 3D uint8 tensor
    img = decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    img = resize(img, [img_height, img_width])
    return tf.cast(img, tf.uint8)

def get_image(pathdx):
    '''Returns decoded image and its corresponding binary encoded
    classification.'''
    img = read_file(pathdx[0])
    return decode_resize(img)

def get_label(pathdx):
    '''Returns decoded image and its corresponding binary encoded
    classification.'''
    target = one_hot_encode(pathdx[1])
    return target # pathdx[1]

def prepare_dataset(df):
    '''Prepares a tensorflow dataset with image path and classification and gets
    the decoded and resized image with its classification. Optimizes the mapping
    with parallel calls tf.data.AUTOTUNE
    Returns a tf.parallelMapDataset'''
    print(df.columns)
    print(df.shape)
    list_ds = Dataset.from_tensor_slices(df[['path','dx']])

    tf_ds = list_ds.map(get_image, num_parallel_calls=AUTOTUNE)
    tf_label = list_ds.map(get_label, num_parallel_calls=AUTOTUNE)
    return tf_ds, tf_label


def one_hot_encode(df):
    encode_category = {'akiec': 0,
                    'bcc': 1,
                    'bkl': 2,
                    'df': 3,
                    'mel': 4,
                    'nv': 5,
                    'vasc': 6}
    df['ohe'] = df['dx'].map(encode_category)
    indices = df.ohe.tolist()
    depth = len(encode_category)

    return tf.one_hot(indices, depth)
