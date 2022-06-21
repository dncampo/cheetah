# ----------------------------------
#            Model Params
# ----------------------------------
BATCH_SIZE=32
IMAGE_HEIGHT=90
IMAGE_WIDTH=120
# ----------------------------------
#               MLFlow
# ----------------------------------

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[FR] [Nice] [abdielrt,dncampo] Computer Vision v1.0"

# ----------------------------------
#      Google Cloud Plateform
# ----------------------------------
#PROJECT_ID='art-ham10k-wagon' if USER == 'art' else 'ham10k-wagon'
BUCKET_NAME = 'ham10k-storage'
REGION='europe-west1'
##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/raw_data/HAM10000_metadata.csv'
BUCKET_AUGMENT_DATA_PATH = 'data/raw_data/augment/mel/mel_augmented.csv'
BUCKET_TEST_DATA_PATH = 'data/raw_data/augment/mel/mel_test.csv'
BUCKET_IMAGE_FOLDER = 'data/raw_data/'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'ResNet50_finetuned'

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME = 'cheetah'
FILENAME='trainer'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'


LOCAL_PATH="raw_data/"
BUCKET_FOLDER='data'
BUCKET_TRAINING_FOLDER='models'

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -
#https://cloud.google.com/ai-platform/training/docs/runtime-version-list
PYTHON_VERSION=3.7
RUNTIME_VERSION=2.7
