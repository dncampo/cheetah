import os
import json
from google.cloud import storage
from cheetah.params import BUCKET_NAME, BUCKET_TRAINING_FOLDER

def upload_model_to_gcp(model, filename="model_default_name.h5"):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(BUCKET_TRAINING_FOLDER + "/" + filename[:-3] + "/" + filename)
    blob.upload_from_filename(filename)
    print("=> h5 model {} uploaded to bucket {} inside {}".format(filename, BUCKET_NAME, BUCKET_TRAINING_FOLDER))

    histo = model.history.history
    histo_filename = 'history.json'

    with open(histo_filename, 'w') as convert_file:
        convert_file.write(json.dumps(histo))

    blob = bucket.blob(BUCKET_TRAINING_FOLDER + "/" + filename[:-3] + "/" + histo_filename)
    blob.upload_from_filename(histo_filename)

def save_model(model, filename):
    """method that saves the model into a .h5 file and uploads it on
    Google Storage /models folder if necessary
    HINTS : use .save() of keras and google-cloud-storage"""
    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    model.save(filename)
    print(f"saved h5 model locally, with filename: {filename}")
    upload_model_to_gcp(model, filename)
