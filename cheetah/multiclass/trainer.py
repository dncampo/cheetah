from cheetah.multiclass.data import multiclass_balancer, get_augmented_and_test
from cheetah import data
from cheetah.params import *
from cheetah.mlflow import MLFlowBase
from cheetah.pipeline import prepare_dataset, tf_train_val_split
from cheetah.utils import configure_for_performance
from cheetah.model import compile_multiclass, initialize_model, compile, fit_with_earlystop
from datetime import datetime
import os
import pandas as pd
import tensorflow as tf

class Trainer(MLFlowBase):

    def __init__(self):
        super().__init__(
            EXPERIMENT_NAME,
            MLFLOW_URI)

    def train(self):

        # create a mlflow training
        self.mlflow_create_run()

        # load original data and get corresponding image paths
        df = data.get_data()
        df = data.path_to_metadata(df)
        # load multiclass augmented images and prepare test list
        aug_test_dict = get_augmented_and_test()

        # balance melanoma + augmented_melanoma images vs non-mel
        train_df, test_df = multiclass_balancer(df, aug_test_dict,
                                                class_size=100)

        # prepare train_val and test dataset with extracted images and labels
        X_train_val_ds, y_train_val_ds = prepare_dataset(train_df)
        X_test, y_test = prepare_dataset(test_df)

        # shuffle the train_val dataset
        train_val_ds = tf.stack([X_train_val_ds, y_train_val_ds], axis=1)
        train_val_ds = train_val_ds.shuffle(len(train_val_ds),
                                            reshuffle_each_iteration=False)

        # train validation holdout
        train, val = tf_train_val_split(train_val_ds,val_ratio=0.2)

        # optimize pipeline performance
        train = configure_for_performance(train)
        val = configure_for_performance(val)
        test = configure_for_performance(test)

        for img, label in train.take(1):
            print("Train Image shape: ", img.numpy().shape)
            print("Train Label: ", label.numpy().shape)

        for img, label in val.take(1):
            print("VAl Image shape: ", img.numpy().shape)
            print("VAl Label: ", label.numpy().shape)
        # initialize model
        model = initialize_model(MODEL_NAME)
        # compile
        model = compile_multiclass(model)
        # train model and get duration via timer_func decorator
        model, duration = fit_with_earlystop(model, train, val, patience=8)

        # evaluate model and get loss and accuracy
        loss, acc = model.evaluate(X_test, y_test)

        # save the trained model
        model_path = ''
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S") # current date and time
        if os.environ.get('ENV') == 'gcp':
            model_path = f'gs://{BUCKET_NAME}/{BUCKET_TRAINING_FOLDER}/'

        model.save(model_path + MODEL_NAME + "_" + ts_str + ".h5")

        # register metrics and custom parameters in MLFlow
        self.mlflow_log_metric("loss", loss)
        self.mlflow_log_metric("accuracy", acc)

        self.mlflow_log_param("model_name", MODEL_NAME)
        self.mlflow_log_param("batch_size",BATCH_SIZE)
        self.mlflow_log_param("n_images", train_val_ds.cardinality().numpy() \
                                            + test.cardinality().numpy())
        self.mlflow_log_param("image_size",(IMAGE_HEIGHT,IMAGE_WIDTH))
        self.mlflow_log_param("n_params", f'{model.count_params():,}')
        self.mlflow_log_param("duration", f'{duration:.2f}')
        self.mlflow_log_param("env", os.environ.get('ENV', default="gcp"))



        return model

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
