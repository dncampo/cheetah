from cheetah.augment.data import data_augment_balancer, get_augment_data
from cheetah.params import *
from cheetah.mlflow import MLFlowBase
from cheetah import data
from cheetah.pipeline import prepare_dataset, tf_train_val_split
from cheetah.utils import configure_for_performance
from cheetah.model import initialize_model, compile, fit_with_earlystop
import os
from datetime import datetime
import pandas as pd

class Trainer(MLFlowBase):

    def __init__(self):
        super().__init__(
            EXPERIMENT_NAME,
            MLFLOW_URI)

    def train(self):

        # create a mlflow training
        self.mlflow_create_run()

        # load data and get corresponding image paths
        df = data.get_data()
        df = data.path_to_metadata(df)
        # load augmented images and test separated
        aug_df, test_mel_df = get_augment_data()

        # balance melanoma + augmented_melanoma images vs non-mel
        df, test_nonmel_df = data_augment_balancer(df, aug_df, class_size=2226)

        # prepare train_val and test dataset with extracted images and labels
        train_val_ds = prepare_dataset(df)
        test_df = pd.concat([test_nonmel_df,test_mel_df], axis=0)
        test = prepare_dataset(test_df)

        # shuffle the train_val dataset
        train_val_ds = train_val_ds.shuffle(len(train_val_ds), reshuffle_each_iteration=False)

        # train validation holdout
        train, val = tf_train_val_split(train_val_ds,val_ratio=0.2)

        # optimize pipeline performance
        train = configure_for_performance(train)
        val = configure_for_performance(val)
        test = configure_for_performance(test)

        # initialize model
        model = initialize_model(MODEL_NAME)
        # compile
        model = compile(model)
        # train model and get duration via timer_func decorator
        model, duration = fit_with_earlystop(model, train, val, patience=8)

        # evaluate model and get loss and accuracy
        loss, acc = model.evaluate(test)

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
