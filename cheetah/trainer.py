from cheetah.params import *
from cheetah.mlflow import MLFlowBase
from cheetah import data
from cheetah.pipeline import prepare_dataset, tf_train_val_split
from cheetah.utils import configure_for_performance
from cheetah.model import initialize_model, compile, fit_with_earlystop
import os
from datetime import datetime

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

        # balance data mel vs non-mel
        df = data.data_balancer(df)

        # prepare tensor dataframe with extracted images and labels
        ds = prepare_dataset(df)

        # shuffle the dataset
        ds = ds.shuffle(len(ds), reshuffle_each_iteration=False)

        # train validation holdout
        train_val, test = tf_train_val_split(ds,val_ratio=0.1)
        train, val = tf_train_val_split(train_val,val_ratio=0.2)

        # optimize pipeline performance
        train = configure_for_performance(train)
        val = configure_for_performance(val)
        test = configure_for_performance(test)

        # initialize model
        model = initialize_model(MODEL_NAME)
        # compile
        model = compile(model)
        # train model and get duration via timer_func decorator
        model, duration = fit_with_earlystop(model, train, val, patience=20)

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
        self.mlflow_log_param("n_images", ds.cardinality().numpy())
        self.mlflow_log_param("image_size",(IMAGE_HEIGHT,IMAGE_WIDTH))
        self.mlflow_log_param("n_params", f'{model.count_params():,}')
        self.mlflow_log_param("duration", f'{duration:.2f}')
        self.mlflow_log_param("env", os.environ.get('ENV', default="gcp"))



        return model

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
