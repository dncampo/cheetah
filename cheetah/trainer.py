from cheetah import params
from cheetah.mlflow import MLFlowBase
from cheetah import data
from cheetah.pipeline import prepare_dataset, tf_train_val_split
from cheetah.utils import configure_for_performance
from cheetah.model import initialize_model, compile, fit_with_earlystop

class Trainer(MLFlowBase):

    def __init__(self):
        super().__init__(
            params.EXPERIMENT_NAME,
            params.MLFLOW_URI)

    def train(self):

        # create a mlflow training
        self.mlflow_create_run()

        # log params
        self.mlflow_log_param("model_name", params.MODEL_NAME)

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
        model = initialize_model(params.MODEL_NAME)
        # compile
        model = compile(model)
        # train
        model = fit_with_earlystop(model, train, val, patience=20)

        # evaluate
        loss, acc = model.evaluate(test)

        # save the trained model
        model.save('model.h5')

        # register score in MLFlow
        self.mlflow_log_metric("loss", loss)
        self.mlflow_log_metric("accuracy", acc)

        return model

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
