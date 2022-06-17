from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dropout, Dense, Rescaling
from tensorflow.keras.callbacks import EarlyStopping

from cheetah.utils import timer_func

def initialize_model(model_name):
    if model_name == "CNN":
        model = Sequential()
        model.add(Rescaling(1./255))
        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=(180,240,3)))
        model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.16))

        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same'))
        model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.20))

        model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
        model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation = 'sigmoid'))

    return model

def compile(model,learning_rate=0.0003, beta_1=0.9, beta_2=0.999):
    adam_opt = optimizers.Adam(learning_rate=learning_rate,
                                beta_1=beta_1, beta_2=beta_2)

    model.compile(loss='binary_crossentropy',
                optimizer=adam_opt,
                metrics=['accuracy'])
    return model

@timer_func
def fit_with_earlystop(model, train_set, validation_set, patience=20):
    es = EarlyStopping(patience=patience, restore_best_weights=True)

    model.fit(train_set,
            validation_data=validation_set,
            epochs=1,
            callbacks=[es],
            verbose=2)
    return model
