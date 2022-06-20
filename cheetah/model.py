from tensorflow.keras import Sequential, optimizers, Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import Dense, Rescaling, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import Xception, ResNet50, vgg19
from cheetah.params import IMAGE_HEIGHT, IMAGE_WIDTH

from cheetah.utils import timer_func

def initialize_model(model_name):
    if model_name == "CNN":
        model = Sequential()
        model.add(Rescaling(1./255))
        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',
                         input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3)))
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

    elif model_name == 'Xception':
        base_model = Xception(
                        weights='imagenet',  # Load weights pre-trained on ImageNet.
                        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                        include_top=False)  # Do not include the ImageNet classifier at the top.

        base_model.trainable = False  # Freeze the model

        inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        # We make sure that the base_model is running in inference mode here,
        # by passing `training=False`. This is important for fine-tuning, as you will
        # learn in a few paragraphs.
        x = base_model(inputs, training=False)
        # Convert features of shape `base_model.output_shape[1:]` to vectors
        x = GlobalAveragePooling2D()(x)
        # A Dense classifier with a single unit (binary classification)
        outputs = Dense(1, activation = 'sigmoid')(x)
        model = Model(inputs, outputs)

    elif model_name == 'ResNet50':
        base_model = ResNet50(
                        weights='imagenet',  # Load weights pre-trained on ImageNet.
                        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                        include_top=False)  # Do not include the ImageNet classifier at the top.

        base_model.trainable = False  # Freeze the model

        inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        # We make sure that the base_model is running in inference mode here,
        # by passing `training=False`. This is important for fine-tuning, as you will
        # learn in a few paragraphs.
        x = base_model(inputs, training=False)
        # Convert features of shape `base_model.output_shape[1:]` to vectors
        x = GlobalAveragePooling2D()(x)
        # A Dense classifier with a single unit (binary classification)
        outputs = Dense(1, activation = 'sigmoid')(x)
        model = Model(inputs, outputs)

    elif model_name == 'ResNet50_finetuned':
        # Uses a learning rate in adam optimizer reduced by 10%:
        # adam_opt = optimizers.Adam(learning_rate=learning_rate/10,
        #                        beta_1=beta_1, beta_2=beta_2)
        base_model = ResNet50(
                        weights='imagenet',  # Load weights pre-trained on ImageNet.
                        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                        include_top=False)  # Do not include the ImageNet classifier at the top.

        base_model.trainable = True  # Fine tune the model

        inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        # We make sure that the base_model is running in inference mode here,
        # by passing `training=False`. This is important for fine-tuning, as you will
        # learn in a few paragraphs.
        x = base_model(inputs, training=False)
        # Convert features of shape `base_model.output_shape[1:]` to vectors
        x = GlobalAveragePooling2D()(x)
        # A Dense classifier with a single unit (binary classification)
        outputs = Dense(1, activation = 'sigmoid')(x)
        model = Model(inputs, outputs)

    elif model_name == 'VGG19':
        base_model = vgg19.VGG19(
                        weights='imagenet',  # Load weights pre-trained on ImageNet.
                        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                        include_top=False)  # Do not include the ImageNet classifier at the top.

        base_model.trainable = False  # Freeze the model

        inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        # We make sure that the base_model is running in inference mode here,
        # by passing `training=False`. This is important for fine-tuning, as you will
        # learn in a few paragraphs.
        x = base_model(inputs, training=False)
        # Convert features of shape `base_model.output_shape[1:]` to vectors
        x = GlobalAveragePooling2D()(x)
        # A Dense classifier with a single unit (binary classification)
        outputs = Dense(1, activation = 'sigmoid')(x)
        model = Model(inputs, outputs)

    return model

def compile(model,learning_rate=0.0003, beta_1=0.9, beta_2=0.999):
    adam_opt = optimizers.Adam(learning_rate=learning_rate/10,
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
            epochs=300,
            callbacks=[es],
            verbose=2)
    return model
