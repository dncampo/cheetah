from email.mime import image
import streamlit as st
import pandas as pd
from PIL import Image
from random import randint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

def upload_photo(model):
    '''Shows a button in order to present a upload dialog to submit an image'''

    st.set_option('deprecation.showfileUploaderEncoding', False)
    to_upload = st.file_uploader("Upload a photo to be classified",
                                 type=([".png", ".jpg", "jpeg", "tiff", "gif", "tga", "bmp"]))
    if to_upload is not None:
        image = Image.open(to_upload)
        st.image(image, width=300,
                 caption='Selected image to be classified', use_column_width=False)

        if st.button('Send image to be assessed'):
            # print is visible in the server output, not in the page
            st.write('Sending image to the model to be assessed')

            # API call here
            # result = result from API
            #print(model.summary())

            #datagenValTest = ImageDataGenerator(preprocessing_function=preprocess_input)
            image = image.resize((224,224))
            image_np = np.array(image)
            image_np = np.expand_dims(image_np, 0)
            print(f"shape of np image: {image_np.shape}")
            #result = randint(0,1) #mocking a result
            result = model.predict(image_np)[0]
            print(result)
            if result[0] >= 0.60:
                st.error(f"You should see a dermatologist. ({result[0]:.2f})")
            elif result[0] >= 0.40:
                st.warning(f"The lesion cannot be classified with precision. ({result[0]:.2f})")
            else:
                st.success(f"It is OK, for now. ({result[0]:.2f})")
        else:
            pass
            #
            #st.write('nothing')
