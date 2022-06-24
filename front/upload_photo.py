from email.mime import image
from unittest import result
import streamlit as st
import pandas as pd
from PIL import Image
from random import randint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

def get_n_highest_clases(result, n=3):
    #sort by values and return the highest n:
    sorted_dict = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_dict.keys())[:n]

def upload_photo(model_bin, model_cat):
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
            #image = image.resize((224,224))
            image = image.resize((120,90))
            image_np = np.array(image)
            image_np = np.expand_dims(image_np, 0)
            print(f"shape of np image: {image_np.shape}")
            #result = randint(0,1) #mocking a result
            result_bin = model_bin.predict(image_np)[0]
            print(f"result_bin: {result_bin}")
            if result_bin[0] >= 0.60:
                st.info(f"Prioritary appointment. ({result_bin[0]:.2f})")
                if st.button('Schedule appointment'):
                    # print is visible in the server output, not in the page
                    st.write('Appointment scheduled for ...')
                    if st.write('Ok'):
                        return
                else:
                    st.write('Please, notify results to the dermatologist and schedule an appoitment')

            else:
                #binary says it's not melanome
                result_cat = model_cat.predict(image_np)[0]
                result_cat = {
                'Actinic Keratoses / Intrapithelial Carcinoma (akiec)': result_cat[0],
                'Basal Cell Carcinoma (bcc)': result_cat[1],
                'Benign Keratosis (bkl)': result_cat[2],
                'Dermatofibroma (df)': result_cat[3],
                'Melanoma (mel)': result_cat[4],
                'Melanocytic Nevi (nv)': result_cat[5],
                'Vascula skin lesion (vasc)': result_cat[6]
                }

                print(f"result_cat: {result_cat}")
                #st.success(f"Résultats categorical. ({result_cat})")
                n_highest = get_n_highest_clases(result=result_cat, n=3)
                n_highest_values = [result_cat.get(key) for key in n_highest]
                n_highest_zip = zip(n_highest, n_highest_values)
                print(f"highest values are: {n_highest}")
                print(f"highest values are: {n_highest_values}")
                for cat, val in n_highest_zip:
                    st.info(
                        f'''
                        ##### {cat} -> {val:.6f}
                        '''
                    )
        else:
            pass
