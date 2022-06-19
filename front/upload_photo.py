from email.mime import image
import streamlit as st
import pandas as pd
from PIL import Image
from random import randint

def upload_photo():
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
            result = randint(0,1) #mocking a result
            if result:
                st.error("You should see a dermatologist.")
            else:
                st.success('It is OK, for now.')
        else:
            pass
            #
            #st.write('nothing')
