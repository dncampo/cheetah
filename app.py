from distutils.command.upload import upload
from unittest.main import MAIN_EXAMPLES
import streamlit as st
from streamlit_option_menu import option_menu
from front.home import home
from front.upload_photo import upload_photo

'''
# HAM10000k front-end
'''

MAIN_MENU = "Main Menu"
HOME = "Home"
TEST_PHOTO = "Select from test photos"
UPLOAD_PHOTO = "Upload photo"
SETTINGS = "Settings"

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu(MAIN_MENU, [HOME, TEST_PHOTO, UPLOAD_PHOTO, SETTINGS],
        icons=['house', 'list-task' , 'cloud-upload', 'gear'], menu_icon="cast", default_index=0)
    print(selected)

if selected==HOME:
    home()


elif selected==UPLOAD_PHOTO:
    upload_photo()



# 2. horizontal menu
#selected2 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'],
#    icons=['house', 'cloud-upload', "list-task", 'gear'],
#    menu_icon="cast", default_index=0, orientation="horizontal")
#selected2

# 3. CSS style definitions
#selected3 = option_menu(None, ["Home", "Upload",  "Tasks", 'Settings'],
#    icons=['house', 'cloud-upload', "list-task", 'gear'],
#    menu_icon="cast", default_index=0, orientation="horizontal",
#    styles={
#        "container": {"padding": "0!important", "background-color": "#fafafa"},
#        "icon": {"color": "orange", "font-size": "25px"},
#        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
#        "nav-link-selected": {"background-color": "green"},
#    }
#)


#st.markdown('''
#This is the first iteration of the front end
#''')
#
#'''
### Here we would like to add some controllers in order to ask the user to test either a test image either upload it's own dermatoscopic image
#
#
### Once we have these, let's call our API in order to retrieve a prediction
#
#'''
#
#
#url = '<insert here the url of the API in order to predcit with the model'
#
#'''
### Finally, we can display the prediction to the user
#'''
