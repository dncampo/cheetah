import streamlit as st
from streamlit_option_menu import option_menu

'''
# HAM10000k front-end
'''


# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Upload photo', 'Settings'], 
        icons=['house', 'cloud-upload', 'gear'], menu_icon="cast", default_index=0)
    print(selected)

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


st.markdown('''
This is the first iteration of the front end
''')

'''
## Here we would like to add some controllers in order to ask the user to test either a test image either upload it's own dermatoscopic image


## Once we have these, let's call our API in order to retrieve a prediction

'''


url = '<insert here the url of the API in order to predcit with the model'

'''
## Finally, we can display the prediction to the user
'''
