import streamlit as st

def home():
    '''Just renders a home page'''
    print("You clicked home")
    st.markdown('''
    This is the home page
    ''')

    st.markdown('''
    ## presentation, project description, about
    ''')


    url = '<insert here the url of the API in order to predcit with the model'

    '''
    ## Finally, we can display the prediction to the user
    '''
