import streamlit as st
import pandas as pd
import numpy as np
import warnings # Supress Warnings
import numpy as np
from PIL import Image

st.set_page_config(page_title="[Deep-Learning]", page_icon="👤", layout='centered', initial_sidebar_state='auto', menu_items=None)
image = Image.open('emotion.png')
def load_image(image_file):
    img = Image.open(image_file)
    return img

st.title('[Deep-Learning] Emotion Detection')

st.write("Welcome to the Emotion Detection web-app .")
st.write("We have collected thousands of images that are classified according to the emotion expressed by the facial expressions (happiness, neutrality, sadness, anger, surprise, disgust, fear).")
st.write("We was then able to make some deep learning to know which emotion is expressed from an image.")

st.image(image, caption='Emotion Detection')

st.subheader("Image")
uploaded_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    # To See details
    file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type,
                    "filesize":uploaded_file.size}
    st.write(file_details)

    # To View Uploaded Image
    st.image(load_image(uploaded_file),width=250)