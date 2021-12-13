import streamlit as st
import pandas as pd
import warnings # Supress Warnings
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from matplotlib import image as img
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="[Deep-Learning]", page_icon="👤", layout='centered', initial_sidebar_state='auto', menu_items=None)
image = Image.open('emotion.png')

def loadLabelEncoder(path):
    le = LabelEncoder()
    le.classes_ = np.load(path)
    return le

model = tf.keras.models.load_model('model3.h5')
le = loadLabelEncoder("class3.npy")

resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.Resizing(48, 48),
  tf.keras.layers.Rescaling(1./127.5, offset=-1),
])

def load_image(image_file):
    img = Image.open(image_file)
    return img

def guessImage(image, le, model):
    result = resize_and_rescale(image)
    pred = model.predict(np.array([result]))
    score = tf.nn.softmax(pred[0])
    res = le.classes_[np.argmax(score)], 100 * np.max(score)
    df = pd.DataFrame(columns=["classe", "score"])
    df['classe'] = le.classes_
    df['score'] = score * 100
    df = df.sort_values(by=['score'],ascending=False)
    df = df.round({'score': 1})
    print(df)
    st.write("The predicted emotion is: ")
    st.write(res)
    st.write("Array of probability")
    st.dataframe(data=df)


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
    #st.write(file_details)
    img_user = load_image(uploaded_file)
    st.image(img_user,width=250)
    img_user.save(uploaded_file.name)
    image = img.imread(uploaded_file.name)
    guessImage(image=image, le=le, model=model)
