import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import base64

# Load style sheet
with open('streamlit/style.css') as style:
    st.markdown(f"<style>{style.read()}</style>", unsafe_allow_html=True)

# Create and run function to apply background
def add_bg():
    with open('streamlit/background.jpg', 'rb') as img:
        img_encoded = base64.b64encode(img.read())

    st.markdown(
        f""" <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{img_encoded.decode()});
            background-size: cover;
        }}
        </style> """,
        unsafe_allow_html=True)
    
add_bg()

# Function to load model to cache
@st.cache_resource(show_spinner="Loading AutoID Model...")
def load_model():
    model = tf.keras.models.load_model('streamlit/model/', compile=False)
    return model

with st.columns([0.4,1,0.01])[1]:
    model = load_model()

def predict(img):
    test_image = img.resize((64,64))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = tf.expand_dims(test_image, axis=0)

    age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    genders = ['Female', 'Male']

    predictions = model.predict(test_image)

    gender_int = int(predictions[1] >= 0.5)
    gender_pred = genders[gender_int]
    gender_conf = round(100*float(predictions[1]), 1) if gender_int == 1 else round(100*(1-float(predictions[1])), 1)

    age_pred = age_groups[predictions[0].argmax()]
    age_conf = round(100*np.max(predictions[0]), 1)

    pred_output = f'This is a {gender_pred} in age group {age_pred}'
    conf_output = f'Gender Confidence: {gender_conf}%, Age Confidence: {age_conf}%'

    return pred_output, conf_output


def main():
    st.write('<h1>AutoID</h1>', unsafe_allow_html=True)
    st.write('<h2>Classifying Demographics</h2>', unsafe_allow_html=True)
    st.write('<p style="font-size: 22px">By Kevin Atkinson</p>', unsafe_allow_html=True)
    st.write('#')

    uploaded_image = st.file_uploader(label='Pick an image to identify:', type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        image_data = Image.open(uploaded_image)
        with st.columns([0.01,1,0.02])[1]:
            st.image(image_data, use_column_width=True)

    # THIS CODE IS EXTREMELY JANKY. STREAMLIT MAKES IT VERY DIFFICULT TO CENTER IMAGES.
    # THIS IS THE SIMPLEST WORK-AROUND I COULD FIND. I'M SORRY TO WHOEVER SEES THIS :(

    col4, col5, col6 = st.columns([1,1,0.3])
    col7, col8, col9 = st.columns([1,1,0.5])
    col10 = st.columns(1)

    identify_btn = col5.button('Identify!')
    if identify_btn:
        try:
            if image_data is None:
                st.write('Please upload an image!')
            else:
                with col8:
                    with st.spinner('Identifying...', ):
                        with col10[0]:
                            predictions, confidence = predict(image_data)
                            st.markdown(f"<h2>{predictions}</h2>", unsafe_allow_html=True)
                            st.markdown(f"<h3>{confidence}</h3>", unsafe_allow_html=True)
        except:
            with col10[0]:
                st.write('An error occurred! Please reload an image.')


if __name__ == '__main__':
    main()
