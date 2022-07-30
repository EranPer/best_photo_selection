import json
from PIL import Image
import os

import streamlit as st
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import img_to_array, load_img


@st.cache(allow_output_mutation=True)
def load_model(path='src/models/inceptionV3_model'):
    """Retrieves the trained model and maps it to the CPU by default,
    can also specify GPU here."""
    model = tf.keras.Sequential([hub.KerasLayer(path)])
    model.build([None, 224, 224, 3])
    return model


@st.cache()
def load_index_to_label_dict(
        path: str = 'src/index_to_class_label.json'
        ) -> dict:
    """Retrieves and formats the
    index to class label
    lookup dictionary needed to
    make sense of the predictions.
    When loaded in, the keys are strings, this also
    processes those keys to integers."""
    with open(path, 'r') as f:
        index_to_class_label_dict = json.load(f)
    index_to_class_label_dict = {
        int(k): v for k, v in index_to_class_label_dict.items()}
    return index_to_class_label_dict


def load_files(
        keys: list,
        path: str = 'src/data'
        ) -> list:
    """Retrieves files from data folder"""
    files = []
    for file in os.listdir(path):
        files.append(Image.open(path + '/' + file))
    return files


def load_image(
        filename: str = '0.jpg',
        path: str = 'src/data'
        ) -> list:
    """return image with path and filename"""
    return Image.open(path + '/' + filename)


@st.cache(ttl=24*3600)
def predict(
        img: Image.Image,
        index_to_label_dict: dict,
        model,
        k: int
        ) -> list:
    """
    This function transforms the image accordingly,
    puts it to the necessary device (cpu by default here),
    feeds the image through the model getting the output tensor,
    converts that output tensor to probabilities using Softmax,
    and then extracts and formats the top k predictions.
    """
    img = img.resize((224, 224))
    image = img_to_array(img)  
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)
    formatted_predictions = list()
    for idx, x in zip(range(0,6), preds[0]):
        formatted_predictions.append([index_to_class_label_dict[idx], np.round(x*100,3)])
    formatted_predictions = sorted(formatted_predictions, key=lambda x:(x[1]), reverse=True)
    return formatted_predictions


if __name__ == '__main__':
    model = load_model()
    index_to_class_label_dict = load_index_to_label_dict()

    st.title('Rate My Photo!')
    instructions = """
        Either upload your own image or select from
        the sidebar to get a preconfigured image.
        The image you select or upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
    st.write(instructions)

    file = st.file_uploader('Upload An Image')
    images_type = {
        'Bright': '0.jpg',
        'Dark': '1.jpg',
        'Good': '2.jpg',
        'Lens Flare': '3.jpg',
        'Loss': '4.jpg',
        'Motion Blur': '5.jpg'
    }
    data_split_names = list(images_type.keys())

    if file:  # if user uploaded file
        img = Image.open(file)
        prediction = predict(img, index_to_class_label_dict, model, k=5)

    else:
        image_type = st.sidebar.selectbox("Examples", data_split_names)
        image_file = images_type[image_type]

        example_images = load_files(keys=images_type)
        img = load_image(image_file)
        prediction = predict(img, index_to_class_label_dict, model, 5)

    st.title("Here is the image you've selected")
    resized_image = img.resize((336, 336))
    st.image(resized_image)
    if prediction[0][0] == "Good":
        text = "good"
    else:
        text = "bad"
    st.title("Your photo is in " + text + " quality")
    st.title("Here are the five most likely reasons why")
    df = pd.DataFrame(data=np.zeros((5, 2)),
                      columns=['Reason', 'Confidence Level'],
                      index=np.linspace(1, 5, 5, dtype=int))

    for idx, p in enumerate(prediction[:5]):
        df.iloc[idx, 0] = p[0]
        df.iloc[idx, 1] = str(p[1]) + '%'
    st.write(df.to_html(escape=False), unsafe_allow_html=True)
    st.write('\n')
    cred1 = f'<a href="{'https://www.linkedin.com/in/eran-perelman/'}" target="_blank">{'Eran Perelman'}</a>'
    credits = """
        This project was developed by cred1, Asi Sheratzki and Ary Korenvais with the guidance of Morris Alper.
        """
    st.write(credits)
