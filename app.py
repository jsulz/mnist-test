"""
A small Streamlit app that loads a Keras model trained on the MNIST dataset and allows the user to draw a digit on a canvas and get a predicted digit from the model.
"""

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import numpy as np
from keras import models
import pandas as pd

# write the title of the page as MNIST Digit Recognizer
st.title("MNIST Digit Recognizer")

# Stroke width slider for people to play with
stroke_width = st.slider("Stroke width: ", 1, 25, 3)

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#FFF",
    fill_color="#000",
    background_color="#000",
    background_image=None,
    update_streamlit=True,
    height=150,
    width=150,
    drawing_mode="freedraw",
    point_display_radius=0,
    key="canvas",
)

if canvas_result is not None and canvas_result.image_data is not None:

    # Get the image data, convert it to grayscale, and resize it to 28x28 (the same size as the MNIST dataset images)
    img_data = canvas_result.image_data
    im = Image.fromarray(img_data.astype("uint8")).convert("L")
    im = im.resize((28, 28))

    # Convert the image to a numpy array and normalize the values
    final = np.array(im, dtype=np.float32) / 255.0

    # load the mnist_model from the artifacts directory
    model = models.load_model(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "models/mnist_model.keras")
        )
    )

    # Create a placeholder prediction array for the bar chart below
    prediction = np.zeros(10)

    # if final is not all zeros, run the prediction
    if not np.all(final == 0):
        # make a prediction using the test data
        prediction = model.predict(final[None, ...])

        print(prediction)

        # print the prediction
        print(f"prediction : {float(np.argmax(prediction))}")

    # create a bar chart to show the predictions
    st.bar_chart(pd.DataFrame(np.ravel(prediction)))
