"""
A small Streamlit app that loads a Keras model trained on the MNIST dataset and allows the user to draw a digit on a canvas and get a predicted digit from the model.
"""

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import numpy as np
from keras import models
import keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import pandas as pd


@st.cache_resource
def load_picture():
    """
    Loads the first 9 images from the mnist dataset and add them to a plot
    to be displayed in streamlit.
    """
    # load the mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # plot the first 9 images
    for i in range(9):
        plt.subplot(330 + 1 + i)
        image = x_train[i] / 255.0
        plt.imshow(image, cmap=plt.get_cmap("gray"))

    # Save the plot as a png file and show it in streamlit
    st.image("img/show.png", width=250, caption="First 9 images from the MNIST dataset")


def main():
    """
    The main function/primary entry point of the app
    """
    # write the title of the page as MNIST Digit Recognizer
    st.title("MNIST Digit Recognizer")

    col1, col2 = st.columns([0.7, 0.3], gap="small")
    with col1:
        st.markdown(
            "This Streamlit app loads a small Keras neural network trained on the MNIST dataset to predict handwritten digits. Draw a digit in the canvas below, see the model's prediction, along with the probability distribution of the predictions. Additionally, you can change the stroke width of the digit you draw using the slider adjacent to the canvas. Like any machine learning model, this model is a function of the data it was fed during training. As you can see in the picture to the right, the numbers in the images have a specific format (location and size). By playing around with the stroke width and where you draw the digit, you can see how the model's prediction changes."
        )
    with col2:
        # Load the first 9 images from the MNIST dataset and show them
        load_picture()

    col3, col4 = st.columns(2, gap="small")

    with col4:
        # Stroke width slider to change the width of the canvas stroke
        # Starts at 10 because that's reasonably close to the width of the MNIST digits
        stroke_width = st.slider("Stroke width: ", 1, 25, 10)

    with col3:
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

        # if final is not all zeros, run the prediction
        if not np.all(final == 0):
            # make a prediction using the test data
            prediction = model.predict(final[None, ...])

            # print the prediction
            st.header(f"Prediction: {np.argmax(prediction)}")

            # Create a 2 column dataframe with one column as the digits and the other as the probability
            data = pd.DataFrame(
                {"Digit": list(range(10)), "Probability": np.ravel(prediction)}
            )

            # create a bar chart to show the predictions
            st.bar_chart(data, x="Digit", y="Probability")


if __name__ == "__main__":
    main()
