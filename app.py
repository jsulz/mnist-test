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
import time
import onnx
import onnxruntime
from scipy.special import softmax


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
    # This is commented out for not because the plot was created and saved in the img directory during the initial run of the app locally
    # plt.savefig("img/show.png")
    st.image("img/show.png", width=250, caption="First 9 images from the MNIST dataset")


def keras_prediction(final, model_path):
    load_time = time.time()
    model = models.load_model(
        os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
    )
    after_load_curr = time.time()
    curr_time = time.time()
    prediction = model.predict(final[None, ...])
    after_time = time.time()
    return prediction, after_time - curr_time, after_load_curr - load_time


def onnx_prediction(final, model_path):
    im_np = np.expand_dims(final, axis=0)  # Add batch dimension
    im_np = np.expand_dims(im_np, axis=0)  # Add channel dimension
    im_np = im_np.astype("float32")
    load_curr = time.time()
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    after_load_curr = time.time()

    curr_time = time.time()
    result = session.run([output_name], {input_name: im_np})
    prediction = softmax(np.array(result).squeeze(), axis=0)
    after_time = time.time()
    return prediction, after_time - curr_time, after_load_curr - load_curr


def main():
    """
    The main function/primary entry point of the app
    """
    # write the title of the page as MNIST Digit Recognizer
    st.title("MNIST Digit Recognizer")

    col1, col2 = st.columns([0.8, 0.2], gap="small")
    with col1:
        st.markdown(
            """
            This Streamlit app loads a Keras neural network trained on the MNIST dataset to predict handwritten digits. Draw a digit in the canvas below and see the model's prediction. You can: 
            - Change the stroke width of the digit using the slider
            - Choose what model you use for predictions
                - Onnx: The mnist-12 Onnx model from <a href="https://xethub.com/XetHub/onnx-models/src/branch/main/vision/classification/mnist">Onnx's pre-trained MNIST models</a>
                - Autokeras: A model generated using the <a href="https://autokeras.com/image_classifier/">Autokeras image classifier class</a>
                - Basic: A simple two layer nueral net where each layer has 300 nodes
            
            Like any machine learning model, this model is a function of the data it was fed during training. As you can see in the picture, the numbers in the images have a specific shape, location, and size. By playing around with the stroke width and where you draw the digit, you can see how the model's prediction changes.""",
            unsafe_allow_html=True,
        )
    with col2:
        # Load the first 9 images from the MNIST dataset and show them
        load_picture()

    col3, col4 = st.columns(2, gap="small")

    with col4:
        # Stroke width slider to change the width of the canvas stroke
        # Starts at 10 because that's reasonably close to the width of the MNIST digits
        stroke_width = st.slider("Stroke width: ", 1, 25, 10)
        model_choice = st.selectbox(
            "Choose what model to use for predictions:", ("Onnx", "Autokeras", "Basic")
        )
        if "Basic" in model_choice:
            model_path = "models/mnist_model.keras"

        if "Auto" in model_choice:
            model_path = "models/autokeras_model.keras"

        if "Onnx" in model_choice:
            model_path = "models/mnist_12.onnx"

    with col3:
        # Create a canvas component
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color="#FFF",
            fill_color="#000",
            background_color="#000",
            background_image=None,
            update_streamlit=True,
            height=200,
            width=200,
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

        # if final is not all zeros, run the prediction
        if not np.all(final == 0):

            if model_choice != "Onnx":
                prediction, pred_time, load_time = keras_prediction(final, model_path)
            else:
                prediction, pred_time, load_time = onnx_prediction(final, model_path)

            # print the prediction
            st.header(f"Using model: {model_choice}")
            st.write(f"Prediction: {np.argmax(prediction)}")
            st.write(f"Load time (in ms): {(load_time) * 1000:.2f}")
            st.write(f"Prediction time (in ms): {(pred_time) * 1000:.2f}")

            # Create a 2 column dataframe with one column as the digits and the other as the probability
            data = pd.DataFrame(
                {"Digit": list(range(10)), "Probability": np.ravel(prediction)}
            )

            col1, col2 = st.columns([0.8, 0.2], gap="small")
            # create a bar chart to show the predictions
            with col1:
                st.bar_chart(data, x="Digit", y="Probability", height=500)

            # show the probability distribution numerically
            with col2:
                data["Probability"] = data["Probability"].apply(lambda x: f"{x:.2%}")
                st.dataframe(data, hide_index=True)


if __name__ == "__main__":
    main()
