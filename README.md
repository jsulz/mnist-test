# MNIST Streamlit

This is a simple Streamlit app that uses a Keras model to classify handwritten digits from the MNIST dataset.

The model is trained in `/src/training.py` and saved in `/models` as `mnist_model.keras`. It is then loaded in the Streamlit app in `app.py` where it is used to make predictions on "images" created from drawing on a small HTML canvas. The app also displays the probability distribution of the model's predictions.

## Usage

To run the Streamlit app locally, clone the repository, `cd` into the created directory, and run the following commands:

- `poetry shell`
- `poetry install`
- `streamlit run app.py`

You can also re-train the model by modifying the net in `src/training.py` and re-running the training by running `python training.py` (which will save the new model in `/models`). This will overwrite the existing model and will be used in subsequent runs of the Streamlit app.
