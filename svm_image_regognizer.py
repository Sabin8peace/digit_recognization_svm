from PIL import Image, ImageOps
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# Load and train model
# ---------------------------


@st.cache_data
def load_and_train_model():
    train_data = pd.read_csv("digit-recognizer/train.csv")
    X = train_data.drop(columns=['label']).values
    y = train_data['label'].values
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)

    # Optional: print accuracy in terminal
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy


svm_clf, acc = load_and_train_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Handwritten Digit Recognition")
st.info(f" Accuracy is {acc}")
st.write("Upload a 28x28 handwritten digit image and the model will predict it!")

# Image uploader
uploaded_file = st.file_uploader(
    "Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file).convert('L')
    img_resized = img.resize((28, 28))
    st.image(img_resized, caption='Uploaded Image')

    # Prepare image for model
    img_array = np.array(img_resized).reshape(1, 784) / 255.0

    # Predict
    prediction = svm_clf.predict(img_array)[0]

    st.success(f"Predicted Digit: {prediction}")
