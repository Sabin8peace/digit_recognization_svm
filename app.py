import streamlit as st
from PIL import Image
import numpy as np
import joblib

# ---------------------------
# Load saved model
# ---------------------------


@st.cache_data
def load_model():
    model = joblib.load("svm_digit_model.pkl")
    acc = joblib.load("accuracy.pkl")

    return model, acc


svm_clf, acc = load_model()

st.info(f"Accuracy is :  {acc*100:.2f}%")
# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🖐 Handwritten Digit Recognition")
st.write("Upload a 28x28 handwritten digit image and the model will predict it!")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    img = Image.open(uploaded_file).convert('L')
    img_resized = img.resize((28, 28))
    st.image(img_resized, caption='Uploaded Image', use_container_width=False)

    # Flatten and normalize
    img_array = np.array(img_resized).reshape(1, 784) / 255.0

    # Predict
    prediction = svm_clf.predict(img_array)[0]

    st.success(f"Predicted Digit: {prediction}")
