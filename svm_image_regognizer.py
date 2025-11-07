from PIL import Image, ImageOps
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2  # <-- add this at top if not already

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
st.info(f"Accuracy is :  {acc*100:.2f}%")
st.write("Upload a 28x28 handwritten digit image and the model will predict it!")

# Image uploader
uploaded_file = st.file_uploader(
    "Choose an image...", type=["png", "jpg", "jpeg"])


if uploaded_file is not None:
    # Load and convert to grayscale
    img = Image.open(uploaded_file).convert('L')
    img = np.array(img)

    # ✅ Adaptive Threshold (works even with lighting changes)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    # Now digit is white, background is black.

    # ✅ Find bounding box of digit
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        st.error("No digit detected — try darker writing or thicker lines.")
        st.stop()

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    img = img[y_min:y_max+1, x_min:x_max+1]

    # ✅ Resize to 20x20
    img = Image.fromarray(img).resize((20, 20), Image.LANCZOS)

    # ✅ Pad to 28x28 (center the digit)
    new_img = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - 20) // 2
    x_offset = (28 - 20) // 2
    new_img[y_offset:y_offset+20, x_offset:x_offset+20] = img

    # Preview processed digit
    st.image(new_img, caption="Processed MNIST-Style Image", width=200)

    # ✅ Flatten + Normalize
    img_array = new_img.reshape(1, 784) / 255.0

    # Predict
    prediction = svm_clf.predict(img_array)[0]
    st.success(f"Predicted Digit: {prediction}")
