import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image

st.title("💾 Save MNIST CSV as Images")

# Upload CSV file
uploaded_file = st.file_uploader("Upload train.csv", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    train_data = pd.read_csv(uploaded_file)
    st.success(f"CSV loaded! Shape: {train_data.shape}")

    # Output folder
    output_dir = st.text_input(
        "Enter folder to save images", value="MNIST_Images")
    os.makedirs(output_dir, exist_ok=True)

    # Start button
    if st.button("Save Images"):
        progress_bar = st.progress(0)
        percentage_text = st.empty()
        total = len(train_data)

        for i in range(total):
            label = train_data.iloc[i, 0]            # Digit label
            pixels = train_data.iloc[i, 1:].values   # Pixel data

            # Convert to 28x28 array and uint8
            img_array = pixels.reshape(28, 28).astype(np.uint8)

            # Convert to PIL Image
            img = Image.fromarray(img_array, mode='L')

            # Create folder per label
            folder_name = os.path.join(output_dir, f"Image_{label}")
            os.makedirs(folder_name, exist_ok=True)

            # Save image
            file_name = f"image_{label}_{i}.png"
            img.save(os.path.join(folder_name, file_name))

            # Update progress bar and percentage every 100 images
            if i % 100 == 0 or i == total - 1:
                progress = (i + 1) / total
                progress_bar.progress(progress)
                percentage_text.text(f"Progress: {progress*100:.2f}%")

        progress_bar.progress(1.0)
        percentage_text.text("Progress: 100%")
        st.success(
            f"✅ All {total} images saved successfully in '{output_dir}'!")
