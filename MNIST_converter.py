import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image

st.title("💾 Convert MNIST CSV to Images")

# Button to start conversion
if st.button("Convert CSV to Images"):
    csv_path = "digit-recognizer/train.csv"

    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at {csv_path}")
    else:
        train_data = pd.read_csv(csv_path)
        st.success(f"CSV loaded! Shape: {train_data.shape}")

        # Output folder
        output_dir = "MNIST_Images"
        os.makedirs(output_dir, exist_ok=True)

        # Progress bar and percentage display
        progress_bar = st.progress(0)
        percentage_text = st.empty()  # Placeholder for percentage
        total = len(train_data)

        # Loop through dataset
        for i in range(total):
            label = train_data.iloc[i, 0]
            pixels = train_data.iloc[i, 1:].values

            # Convert to 28x28 and uint8
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
