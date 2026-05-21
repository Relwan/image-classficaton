"""
Project 1: Image Height Classification System (IMT 503 Deployment)
Description: Standalone inference script that loads the trained weights 
             and predicts if a person in an image is Short, Moderate, or Tall.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Configuration and Paths
MODEL_PATH = 'height_classifier_model.h5'
IMG_SIZE = (224, 224)
LABELS = ['Moderate', 'Short', 'Tall']

def run_inference():
    print("=========================================")
    print("     Image Height Classification System   ")
    print("=========================================\n")
    
    # Check if the model file exists in the directory
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model weights file '{MODEL_PATH}' not found in this directory.")
        return

    # Load the pre-trained weights
    print("🔄 Loading trained AI model weights...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.\n")

    # Ask user for the local path of the image they want to test
    img_filename = input("Enter the path/name of the image file to classify (e.g., test.jpg): ").strip()
    
    if not os.path.exists(img_filename):
        print(f"❌ Error: Image file '{img_filename}' could not be found.")
        return

    try:
        # Load and resize image to match training parameters
        img = image.load_img(img_filename, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Create batch dimensions

        print("\n🧠 Analysing image features...")
        predictions = model.predict(img_array, verbose=0)
        score_index = np.argmax(predictions[0])
        confidence = 100 * np.max(predictions[0])

        # Output the classification results
        print("=" * 40)
        print(f"🎯 Result: The AI predicts this person is {LABELS[score_index]}")
        print(f"📊 Confidence Score: {confidence:.2f}%")
        print("=" * 40)

    except Exception as e:
        print(f"❌ An error occurred during classification: {e}")

if __name__ == "__main__":
    run_inference()