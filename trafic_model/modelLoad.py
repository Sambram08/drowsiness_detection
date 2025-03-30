import tensorflow as tf
import numpy as np
import os
# from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

MODEL_PATH = os.path.join(os.path.dirname(__file__),"MLmodel","drowsiness_model.h5")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
        
        
# Class labels (ensure they match your model's output classes)
CLASS_NAMES = [
    'Close-Eyes', 'Open-Eyes',
]

IMG_SIZE = (64, 64)

def predict(image):
    '''
    Predicts the drowsiness class of the input image.

    Args:
        image (PIL.Image.Image): The image to classify.

    Returns:
        dict: The predicted class and confidence score.
    '''
    if model is None:
        return {"error": "Model not loaded. Check the path and model file."}
    
    try:
        # Convert PIL Image to NumPy array
        image = np.array(image.convert("L"))  # Convert to grayscale

        # Resize and normalize the image
        image = cv2.resize(image, IMG_SIZE)
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=[0, -1])  # Add batch & channel dimensions

        # Get prediction
        prediction = model.predict(image)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        return {"class": predicted_class, "confidence": float(confidence)}

    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

        
        