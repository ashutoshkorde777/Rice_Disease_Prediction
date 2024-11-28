from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


# Load the saved model
model = load_model("rice_disease_model.keras")


# Path to the test image
test_image_path = os.path.join(os.getcwd(), "1.jpg")

if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"File not found: {test_image_path}")

# Preprocess the image
def preprocess_image(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)  # Read the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, img_size)  # Resize to the required size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Preprocess the test image
input_image = preprocess_image(test_image_path)

# Make prediction
predictions = model.predict(input_image)
predicted_class = np.argmax(predictions)

# Print the predicted class
print(f"Predicted Class: {predicted_class}")

