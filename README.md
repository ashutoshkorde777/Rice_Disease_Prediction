# Rice_Disease_Prediction
This is a FastAPI-based web application for predicting rice diseases using a pre-trained TensorFlow Keras model. The application allows users to upload images of rice leaves and identifies the disease from four common categories.

Features
Homepage: A user-friendly interface to upload images of rice leaves for prediction.
Disease Prediction: Utilizes a deep learning model (rice_disease_model.keras) to identify diseases with high accuracy.
Supported Diseases:
Bacterial Blight Disease
Blast Disease
Brown Spot Disease
False Smut Disease
Static File Hosting: Supports integration of CSS, JavaScript, and other static assets for UI enhancements.
How It Works
Navigate to the homepage (/).
Upload an image of a rice leaf.
The app preprocesses the image and predicts the disease.
Results are displayed, including the predicted disease name.
Requirements
Python 3.8+
FastAPI
TensorFlow/Keras
OpenCV
Jinja2 (for templating)
