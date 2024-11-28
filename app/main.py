from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from io import BytesIO
from fastapi.staticfiles import StaticFiles
from fastapi import Request

app = FastAPI()

# Load the saved model
model = load_model("rice_disease_model.keras")

# Define categories
categories = [
    "Bacterial Blight Disease",
    "Blast Disease",
    "Brown Spot Disease",
    "False Smut Disease",
]

# Preprocess the image
def preprocess_image(image_data, img_size=(128, 128)):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)  # Read image from buffer
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, img_size)  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Serve HTML and static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file and preprocess it
        image_data = await file.read()
        input_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(input_image)
        predicted_class = np.argmax(predictions)
        
        # Return the predicted disease name
        return {
            "Predicted Class Index": int(predicted_class),
            "Predicted Disease": categories[predicted_class],
        }
    except Exception as e:
        return {"error": str(e)}
