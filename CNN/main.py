# main.py
import uvicorn
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

# Load model and classes
model = tf.keras.models.load_model("transfer_skin_model.h5")
class_names = [
    'eczema', 'acne', 'psoriasis', 'measles', 'chickenpox',
    'impetigo', 'ringworm', 'hives', 'scabies',
    'dermatitis', 'rosacea', 'folliculitis'
]

# Setup FastAPI app
app = FastAPI()

# Enable CORS for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Expo dev server IP in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper: process image
def read_image(file) -> np.ndarray:
    img = Image.open(BytesIO(file)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Endpoint: Predict skin condition
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    input_image = read_image(contents)
    predictions = model.predict(input_image)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 2)
    }

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
