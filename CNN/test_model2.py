import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# ✅ Load the trained MobileNetV2 model
MODEL_PATH = "transfer_skin_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Class names (ensure these match your training folders)
class_names = [
    'eczema', 'acne', 'psoriasis', 'measles', 'chickenpox',
    'impetigo', 'ringworm', 'hives', 'scabies',
    'dermatitis', 'rosacea', 'folliculitis'
]

# ✅ Path to test image
image_path = "test1.jpg"  # 🔁 Replace with your test image path

# ✅ Ensure image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ File not found: {image_path}")

# ✅ Load and preprocess image for MobileNetV2
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# ✅ Predict
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction[0])
predicted_label = class_names[predicted_index]
confidence = prediction[0][predicted_index] * 100

# ✅ Show result
print(f"✅ Predicted Skin Condition: {predicted_label} ({confidence:.2f}% confidence)")

plt.imshow(img)
plt.axis('off')
plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}%")
plt.show()
