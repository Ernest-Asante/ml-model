import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Load dataset
df = pd.read_csv("disease_symptoms(2).csv")

# Fill NaN values with an empty string
df.fillna("", inplace=True)

# Convert symptoms into a list (removing empty values)
df["Symptoms"] = df.iloc[:, 1:].values.tolist()
df["Symptoms"] = df["Symptoms"].apply(lambda x: [s for s in x if s])

# One-hot encode symptoms
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])

# Encode disease labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Disease"])

# --- Train Decision Tree on ALL DATA ---
dt_model = DecisionTreeClassifier()
dt_model.fit(X, y)

# --- Train Neural Network on ALL DATA ---
nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000)
nn_model.fit(X, y)

# --- Function to Predict Disease Based on Symptoms ---
def predict_disease(model, symptoms):
    try:
        input_vector = mlb.transform([symptoms])
        prediction = model.predict(input_vector)
        disease = label_encoder.inverse_transform(prediction)[0]
        return disease
    except Exception as e:
        return f"Error: {str(e)}"

# --- Test with Your Own Symptoms ---
user_symptoms = ["cough","dry mouth","increased thirst","diarrhea"]

print("\nTesting with symptoms:", user_symptoms)
print("Decision Tree Prediction:", predict_disease(dt_model, user_symptoms))
print("Neural Network Prediction:", predict_disease(nn_model, user_symptoms))
