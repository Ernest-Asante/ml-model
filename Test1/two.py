import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Load cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# Convert symptoms into a list
df["All_Symptoms"] = df["All_Symptoms"].apply(lambda x: x.split(","))

# Encode symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["All_Symptoms"])
y = df["Disease"]

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
rf_model.fit(X, y)

# Predict top 5 probable diseases
def predict_top_diseases(symptoms, model, mlb, top_n=5):
    # Filter valid symptoms
    symptoms = [s for s in symptoms if s in mlb.classes_]
    if not symptoms:
        print("⚠️ No valid symptoms found in dataset! Please check input.")
        return []
    
    # Transform symptoms into feature vector
    symptom_vector = mlb.transform([symptoms])
    
    # Get probability predictions
    probs = model.predict_proba(symptom_vector)[0]
    classes = model.classes_
    
    # Get top 5 diseases
    top_indices = np.argsort(probs)[-top_n:][::-1]
    return [(classes[i], round(probs[i], 2)) for i in top_indices]

# Example usage
input_symptoms = [" fatigue", " high_fever", " cough"]  # Change this to your symptoms
rf_predictions = predict_top_diseases(input_symptoms, rf_model, mlb)

print("🔹 Top 5 Predicted Diseases:", rf_predictions)
