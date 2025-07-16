import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# --- Load Dataset ---
df = pd.read_csv("disease_symptoms(2).csv")

# Fill NaN values with an empty string
df.fillna("", inplace=True)

# Convert symptoms into a list (removing empty values)
df["Symptoms"] = df.iloc[:, 1:].values.tolist()
df["Symptoms"] = df["Symptoms"].apply(lambda x: [s for s in x if s])

# Flatten symptoms to create a column for each symptom
all_symptoms = set(s for symptoms in df["Symptoms"] for s in symptoms)
for symptom in all_symptoms:
    df[symptom] = df["Symptoms"].apply(lambda x: 1 if symptom in x else 0)

# Keep only relevant columns (Disease + Symptoms)
columns = ["Disease"] + list(all_symptoms)
df = df[columns]

# --- Build Bayesian Network Structure ---
hc = HillClimbSearch(df)
best_model = hc.estimate(scoring_method=BicScore(df))

# Define Bayesian Network Model
bayesian_model = BayesianNetwork(best_model.edges())

# Fit the model using Maximum Likelihood Estimation
bayesian_model.fit(df, estimator=MaximumLikelihoodEstimator)

# Perform Inference
inference = VariableElimination(bayesian_model)

# --- Function to Predict Disease ---
def predict_disease_bn(symptoms):
    try:
        query_evidence = {s: 1 for s in symptoms if s in df.columns}
        
        if not query_evidence:
            return "No known symptoms provided. Unable to make a prediction."
        
        prediction = inference.map_query(["Disease"], evidence=query_evidence)
        return prediction["Disease"] if prediction else "Unknown Disease"
    except Exception as e:
        return f"Error: {str(e)}"

# --- Test with Your Own Symptoms ---
user_symptoms = ["cough", "dry mouth", "increased thirst", "diarrhea", "Fever"]

print("\nTesting with symptoms:", user_symptoms)
print("Bayesian Network Prediction:", predict_disease_bn(user_symptoms))
