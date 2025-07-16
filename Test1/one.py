import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from scipy.stats import entropy

# Load cleaned dataset
df = pd.read_csv("cleaned_dataset_cleaned.csv")

# Convert symptoms into a list and remove spaces
df["All_Symptoms"] = df["All_Symptoms"].apply(lambda x: [s.strip() for s in x.split(",")])

# Encode symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["All_Symptoms"])
y = df["Disease"]

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
rf_model.fit(X, y)

# Bayesian Network Training
symptom_columns = mlb.classes_
bn_structure = [("Disease", symptom) for symptom in symptom_columns]
model = BayesianNetwork(bn_structure)
model.fit(pd.concat([y, pd.DataFrame(X, columns=symptom_columns)], axis=1), estimator=MaximumLikelihoodEstimator)

# Bayesian Inference
inference = VariableElimination(model)

def predict_top_diseases(symptoms, model, mlb, top_n=5):
    """Predict the top diseases from given symptoms."""
    symptoms = [s.strip() for s in symptoms if s.strip() in mlb.classes_]  # Remove spaces & unknowns
    if not symptoms:
        print("⚠️ No valid symptoms found! Check input.")
        return []
    
    symptom_vector = mlb.transform([symptoms])
    probs = model.predict_proba(symptom_vector)[0]
    classes = model.classes_
    top_indices = np.argsort(probs)[-top_n:][::-1]
    return [(classes[i], round(probs[i], 2)) for i in top_indices]

def refine_diagnosis(disease_candidates, known_symptoms):
    """Refine diagnosis using Bayesian Network by considering more evidence."""
    evidence = {symptom: 1 for symptom in known_symptoms if symptom in symptom_columns}
    probabilities = inference.map_query(variables=["Disease"], evidence=evidence)
    return probabilities

def next_best_symptom(known_symptoms):
    """Find the most informative symptom to ask next."""
    remaining_symptoms = [s for s in symptom_columns if s not in known_symptoms]
    if not remaining_symptoms:
        return None  # No more symptoms to ask
    
    entropy_scores = {}
    for symptom in remaining_symptoms:
        try:
            evidence = {**{s: 1 for s in known_symptoms}, symptom: 1}
            disease_probs = inference.query(variables=["Disease"], evidence=evidence).values
            entropy_scores[symptom] = entropy(disease_probs)
        except:
            continue  # Skip if inference fails

    if not entropy_scores:
        return None
    return min(entropy_scores, key=entropy_scores.get)  # Ask symptom with lowest entropy

def interactive_diagnosis(initial_symptoms):
    """Perform interactive diagnosis using symptoms and Bayesian refinement."""
    print(f"🔎 Checking symptoms: {initial_symptoms}")
    
    known_symptoms = set(initial_symptoms)
    top_diseases = predict_top_diseases(list(known_symptoms), rf_model, mlb)

    if not top_diseases:
        print("❌ Error: No diseases predicted. Check symptom list.")
        return
    
    print("Initial probable diseases:", top_diseases)

    while True:
        refined_diagnosis = refine_diagnosis([d[0] for d in top_diseases], known_symptoms)
        print("Current refined diagnosis:", refined_diagnosis)
        
        if refined_diagnosis.get("Disease"):
            next_symptom = next_best_symptom(known_symptoms)
            print(f"Next symptom to ask: {next_symptom}")  # Debugging print
            
            if next_symptom:
                response = input(f"Do you have {next_symptom.replace('_', ' ')}? (yes/no): ").strip().lower()
                if response == "yes":
                    known_symptoms.add(next_symptom)
                else:
                    print("Skipping to next possible symptom...")
            else:
                print("✅ Final Diagnosis:", refined_diagnosis["Disease"])
                break
        else:
            print("🤷 Diagnosis uncertain. No strong match found.")
            break

# Example usage
initial_symptoms = ["skin_rush", "scurring"]  # Remove spaces before passing
interactive_diagnosis(initial_symptoms)
