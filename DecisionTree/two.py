import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("disease_symptoms_dataset.csv")

# Display dataset preview
print("Dataset Preview:")
print(df.head())

# Separate features (X) and target (y)
X = df.drop("Disease", axis=1)  # Symptoms as features
y = df["Disease"]  # Disease as target

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier with better depth
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Test the model on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict disease from user symptoms
def predict_disease(symptoms):
    """Predicts the disease based on given symptoms."""
    symptoms = np.array(symptoms).reshape(1, -1)  # Reshape input
    prediction = model.predict(symptoms)
    return prediction[0]

# User input testing
print("\nEnter symptoms (1 for Yes, 0 for No):")
user_symptoms = [int(input(f"{col} (1/0): ")) for col in X.columns]

# Predict based on user input
predicted_disease = predict_disease(user_symptoms)
print(f"\nPredicted Disease: {predicted_disease}")

# Plot the Decision Tree AFTER user input test
plt.figure(figsize=(16, 10))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
