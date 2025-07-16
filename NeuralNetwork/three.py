import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier

# --- Load and Clean Dataset ---
file_path = "modified_dataset.csv"
df = pd.read_csv(file_path)

# Convert "All_Symptoms" into a list of symptoms
df["Symptoms"] = df["All_Symptoms"].apply(lambda x: [s.strip().lower() for s in x.split(",")])

# Drop the old "All_Symptoms" column
df = df[["Disease", "Symptoms"]]

# --- One-hot Encode Symptoms ---
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])  # Convert symptoms into binary format

# Encode disease labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Disease"])  # Convert diseases to numbers

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# --- Define Neural Network Model ---
class DiseaseNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=None):
        super(DiseaseNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output raw logits
        return x

# Initialize model
input_size = X.shape[1]
output_size = len(np.unique(y))
nn_model = DiseaseNN(input_size, output_size=output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# --- Train the Neural Network ---
num_epochs = 1000
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    outputs = nn_model(X_tensor)
    loss = criterion(outputs, y_tensor)  
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

# --- Train Decision Tree Model ---
dt_model = DecisionTreeClassifier()
dt_model.fit(X, y)

# --- Function to Predict Disease ---
def predict_disease(model, symptoms, use_torch=False):
    known_symptoms = [s.strip().lower() for s in symptoms if s.strip().lower() in mlb.classes_]

    if not known_symptoms:
        return "No known symptoms provided. Unable to make a prediction."

    input_vector = mlb.transform([known_symptoms])
    
    if use_torch:
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        with torch.no_grad():
            output = nn_model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        disease = label_encoder.inverse_transform([predicted_class])[0]
    else:
        prediction = model.predict(input_vector)
        disease = label_encoder.inverse_transform(prediction)[0]

    return disease

# --- Debugging: Print Available Symptoms ---
print("Available Symptoms in Model:", list(mlb.classes_))

# --- Test with Your Own Symptoms ---
user_symptoms = ["chills", "vomiting", "weight_loss", "cough"]

print("\nTesting with symptoms:", user_symptoms)
print("Decision Tree Prediction:", predict_disease(dt_model, user_symptoms))
print("Neural Network Prediction:", predict_disease(nn_model, user_symptoms, use_torch=True))
