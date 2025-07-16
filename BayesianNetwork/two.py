import torch
import torch.nn as nn
import torch.optim as optim
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("disease_symptoms_only.csv")

# Encode categorical values
df["disease"] = df["disease"].astype('category').cat.codes  # Convert diseases to numerical labels

# Split data into features and labels
X = df.drop(columns=["disease"]).values
y = df["disease"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define Bayesian model using Pyro
def bayesian_model(x, y=None):
    num_features = x.shape[1]
    num_classes = y_train.max().item() + 1  # Get number of classes

    with pyro.plate("layer1", 128):
        weight1 = pyro.sample("weight1", dist.Normal(torch.zeros(num_features), torch.ones(num_features)))

    with pyro.plate("layer2", num_classes):
        weight2 = pyro.sample("weight2", dist.Normal(torch.zeros(128), torch.ones(128)))

    hidden = torch.relu(torch.matmul(x, weight1.T))  # [batch, 128]
    output = torch.softmax(torch.matmul(hidden, weight2.T), dim=1)  # [batch, num_classes]

    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Categorical(output), obs=y)

# Define guide function
def guide(x, y=None):
    num_features = x.shape[1]
    num_classes = y_train.max().item() + 1

    weight1_mu = pyro.param("weight1_mu", torch.randn(128, num_features))
    weight1_sigma = pyro.param("weight1_sigma", torch.ones(128, num_features), constraint=dist.constraints.positive)
    
    weight2_mu = pyro.param("weight2_mu", torch.randn(num_classes, 128))
    weight2_sigma = pyro.param("weight2_sigma", torch.ones(num_classes, 128), constraint=dist.constraints.positive)

    with pyro.plate("layer1", 128):
        pyro.sample("weight1", dist.Normal(weight1_mu, weight1_sigma))

    with pyro.plate("layer2", num_classes):
        pyro.sample("weight2", dist.Normal(weight2_mu, weight2_sigma))

# Training the BNN
optimizer = Adam({"lr": 0.01})
svi = SVI(bayesian_model, guide, optimizer, loss=Trace_ELBO())

num_epochs = 500
for epoch in range(num_epochs):
    loss = svi.step(X_train, y_train)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Predict on new data
def predict(X_new):
    predictive = pyro.infer.Predictive(bayesian_model, guide=guide, num_samples=1000)
    samples = predictive(X_new)
    return torch.argmax(samples["obs"].float().mean(dim=0), dim=1)

predictions = predict(X_test)
print("Predictions:", predictions[:10])
