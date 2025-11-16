import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import shap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# --- Setup Output Directory ---
output_folder = 'torch_nn_outputs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

# --- Data Loading and Preparation ---
df = pd.read_csv('dataset.csv')
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train.values)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val.values)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test.values)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# --- PyTorch Model Definition ---
class Net(nn.Module):
    def __init__(self, input_features, hidden_layers):
        super(Net, self).__init__()
        layers = []
        prev_size = input_features
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.Sigmoid())
            prev_size = size
        layers.append(nn.Linear(prev_size, 2)) # 2 output classes
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- Hyperparameter Tuning ---
print("--- Finding Best Hidden Layer Size for PyTorch MLP ---")
layer_sizes = [(100, 50, 25), (50,50,10), (15,10,5)]
val_scores = []

for size in layer_sizes:
    print(f"\nTraining with hidden_layer_sizes: {size}")
    model = Net(input_features=X_train.shape[1], hidden_layers=size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Simple training loop
    for epoch in range(50): # Number of epochs for tuning
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Print status every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"  Epoch [{epoch+1:2d}/50], Loss: {avg_loss:.4f}")

    # Validation
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        _, predicted = torch.max(val_outputs.data, 1)
        accuracy = (predicted == y_val_tensor).sum().item() / len(y_val_tensor)
        print(f"  Validation Accuracy: {accuracy:.4f}")
        val_scores.append(accuracy)

best_size = layer_sizes[np.argmax(val_scores)]
print(f"Best hidden_layer_sizes found: {best_size}")

# --- Training Final Model ---
print("\n--- Training Final Neural Network Model ---")
final_model = Net(input_features=X_train.shape[1], hidden_layers=best_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.005)

for epoch in range(100): # More epochs for final model
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = final_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1:3d}/100], Loss: {avg_loss:.4f}")

# --- Final Model Evaluation on the Test Set ---
print("\n--- Final Evaluation on Test Set ---")
final_model.eval() # Set model to evaluation mode
with torch.no_grad():
    y_pred_tensor = final_model(X_test_tensor)
    _, y_pred = torch.max(y_pred_tensor, 1)

y_pred = y_pred.numpy()
y_test_numpy = y_test.to_numpy()

accuracy = accuracy_score(y_test_numpy, y_pred)
f1 = f1_score(y_test_numpy, y_pred, average='weighted')
print(f"Model Accuracy on Test Set: {accuracy}")
print(f"F1 Score on Test Set: {f1}")

# --- Model Interpretation and Explainability with SHAP ---
print("\n--- Generating SHAP Feature Importance ---")
# DeepExplainer is more efficient for deep learning models
background = X_train_tensor[np.random.choice(X_train_tensor.shape[0], 100, replace=False)]
explainer = shap.DeepExplainer(final_model, background)
shap_values = explainer.shap_values(X_test_tensor)

# Plotting SHAP Feature Importance (for class 1)
plt.figure()
shap.summary_plot(shap_values[1], features=X_test_tensor, feature_names=X.columns, plot_type="bar", show=False)
plt.title('SHAP Feature Importance for Neural Network (Class 1)')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'torch_nn_shap_importance.png'), dpi=300)
print(f"\nSaved SHAP feature importance plot to {output_folder}/torch_nn_shap_importance.png")
plt.close()

# --- Confusion Matrix ---
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test_numpy, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix for PyTorch Neural Network')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(output_folder, 'torch_nn_confusion_matrix.png'), dpi=300)
print(f"Saved confusion matrix plot to {output_folder}/torch_nn_confusion_matrix.png")