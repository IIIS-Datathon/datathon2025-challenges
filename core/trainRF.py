import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# --- Setup Output Directory ---
output_folder = '../extras/rf_outputs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

# Read data from dataset.csv
df = pd.read_csv('../data/dataset.csv')

# Assuming the last column is the target variable and others are features
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Split data into training, validation, and testing sets (80%, 10%, 10%)
# First, split into training (80%) and a temporary set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Hyperparameter Tuning (Pruning) ---
# Find the best max_depth using the validation set to prevent overfitting
# print("--- Finding Best Max Depth for Random Forest ---")
# depths = range(30, 40) 
# val_scores = []
# for depth in depths:
#     # Using n_estimators=100 is a common and robust starting point
#     model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42, n_jobs=-1)
#     model.fit(X_train, y_train)
#     val_scores.append(model.score(X_val, y_val))

best_depth = 35
print(f"Best max_depth found: {best_depth}")

# Initialize and fit the final model with the best hyperparameter
print("\n--- Training Final Random Forest Model ---")
model = RandomForestClassifier(n_estimators=65, max_depth=best_depth, random_state=42, min_samples_split=6, min_samples_leaf= 1, n_jobs=-1)
model.fit(X_train, y_train)

# --- Final Model Evaluation on the Test Set ---
print("\n--- Final Evaluation on Test Set ---")
y_pred = model.predict(X_test)

# Calculate and print key metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Model Accuracy on Test Set: {accuracy}")
print(f"F1 Score on Test Set: {f1}")

# Define the filename for the saved model
model_filename = os.path.join('../model', 'random_forest_model.pkl')

# Save the model to a .pkl file
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model successfully saved to {model_filename}")

# 2. Standard Feature Importance
print("\n--- Feature Importance ---")
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print(feature_importance_df)
with open(os.path.join('../model', 'objs.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([X_train, y_train, X_test, y_test, feature_names], f)

# Plotting Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'rf_feature_importance.png'), dpi=300)
print(f"\nSaved feature importance plot to {output_folder}/rf_feature_importance.png")

# 3. Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix for Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(output_folder, 'rf_confusion_matrix.png'), dpi=300)
print(f"Saved confusion matrix plot to {output_folder}/rf_confusion_matrix.png")