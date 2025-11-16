import shap, os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

# --- Setup Output Directory ---
output_folder = '../extras/rf_shap'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

print("\n--- Generating SHAP explanations ---")

with open('../model/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Use the newer shap.Explainer for robustness, which automatically chooses TreeExplainer for tree models.
# The background dataset (X_train) should be representative of the data.
explainer = shap.TreeExplainer(model)

df = pd.read_csv('../data/dataset.csv')
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# To speed up the calculation and explain test set predictions, we sample from X_test.
# X_test_sample_for_shap = shap.sample(X_test, 2, random_state=42)
X_test_sample_for_shap = X_test.copy()
# Explicitly set check_additivity=False to prevent the ExplainerError
shap_values_explanation = explainer(X_test_sample_for_shap, check_additivity=False)

# For binary classification, shap_values_explanation.values will be 3D (num_samples, num_features, num_classes).
# We are interested in the SHAP values for the positive class (class 1).
shap_values_positive_class = shap_values_explanation.values[:, :, 1]

# 1. SHAP Feature Importance Summary Plot
print("\n--- Generating SHAP Feature Importance Summary Plot ---")
plt.figure()
shap.summary_plot(shap_values_positive_class, X_test_sample_for_shap, feature_names=X_test_sample_for_shap.columns.tolist(),
    max_display=20,
    show=False)
plt.title('SHAP Feature Importance for Random Forest (Positive Class)')
plt.savefig(os.path.join(output_folder, 'rf_shap_summary_importance.png'), dpi=300)
print(f"Saved SHAP feature importance summary plot to {output_folder}/rf_shap_summary_importance.png")
plt.close()

with open('../model/shap_values_explanation.pkl', 'wb') as f:
    pickle.dump(shap_values_explanation, f)