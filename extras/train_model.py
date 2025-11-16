import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

# Read data from dataset.csv
df = pd.read_csv('dataset.csv')

# Assuming the last column is the target variable and others are features
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Split data into training, validation, and testing sets (80%, 10%, 10%)
# First, split into training (80%) and a temporary set (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
# Then, split the temporary set into validation (10%) and testing (10%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize and fit a Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_val)

# Calculate and print the F1 score
f1 = f1_score(y_val, y_pred, average='weighted') # Use 'weighted' for multi-class or imbalanced binary
print(f"F1 Score: {f1}")
print(f"Model Accuracy: {model.score(X_test, y_test)}")
print(f"Depth: {model.tree_.max_depth}")

# --- Model Interpretation and Explainability ---

# 1. Feature Importance
print("\n--- Feature Importance ---")
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print(feature_importance_df)

# Plotting Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance from Decision Tree')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("\nSaved feature importance plot to feature_importance.png")

# 2. Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=300)
print("Saved confusion matrix plot to confusion_matrix.png")

# 3. Decision Tree Visualization
print("\n--- Visualizing Decision Tree ---")
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=feature_names, class_names=[str(c) for c in model.classes_], filled=True, rounded=True, max_depth=3)
plt.title("Decision Tree Structure (Top 3 Levels)")
plt.savefig('decision_tree.png', dpi=300)
print("Saved decision tree visualization to decision_tree.png")

# To display the plots if running in an interactive environment
# plt.show()
