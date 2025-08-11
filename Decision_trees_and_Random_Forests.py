import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#  Load Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

#  Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Train Decision Tree (no depth limit)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predict & evaluate
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

#  Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree (no depth limit)")
plt.show()

#  Analyze overfitting → limit tree depth
dt_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_pruned.fit(X_train, y_train)

y_pred_pruned = dt_pruned.predict(X_test)
print("\nPruned Decision Tree Accuracy:", accuracy_score(y_test, y_pred_pruned))

# Compare train vs test accuracy to detect overfitting
train_acc = accuracy_score(y_train, dt.predict(X_train))
test_acc = accuracy_score(y_test, y_pred_dt)
print(f"Overfitting Check → Train: {train_acc:.2f}, Test: {test_acc:.2f}")

#  Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

#  Feature Importances
feature_importance = pd.Series(rf.feature_importances_, index=iris.feature_names)
feature_importance.sort_values(ascending=False).plot(kind="bar", color="teal")
plt.title("Feature Importances (Random Forest)")
plt.show()

#  Cross-validation for robust accuracy estimation
cv_scores_dt = cross_val_score(dt_pruned, X, y, cv=5)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)
print("\nCross-Validation Accuracy (Decision Tree, depth=3):", np.mean(cv_scores_dt))
print("Cross-Validation Accuracy (Random Forest):", np.mean(cv_scores_rf))

#  Confusion Matrix for Random Forest
print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))
