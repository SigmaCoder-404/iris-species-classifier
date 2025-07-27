import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("iris.csv")
df.drop("Id", axis=1, inplace=True)

# Define features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numeric pipeline
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combine all preprocessing
preprocessing = ColumnTransformer([
    ('num', num_pipe, X.columns)
])

# KNN Model pipeline
model = Pipeline([
    ('preprocessing', preprocessing),
    ('model', KNeighborsClassifier())
])

# Decision Tree pipeline
model2 = Pipeline([
    ('preprocessing', preprocessing),
    ('model2', DecisionTreeClassifier(random_state=42))
])

# KNN training and evaluation
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("KNN Cross-validation score:", cross_val_score(model, X, y, cv=5).mean())
print("KNN Classification Report:\n", classification_report(y_test, y_pred))

# Decision Tree training and evaluation
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
print("Decision Tree Cross-validation score:", cross_val_score(model2, X, y, cv=5).mean())
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred2))

# Visualize decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(model2.named_steps["model2"], feature_names=X.columns, class_names=model2.named_steps["model2"].classes_, filled=True)
plt.show()
