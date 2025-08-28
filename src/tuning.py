import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
from tabulate import tabulate

# Load data
df = pd.read_csv(r"data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocessing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.drop("customerID", axis=1, inplace=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

le_y = LabelEncoder()
y = le_y.fit_transform(y)

categorical_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Hyperparameter tuning for CatBoost
print("--- Starting Hyperparameter Tuning for CatBoost ---")
start_time = time.time()

param_grid = {
    'iterations': [100, 300],
    'depth': [4, 6],
    'learning_rate': [0.05, 0.3],
    'l2_leaf_reg': [1, 3]
}

cat_model = CatBoostClassifier(cat_features=categorical_features_indices, verbose=False, random_state=42)

grid_search = GridSearchCV(estimator=cat_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

end_time = time.time()
print(f"Tuning completed in {end_time - start_time:.2f} seconds.")
print("Best parameters found: ", grid_search.best_params_)

# Evaluate the tuned model
best_model = grid_search.best_estimator_
preds = best_model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, pos_label=1)

results = [["Tuned CatBoost", acc, f1, end_time - start_time]]
headers = ["Model", "Accuracy", "F1 Score", "Tuning Time (s)"]
print("\n--- Tuned Model Performance ---")
print(tabulate(results, headers=headers, floatfmt=".4f", tablefmt="grid"))
