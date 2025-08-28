import pandas as pd
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
from tabulate import tabulate
import joblib
import os

# --- 1. Load Data ---
print("Loading data...")
df = pd.read_csv(r"data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# --- 2. Preprocessing ---
print("Preprocessing data...")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.drop("customerID", axis=1, inplace=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Encode target variable
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Identify categorical features for CatBoost
categorical_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']

# --- 3. Data Splitting ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- 4. Model Training ---
print("Training CatBoost model...")
start_time = time.time()

# Initialize CatBoostClassifier with best parameters from tuning
# You can adjust these parameters based on your tuning results
cat_model = CatBoostClassifier(
    iterations=100,
    depth=4,
    learning_rate=0.2,
    l2_leaf_reg=1,
    cat_features=categorical_features_indices,
    verbose=False,
    random_state=42
)

cat_model.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# --- 5. Model Evaluation ---
print("Evaluating model performance...")
preds = cat_model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, pos_label=1)

results = [["CatBoost", acc, f1, training_time]]
headers = ["Model", "Accuracy", "F1 Score", "Training Time (s)"]
print("\n--- Model Performance ---")
print(tabulate(results, headers=headers, floatfmt=".4f", tablefmt="grid"))

# --- 6. Save Model ---
print("Saving the trained model...")
output_dir = 'src/models'
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, 'catboost_churn_model.joblib')
joblib.dump(cat_model, model_path)
print(f"Model saved to {model_path}")
