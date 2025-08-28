# ====== INSTALL LIBRARIES (uncomment if needed) ======
# pip install pandas scikit-learn xgboost lightgbm catboost

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from tabulate import tabulate

# ====== 1. LOAD DATA ======
df = pd.read_csv(r"data\WA_Fn-UseC_-Telco-Customer-Churn.csv")  # change filename if needed

# Clean TotalCharges (convert to numeric, handle missing)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop customerID (not useful)
df.drop("customerID", axis=1, inplace=True)

# ====== 2. SPLIT FEATURES / TARGET ======
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Encode target variable
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# ====== 3. TRAIN/TEST SPLIT ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ====== 4. ENCODING FOR XGBOOST & LIGHTGBM ======
# Copy to avoid affecting CatBoost
X_train_enc = X_train.copy()
X_test_enc = X_test.copy()

label_encoders = {}
for col in X_train_enc.columns:
    if X_train_enc[col].dtype == "object":
        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train_enc[col])
        X_test_enc[col] = le.transform(X_test_enc[col])
        label_encoders[col] = le

# ====== 5. TRAIN & EVALUATE ======

results = []
def evaluate_model(name, model, X_train, y_train, X_test, y_test, **fit_params):
    start = time.time()
    model.fit(X_train, y_train, **fit_params)
    end = time.time()
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, pos_label=1)
    results.append([name, acc, f1, end - start])

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1
)
evaluate_model("XGBoost", xgb_model, X_train_enc, y_train, X_test_enc, y_test)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=-1, random_state=42, n_jobs=-1
)
evaluate_model("LightGBM", lgb_model, X_train_enc, y_train, X_test_enc, y_test)

# CatBoost
cat_features = [i for i, col in enumerate(X_train.columns) if X_train[col].dtype == "object"]
cat_model = CatBoostClassifier(
    iterations=300, learning_rate=0.1, depth=6, random_state=42, verbose=False
)
evaluate_model("CatBoost", cat_model, X_train, y_train, X_test, y_test, cat_features=cat_features)

# ====== 6. DISPLAY RESULTS ======
headers = ["Model", "Accuracy", "F1 Score", "Training Time (s)"]
print("\n--- Model Comparison ---")
print(tabulate(results, headers=headers, floatfmt=".4f", tablefmt="grid"))
