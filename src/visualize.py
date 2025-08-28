
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
DATA_PATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
OUTPUT_DIR = 'output/visualizations'

# --- Create output directory if it does not exist ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load data ---
try:
    df = pd.read_csv(DATA_PATH)
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: File not found at '{DATA_PATH}'.")
    exit()

# --- Convert 'TotalCharges' to numeric (handling errors) ---
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Replace NaNs with the median to avoid data loss
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# --- 2. Configure plot style ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12

# --- 3. Visualizations ---

# A. Distribution of the target variable 'Churn'
plt.figure()
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title("Distribution of Churn")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.savefig(os.path.join(OUTPUT_DIR, 'churn_distribution.png'))
print("📊 Churn distribution graph saved.")

# B. Relationship between gender and churn
plt.figure()
sns.countplot(x='gender', hue='Churn', data=df, palette='pastel')
plt.title("Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.savefig(os.path.join(OUTPUT_DIR, 'churn_by_gender.png'))
print("📊 Churn by gender graph saved.")

# C. Relationship between contract type and churn
plt.figure()
sns.countplot(x='Contract', hue='Churn', data=df, palette='plasma')
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.savefig(os.path.join(OUTPUT_DIR, 'churn_by_contract.png'))
print("📊 Churn by contract graph saved.")

# D. Distribution of tenure in relation to churn
plt.figure()
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', kde=True, palette='coolwarm')
plt.title("Distribution of Tenure by Churn Status")
plt.xlabel("Tenure (months)")
plt.ylabel("Number of Customers")
plt.savefig(os.path.join(OUTPUT_DIR, 'churn_by_tenure.png'))
print("📊 Churn by tenure graph saved.")

# E. Distribution of monthly charges in relation to churn
plt.figure()
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='autumn')
plt.title("Distribution of Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges ($)")
plt.savefig(os.path.join(OUTPUT_DIR, 'churn_by_monthly_charges.png'))
print("📊 Churn by monthly charges graph saved.")

# --- Display plots ---
print("\n🚀 Displaying plots. Close the windows to end the script.")
plt.show()

print("\n✅ Visualization script finished.")
