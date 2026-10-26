# =====================================================================
# IMPORT REQUIRED LIBRARIES
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Style for better looking graphs
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

# =====================================================================
# LOAD & EXPLORE DATASET
# =====================================================================
print(" Ingesting Food Safety Dataset...")
df = pd.read_csv("food_safety.csv")

print(f" Dataset successfully loaded! Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")

# Clean missing values if any
df = df.dropna()

# =====================================================================
# VISUALIZATIONS (ENHANCED GRAPHS)
# =====================================================================
print(" Generating and saving beautiful charts...")

# 1. Product Approval Count Plot
plt.figure(figsize=(6, 5))
sns.countplot(x="product_approval", data=df, palette=["#2ecc71", "#e74c3c"])
plt.title("Product Approval Distribution (Approved vs Rejected)", pad=15)
plt.xlabel("Approval Status")
plt.ylabel("Number of Products")
plt.tight_layout()
plt.savefig("graph1_approval_count.png", dpi=300)
plt.close()

# 2. Histogram of Inspection Score
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="inspection_score_0_100", hue="product_approval", 
             kde=True, bins=15, palette=["#2ecc71", "#e74c3c"], multiple="stack")
plt.title("Distribution of Inspection Scores", pad=15)
# Set exact limit for score
plt.xlim(0, 100) 
plt.xlabel("Inspection Score (0-100)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("graph2_inspection_score_histogram.png", dpi=300)
plt.close()

# 3. Scatter Plot: Temperature vs Inspection Score
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="storage_temperature_C", y="inspection_score_0_100", 
                hue="product_approval", palette=["#2ecc71", "#e74c3c"], s=70, alpha=0.8)
plt.title("Storage Temperature vs Inspection Score", pad=15)
plt.xlabel("Storage Temperature (°C)")
plt.ylabel("Inspection Score (0-100)")
plt.legend(title="Status")
plt.tight_layout()
plt.savefig("graph3_temp_vs_inspection_scatter.png", dpi=300)
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(9, 7))
numeric_df = df.select_dtypes(include="number")
mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool)) # Elegant half-heatmap
sns.heatmap(numeric_df.corr(), annot=True, mask=mask, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Food Metrics", pad=15)
plt.tight_layout()
plt.savefig("graph4_correlation_heatmap.png", dpi=300)
plt.close()

print(" Graphs saved beautifully as high-quality PNGs!\n")

# =====================================================================
# DATA PREPROCESSING
# =====================================================================
cat_cols = ["product_type", "packaging_type"]
encoders = {}
df_encoded = df.copy()

for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

target_encoder = LabelEncoder()
df_encoded["product_approval"] = target_encoder.fit_transform(df_encoded["product_approval"])

# Target variable mapping info
# 'Approved' -> 0, 'Rejected' -> 1 (or vice-versa depending on data alphabet)
approved_class_value = target_encoder.transform(["Approved"])[0]

X = df_encoded.drop(columns=["product_approval"])
y = df_encoded["product_approval"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================================
# MODEL TRAINING & SELECTION
# =====================================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
}

results = {}
trained_models = {}

print(" Evaluating Machine Learning Models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    trained_models[name] = model
    print(f"   🔹 {name:25s} -> Accuracy: {acc:.4f}")

best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]
best_model = trained_models[best_model_name]

print(f"\n Best Algorithm Found: {best_model_name} ({best_accuracy*100:.2f}% Accuracy)")

# =====================================================================
# SAVE THE BEST MODEL USING PICKLE
# =====================================================================
best_model = models[best_model_name]

model_artifacts = {
    "model": best_model,
    "encoders": encoders,
    "target_encoder": target_encoder,
    "features": list(X.columns)
}
pickle_filename = "best_food_safety_model.pkl"
with open(pickle_filename, "wb") as file:
    pickle.dump(model_artifacts, file)

print(f"SUCCESS: The best model '{best_model_name}' has been saved to '{pickle_filename}' successfully!\n")
# =====================================================================
# SMART PREDICTION LOGIC WITH DETAILED ADVANCED ANALYSIS
# =====================================================================
print(" --- ADVANCED LIVE PREDICTION RUN ---")

# Let's test with a simulated new product data row
# You can change these values to test different situations!
new_product_sample = {
    "product_type": "Dairy",
    "packaging_type": "Plastic",
    "shelf_life_days": 45,
    "storage_temperature_C": 14.5,      # High temperature (>10°C is risky)
    "humidity_level_percent": 75.0,
    "moisture_content": 40.0,
    "preservative_level": 1.2,
    "pH_level": 3.4,                   # Acidic pH
    "inspection_score_0_100": 55,      # Low inspection score
    "customer_rating_1_5": 2
}

# Convert sample to Dataframe format
new_df = pd.DataFrame([new_product_sample])

# Apply the saved LabelEncoders to the new prediction data
for col in cat_cols:
    new_df[col] = encoders[col].transform(new_df[col])

# Predict using our saved model
prediction_encoded = best_model.predict(new_df)[0]
prediction_text = target_encoder.inverse_transform([prediction_encoded])[0]

print(f"\n Input Product Type: {new_product_sample['product_type']}")
print(f" Prediction Result  : {prediction_text.upper()}")

# Safety Check Analysis (Checking critical features manually to show reasons)
print("\n Crucial Metrics Analysis Report:")

if new_product_sample['storage_temperature_C'] > 10:
    print("  WARNING: Storage temperature is high! Bacterial growth risk increased.")
else:
    print("  Temperature is within safe limits.")

if new_product_sample['inspection_score_0_100'] < 70:
    print(f"  WARNING: Low Factory Inspection Score ({new_product_sample['inspection_score_0_100']}/100). Safety violations found.")
else:
    print(" Factory inspection score passed successfully.")

if new_product_sample['pH_level'] < 4.0 or new_product_sample['pH_level'] > 8.0:
    print(f" WARNING: Abnormal pH level ({new_product_sample['pH_level']}). Product might be spoiled or highly acidic/alkaline.")
else:
    print("  pH level is stable.")

print("\n System Processing Complete!")
