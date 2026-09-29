import pandas as pd
import numpy as np

# ==========================================
# ROUND 2 - FEATURE ENGINEERING
# ==========================================

# Load dataset
df = pd.read_csv("Dataset/train.csv")

print("=" * 60)
print("ROUND 2 - FEATURE ENGINEERING")
print("=" * 60)

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nTarget:")
print(df["Irrigation_Need"].value_counts())



# ==========================================
# FEATURE ENGINEERING
# ==========================================

print("\n" + "=" * 60)
print("CREATING ENGINEERED FEATURES")
print("=" * 60)

# 1. Soil Moisture Deficit
df["Moisture_Deficit"] = 100 - df["Soil_Moisture"]

# 2. Temperature-Humidity Interaction
df["Temperature_Humidity_Index"] = (
    df["Temperature_C"] * (100 - df["Humidity"])
)

# 3. Heat-Sunlight Stress
df["Heat_Sunlight_Stress"] = (
    df["Temperature_C"] * df["Sunlight_Hours"]
)

# 4. Rainfall and Previous Irrigation
df["Water_Availability"] = (
    df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
)

# 5. Evaporation Risk
df["Evaporation_Risk"] = (
    df["Temperature_C"]
    + df["Wind_Speed_kmh"]
    + df["Sunlight_Hours"]
    - df["Humidity"] / 10
)

# 6. Moisture-Rainfall Stress
df["Moisture_Rainfall_Stress"] = (
    (100 - df["Soil_Moisture"])
    / (df["Rainfall_mm"] + 1)
)

# 7. Irrigation History Ratio
df["Previous_Irrigation_Ratio"] = (
    df["Previous_Irrigation_mm"]
    / (df["Rainfall_mm"] + 1)
)

print("\nEngineered Features Created:")

engineered_features = [
    "Moisture_Deficit",
    "Temperature_Humidity_Index",
    "Heat_Sunlight_Stress",
    "Water_Availability",
    "Evaporation_Risk",
    "Moisture_Rainfall_Stress",
    "Previous_Irrigation_Ratio"
]

for feature in engineered_features:
    print("-", feature)

print("\nNew Dataset Shape:")
print(df.shape)

print("\nEngineered Feature Sample:")
print(df[engineered_features].head())



# ==========================================
# FEATURE SELECTION & PREPROCESSING
# ==========================================

from sklearn.preprocessing import OneHotEncoder

print("\n" + "=" * 60)
print("FEATURE SELECTION & PREPROCESSING")
print("=" * 60)

# Target
target_column = "Irrigation_Need"

# Remove ID because it is only an identifier
features = df.drop(columns=[target_column, "id"])
y = df[target_column]

# Identify categorical and numerical features
categorical_features = features.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_features = features.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

print("\nNumber of Numerical Features:", len(numerical_features))
print("Number of Categorical Features:", len(categorical_features))

# One-hot encode categorical variables
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_categorical = encoder.fit_transform(
    features[categorical_features]
)

encoded_columns = encoder.get_feature_names_out(
    categorical_features
)

encoded_df = pd.DataFrame(
    encoded_categorical,
    columns=encoded_columns,
    index=features.index
)

# Combine numerical + encoded categorical features
X = pd.concat(
    [
        features[numerical_features].reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ],
    axis=1
)

print("\nFinal Feature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nFinal Feature Sample:")
print(X.head())

print("\nTarget Sample:")
print(y.head())


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Feature Shape:")
print(X_train.shape)

print("\nTesting Feature Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())


# ==========================================
# DECISION TREE MODEL TRAINING
# ==========================================

from sklearn.tree import DecisionTreeClassifier

print("\n" + "=" * 60)
print("DECISION TREE MODEL TRAINING")
print("=" * 60)

# Create Decision Tree
model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
    max_depth=12,
    min_samples_leaf=10
)

print("\nTraining Decision Tree...")

# Train model
model.fit(X_train, y_train)

print("\nDecision Tree Training Completed!")

print("\nModel Parameters:")
print("Algorithm      : Decision Tree")
print("Max Depth      :", model.max_depth)
print("Min Samples Leaf:", model.min_samples_leaf)
print("Class Weight   :", model.class_weight)


# ==========================================
# MODEL PREDICTION & EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Predict on test data
y_pred = model.predict(X_test)

print("\nPrediction Completed!")

# Metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "-" * 60)
print("OVERALL PERFORMANCE")
print("-" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n" + "-" * 60)
print("CLASSIFICATION REPORT")
print("-" * 60)

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\n" + "-" * 60)
print("CONFUSION MATRIX")
print("-" * 60)

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ==========================================
# ERROR ANALYSIS & FEATURE IMPORTANCE
# ==========================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS & FEATURE IMPORTANCE")
print("=" * 60)

# -------------------------------
# ERROR ANALYSIS
# -------------------------------

errors = y_test != y_pred

print("\nTotal Test Samples:", len(y_test))
print("Correct Predictions:", (~errors).sum())
print("Incorrect Predictions:", errors.sum())

error_rate = errors.mean()

print(f"Error Rate: {error_rate:.4f}")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(
    feature_importance.head(15).to_string(index=False)
)

# -------------------------------
# TOP FEATURES
# -------------------------------

print("\n" + "-" * 60)
print("TOP 10 IMPORTANT FACTORS")
print("-" * 60)

for i, row in feature_importance.head(10).iterrows():
    print(
        f"{row['Feature']:<40} "
        f"{row['Importance']:.6f}"
    )


# ==========================================
# SAVE MODEL
# ==========================================

import joblib
import os

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

# Create Models folder
os.makedirs("Models", exist_ok=True)

# Save model
model_path = "Models/irrigation_decision_tree.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved Path:", model_path)

# Verify file
if os.path.exists(model_path):
    print("Model File Status: READY")
else:
    print("Model File Status: NOT FOUND")