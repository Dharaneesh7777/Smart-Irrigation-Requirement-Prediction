import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# ============================================================
# FINAL SMART IRRIGATION PIPELINE
# ============================================================

print("=" * 60)
print("CREATING FINAL IRRIGATION ML PIPELINE")
print("=" * 60)

# Load dataset
df = pd.read_csv("Dataset/train.csv")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["Moisture_Deficit"] = 100 - df["Soil_Moisture"]

df["Temperature_Humidity_Index"] = (
    df["Temperature_C"] * (100 - df["Humidity"])
)

df["Heat_Sunlight_Stress"] = (
    df["Temperature_C"] * df["Sunlight_Hours"]
)

df["Water_Availability"] = (
    df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
)

df["Evaporation_Risk"] = (
    df["Temperature_C"]
    + df["Wind_Speed_kmh"]
    + df["Sunlight_Hours"]
    - df["Humidity"] / 10
)

df["Moisture_Rainfall_Stress"] = (
    (100 - df["Soil_Moisture"])
    / (df["Rainfall_mm"] + 1)
)

df["Previous_Irrigation_Ratio"] = (
    df["Previous_Irrigation_mm"]
    / (df["Rainfall_mm"] + 1)
)

# ============================================================
# INPUT / TARGET
# ============================================================

X = df.drop(columns=["Irrigation_Need", "id"])
y = df["Irrigation_Need"]

# ============================================================
# FEATURE TYPES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))

# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# ============================================================
# DECISION TREE
# ============================================================

model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
    max_depth=12,
    min_samples_leaf=10
)

# ============================================================
# COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training completed!")

# ============================================================
# SAVE PIPELINE
# ============================================================

joblib.dump(
    pipeline,
    "Models/irrigation_pipeline.pkl"
)

print("\n" + "=" * 60)
print("PIPELINE SAVED SUCCESSFULLY")
print("=" * 60)

print("\nFile:")
print("Models/irrigation_pipeline.pkl")

print("\nPipeline contains:")
print("✓ Feature preprocessing")
print("✓ One-hot encoding")
print("✓ Decision Tree model")
print("✓ Unknown-category handling")

print("\nReady for Streamlit deployment!")