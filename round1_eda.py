import pandas as pd

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("Dataset/train.csv")


# ==========================================
# TASK 1-4 - BASIC DATASET INSPECTION
# ==========================================

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)
print(df["Irrigation_Need"].value_counts())

print("\n" + "=" * 60)
print("TARGET PERCENTAGE")
print("=" * 60)
print(df["Irrigation_Need"].value_counts(normalize=True) * 100)


# ==========================================
# TASK 5 - OUTLIER ANALYSIS USING IQR
# ==========================================

numerical_columns = df.select_dtypes(include="number").columns.tolist()

# ID is an identifier, not a meaningful feature
if "id" in numerical_columns:
    numerical_columns.remove("id")

print("\n" + "=" * 60)
print("OUTLIER ANALYSIS - IQR METHOD")
print("=" * 60)

outlier_summary = []

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ).sum()

    outlier_percentage = (outliers / len(df)) * 100

    outlier_summary.append({
        "Feature": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": outliers,
        "Outlier_Percentage": outlier_percentage
    })

outlier_df = pd.DataFrame(outlier_summary)

print(outlier_df.to_string(index=False))





# ==========================================
# TASK 6 - CATEGORICAL FEATURE ANALYSIS
# ==========================================

categorical_columns = df.select_dtypes(include=["str", "object"]).columns.tolist()

# Remove target from feature columns
if "Irrigation_Need" in categorical_columns:
    categorical_columns.remove("Irrigation_Need")

print("\n" + "=" * 60)
print("CATEGORICAL FEATURES")
print("=" * 60)

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts().to_string())

print("\n" + "=" * 60)
print("CATEGORICAL FEATURE COUNT")
print("=" * 60)
print(len(categorical_columns))


# ==========================================
# TASK 7 - DESCRIPTIVE STATISTICS
# ==========================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS - NUMERICAL FEATURES")
print("=" * 60)

print(df[numerical_columns].describe().T.to_string())


# ==========================================
# TASK 8 & 9 - EDA AND VISUALIZATIONS
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
import os
os.makedirs("outputs", exist_ok=True)


# ------------------------------------------
# 1. TARGET DISTRIBUTION
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Irrigation_Need",
    order=["Low", "Medium", "High"]
)

plt.title("Irrigation Requirement Distribution")
plt.xlabel("Irrigation Requirement")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.savefig("outputs/01_target_distribution.png", dpi=150)
plt.show()


# ------------------------------------------
# 2. SOIL MOISTURE vs IRRIGATION NEED
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Irrigation_Need",
    y="Soil_Moisture",
    order=["Low", "Medium", "High"]
)

plt.title("Soil Moisture vs Irrigation Requirement")
plt.xlabel("Irrigation Requirement")
plt.ylabel("Soil Moisture")

plt.tight_layout()
plt.savefig("outputs/02_soil_moisture_vs_irrigation.png", dpi=150)
plt.show()


# ------------------------------------------
# 3. TEMPERATURE vs IRRIGATION NEED
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Irrigation_Need",
    y="Temperature_C",
    order=["Low", "Medium", "High"]
)

plt.title("Temperature vs Irrigation Requirement")
plt.xlabel("Irrigation Requirement")
plt.ylabel("Temperature (°C)")

plt.tight_layout()
plt.savefig("outputs/03_temperature_vs_irrigation.png", dpi=150)
plt.show()


# ------------------------------------------
# 4. RAINFALL vs IRRIGATION NEED
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Irrigation_Need",
    y="Rainfall_mm",
    order=["Low", "Medium", "High"]
)

plt.title("Rainfall vs Irrigation Requirement")
plt.xlabel("Irrigation Requirement")
plt.ylabel("Rainfall (mm)")

plt.tight_layout()
plt.savefig("outputs/04_rainfall_vs_irrigation.png", dpi=150)
plt.show()


# ------------------------------------------
# 5. CORRELATION HEATMAP
# ------------------------------------------

plt.figure(figsize=(12, 8))

correlation = df[numerical_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap - Numerical Features")

plt.tight_layout()
plt.savefig("outputs/05_correlation_heatmap.png", dpi=150)
plt.show()


# ==========================================
# TASK 10 - FEATURE RELEVANCE
# ==========================================

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

print("\n" + "=" * 60)
print("TASK 10 - FEATURE RELEVANCE")
print("=" * 60)

# Encode target temporarily
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(df["Irrigation_Need"])

# ------------------------------------------
# Numerical Feature Relevance
# ------------------------------------------

X_numeric = df[numerical_columns]

f_scores, p_values = f_classif(
    X_numeric,
    y_encoded
)

numeric_relevance = pd.DataFrame({
    "Feature": numerical_columns,
    "F_Score": f_scores,
    "P_Value": p_values
})

numeric_relevance = numeric_relevance.sort_values(
    by="F_Score",
    ascending=False
)

print("\nNUMERICAL FEATURE RELEVANCE")
print("-" * 60)
print(numeric_relevance.to_string(index=False))

# ------------------------------------------
# Categorical Feature Relevance
# ------------------------------------------

X_categorical = df[categorical_columns].copy()

for column in categorical_columns:
    X_categorical[column] = LabelEncoder().fit_transform(
        X_categorical[column]
    )

mi_scores = mutual_info_classif(
    X_categorical,
    y_encoded,
    random_state=42
)

categorical_relevance = pd.DataFrame({
    "Feature": categorical_columns,
    "Mutual_Information": mi_scores
})

categorical_relevance = categorical_relevance.sort_values(
    by="Mutual_Information",
    ascending=False
)

print("\nCATEGORICAL FEATURE RELEVANCE")
print("-" * 60)
print(categorical_relevance.to_string(index=False))