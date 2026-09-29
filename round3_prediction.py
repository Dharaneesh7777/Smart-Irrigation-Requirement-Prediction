import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# ROUND 3 - SMART IRRIGATION PREDICTION
# ==========================================

print("=" * 60)
print("ROUND 3 - SMART IRRIGATION PREDICTION")
print("=" * 60)

# Load trained Decision Tree
model_path = "Models/irrigation_decision_tree.pkl"

model = joblib.load(model_path)

print("\nModel loaded successfully!")
print("Model Type:", type(model).__name__)



# ==========================================
# ROUND 3 - LOAD TRAINING DATA FOR ENCODER
# ==========================================

print("\n" + "=" * 60)
print("LOADING PREPROCESSING INFORMATION")
print("=" * 60)

# Load original dataset
df = pd.read_csv("Dataset/train.csv")

# Recreate the same engineered features
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

# Same feature selection used in Round 2
features = df.drop(
    columns=["Irrigation_Need", "id"]
)

categorical_features = features.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_features = features.select_dtypes(
    include=["number"]
).columns.tolist()

# Recreate encoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoder.fit(
    features[categorical_features]
)

print("\nPreprocessing information loaded!")
print("Numerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))

# ==========================================
# ROUND 3 - NEW UNSEEN INPUT
# ==========================================

print("\n" + "=" * 60)
print("NEW UNSEEN INPUT")
print("=" * 60)

new_input = pd.DataFrame([{
    "Soil_Type": "Loamy",
    "Soil_pH": 6.5,
    "Soil_Moisture": 25.0,
    "Organic_Carbon": 0.9,
    "Electrical_Conductivity": 1.5,
    "Temperature_C": 34.0,
    "Humidity": 45.0,
    "Rainfall_mm": 800.0,
    "Sunlight_Hours": 9.0,
    "Wind_Speed_kmh": 14.0,
    "Crop_Type": "Rice",
    "Crop_Growth_Stage": "Vegetative",
    "Season": "Kharif",
    "Irrigation_Type": "Canal",
    "Water_Source": "Reservoir",
    "Field_Area_hectare": 5.0,
    "Mulching_Used": "No",
    "Previous_Irrigation_mm": 30.0,
    "Region": "South"
}])

print("\nNew Input:")
print(new_input.T)

# ==========================================
# ROUND 3 - FEATURE ENGINEERING FOR NEW INPUT
# ==========================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING FOR NEW INPUT")
print("=" * 60)

# Create engineered features
new_input["Moisture_Deficit"] = (
    100 - new_input["Soil_Moisture"]
)

new_input["Temperature_Humidity_Index"] = (
    new_input["Temperature_C"]
    * (100 - new_input["Humidity"])
)

new_input["Heat_Sunlight_Stress"] = (
    new_input["Temperature_C"]
    * new_input["Sunlight_Hours"]
)

new_input["Water_Availability"] = (
    new_input["Rainfall_mm"]
    + new_input["Previous_Irrigation_mm"]
)

new_input["Evaporation_Risk"] = (
    new_input["Temperature_C"]
    + new_input["Wind_Speed_kmh"]
    + new_input["Sunlight_Hours"]
    - new_input["Humidity"] / 10
)

new_input["Moisture_Rainfall_Stress"] = (
    (100 - new_input["Soil_Moisture"])
    / (new_input["Rainfall_mm"] + 1)
)

new_input["Previous_Irrigation_Ratio"] = (
    new_input["Previous_Irrigation_mm"]
    / (new_input["Rainfall_mm"] + 1)
)

print("\nEngineered Features:")

print(
    new_input[
        [
            "Moisture_Deficit",
            "Temperature_Humidity_Index",
            "Heat_Sunlight_Stress",
            "Water_Availability",
            "Evaporation_Risk",
            "Moisture_Rainfall_Stress",
            "Previous_Irrigation_Ratio"
        ]
    ].T
)

# ==========================================
# ENCODE NEW INPUT
# ==========================================

print("\n" + "=" * 60)
print("ENCODING NEW INPUT")
print("=" * 60)

# Numerical features
new_numeric = new_input[numerical_features]

# Categorical features
new_categorical_encoded = encoder.transform(
    new_input[categorical_features]
)

new_encoded_df = pd.DataFrame(
    new_categorical_encoded,
    columns=encoder.get_feature_names_out(
        categorical_features
    )
)

# Combine numerical + encoded categorical
new_X = pd.concat(
    [
        new_numeric.reset_index(drop=True),
        new_encoded_df.reset_index(drop=True)
    ],
    axis=1
)

# Ensure EXACT same feature order as training
# Combine numerical + encoded categorical
new_X = pd.concat(
    [
        new_numeric.reset_index(drop=True),
        new_encoded_df.reset_index(drop=True)
    ],
    axis=1
)

# Ensure EXACT same feature order as training
new_X = new_X[model.feature_names_in_]

print("\nNew Feature Matrix Shape:")
print(new_X.shape)

print("\nExpected Feature Shape:")
print((1, len(model.feature_names_in_)))

# ==========================================
# PREDICTION
# ==========================================

print("\n" + "=" * 60)
print("IRRIGATION NEED PREDICTION")
print("=" * 60)

prediction = model.predict(new_X)[0]

print("\nPredicted Irrigation Need:", prediction)


# ==========================================
# PREDICTION CONFIDENCE
# ==========================================

print("\n" + "=" * 60)
print("PREDICTION CONFIDENCE")
print("=" * 60)

# Get class probabilities
probabilities = model.predict_proba(new_X)[0]

classes = model.classes_

print("\nClass Probabilities:")

for class_name, probability in zip(classes, probabilities):
    print(f"{class_name:<10}: {probability * 100:.2f}%")

# Highest probability
confidence = probabilities.max() * 100

print("\n" + "-" * 60)
print(f"Predicted Class : {prediction}")
print(f"Confidence      : {confidence:.2f}%")
print("-" * 60)


# ==========================================
# PREDICTION EXPLANATION
# ==========================================

print("\n" + "=" * 60)
print("PREDICTION EXPLANATION")
print("=" * 60)

print("\nKey Factors Considered by the Model:")

# Get feature importance
feature_importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Display top important factors
top_factors = feature_importance[
    feature_importance["Importance"] > 0
].head(8)

for _, row in top_factors.iterrows():
    feature = row["Feature"]
    importance = row["Importance"]

    print(
        f"- {feature}: "
        f"{importance * 100:.2f}% importance"
    )

print("\nCurrent Important Sensor Conditions:")

print(
    f"- Soil Moisture     : "
    f"{new_input['Soil_Moisture'].iloc[0]:.2f}"
)

print(
    f"- Temperature       : "
    f"{new_input['Temperature_C'].iloc[0]:.2f} °C"
)

print(
    f"- Rainfall          : "
    f"{new_input['Rainfall_mm'].iloc[0]:.2f} mm"
)

print(
    f"- Wind Speed        : "
    f"{new_input['Wind_Speed_kmh'].iloc[0]:.2f} km/h"
)

print(
    f"- Crop Growth Stage : "
    f"{new_input['Crop_Growth_Stage'].iloc[0]}"
)

print(
    f"- Mulching Used     : "
    f"{new_input['Mulching_Used'].iloc[0]}"
)

# ==========================================
# ABNORMAL SENSOR DETECTION
# ==========================================

print("\n" + "=" * 60)
print("ABNORMAL SENSOR DETECTION")
print("=" * 60)

# Training-data observed ranges
sensor_ranges = {
    "Soil_pH": (4.80, 8.20),
    "Soil_Moisture": (8.00, 64.99),
    "Temperature_C": (12.00, 42.00),
    "Humidity": (25.00, 94.99),
    "Rainfall_mm": (0.38, 2499.69),
    "Sunlight_Hours": (4.00, 11.00),
    "Wind_Speed_kmh": (0.50, 20.00),
    "Field_Area_hectare": (0.30, 15.00),
    "Previous_Irrigation_mm": (0.02, 119.99)
}

abnormal_sensors = []

print("\nSensor Validation:")

for sensor, (minimum, maximum) in sensor_ranges.items():

    value = new_input[sensor].iloc[0]

    if value < minimum or value > maximum:

        abnormal_sensors.append(sensor)

        print(
            f"⚠ {sensor}: {value:.2f} "
            f"(Expected {minimum:.2f} - {maximum:.2f})"
        )

    else:

        print(
            f"✓ {sensor}: {value:.2f} "
            f"(Normal)"
        )

print("\n" + "-" * 60)

if len(abnormal_sensors) == 0:

    print("Sensor Status: NORMAL")
    print("No abnormal sensor values detected.")

else:

    print("Sensor Status: ABNORMAL")
    print("Abnormal Sensors:")

    for sensor in abnormal_sensors:
        print("-", sensor)


# ==========================================
# WATER QUANTITY RECOMMENDATION
# ==========================================

print("\n" + "=" * 60)
print("WATER QUANTITY RECOMMENDATION")
print("=" * 60)

field_area = new_input["Field_Area_hectare"].iloc[0]

# Recommendation basis:
# This is a decision-support rule, NOT an ML prediction.
water_rate = {
    "Low": 10,
    "Medium": 20,
    "High": 30
}

rate_mm = water_rate[prediction]

# 1 hectare = 10,000 m²
# 1 mm water over 1 hectare = 10,000 litres
water_litres = rate_mm * field_area * 10000

water_m3 = water_litres / 1000

print("\nPredicted Irrigation Need :", prediction)
print("Field Area                :", field_area, "hectares")
print("Recommended Water Depth   :", rate_mm, "mm")
print("Recommended Water Quantity:", f"{water_litres:,.0f}", "litres")
print("Recommended Water Volume  :", f"{water_m3:,.2f}", "m³")

print("\nNote:")
print("Water quantity is a rule-based recommendation.")
print("It is not a separately trained ML prediction.")