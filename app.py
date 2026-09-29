import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Irrigation Prediction",
    page_icon="💧",
    layout="wide"
)

# ============================================================
# LOAD PIPELINE
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("Models/irrigation_pipeline.pkl")


pipeline = load_model()

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    df = df.copy()

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

    return df


# ============================================================
# SENSOR RANGE VALIDATION
# ============================================================

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


# ============================================================
# HEADER
# ============================================================

st.title("💧 Smart Irrigation Requirement Prediction")

st.write(
    "Data-driven irrigation decision-support system using "
    "soil, weather, crop and irrigation-related conditions."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("🌱 Field & Sensor Conditions")

col1, col2, col3 = st.columns(3)

with col1:

    soil_type = st.selectbox(
        "Soil Type",
        ["Sandy", "Clay", "Loamy", "Silt"]
    )

    soil_ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5
    )

    soil_moisture = st.number_input(
        "Soil Moisture",
        min_value=0.0,
        max_value=100.0,
        value=25.0
    )

    organic_carbon = st.number_input(
        "Organic Carbon",
        min_value=0.0,
        value=0.9
    )

    electrical_conductivity = st.number_input(
        "Electrical Conductivity",
        min_value=0.0,
        value=1.5
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=60.0,
        value=34.0
    )


with col2:

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=45.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=800.0
    )

    sunlight = st.number_input(
        "Sunlight Hours",
        min_value=0.0,
        max_value=24.0,
        value=9.0
    )

    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        value=14.0
    )

    crop_type = st.selectbox(
        "Crop Type",
        ["Sugarcane", "Rice", "Cotton", "Maize", "Wheat", "Potato"]
    )

    growth_stage = st.selectbox(
        "Crop Growth Stage",
        ["Sowing", "Vegetative", "Flowering", "Harvest"]
    )


with col3:

    season = st.selectbox(
        "Season",
        ["Kharif", "Rabi", "Zaid"]
    )

    irrigation_type = st.selectbox(
        "Irrigation Type",
        ["Canal", "Sprinkler", "Rainfed", "Drip"]
    )

    water_source = st.selectbox(
        "Water Source",
        ["Reservoir", "River", "Groundwater", "Rainwater"]
    )

    field_area = st.number_input(
        "Field Area (hectares)",
        min_value=0.0,
        value=5.0
    )

    mulching = st.selectbox(
        "Mulching Used",
        ["No", "Yes"]
    )

    previous_irrigation = st.number_input(
        "Previous Irrigation (mm)",
        min_value=0.0,
        value=30.0
    )

    region = st.selectbox(
        "Region",
        ["South", "West", "East", "Central", "North"]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🚀 Predict Irrigation Requirement",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    new_input = pd.DataFrame([{
        "Soil_Type": soil_type,
        "Soil_pH": soil_ph,
        "Soil_Moisture": soil_moisture,
        "Organic_Carbon": organic_carbon,
        "Electrical_Conductivity": electrical_conductivity,
        "Temperature_C": temperature,
        "Humidity": humidity,
        "Rainfall_mm": rainfall,
        "Sunlight_Hours": sunlight,
        "Wind_Speed_kmh": wind_speed,
        "Crop_Type": crop_type,
        "Crop_Growth_Stage": growth_stage,
        "Season": season,
        "Irrigation_Type": irrigation_type,
        "Water_Source": water_source,
        "Field_Area_hectare": field_area,
        "Mulching_Used": mulching,
        "Previous_Irrigation_mm": previous_irrigation,
        "Region": region
    }])

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    processed_input = engineer_features(new_input)

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = pipeline.predict(processed_input)[0]

    probabilities = pipeline.predict_proba(processed_input)[0]

    classes = pipeline.classes_

    probability_dict = dict(
        zip(classes, probabilities)
    )

    predicted_probability = probability_dict[prediction]

    # --------------------------------------------------------
    # WATER RECOMMENDATION
    # --------------------------------------------------------

    water_rate = {
        "Low": 10,
        "Medium": 20,
        "High": 30
    }

    rate_mm = water_rate[prediction]

    water_litres = (
        rate_mm
        * field_area
        * 10000
    )

    water_m3 = water_litres / 1000

    # --------------------------------------------------------
    # SENSOR VALIDATION
    # --------------------------------------------------------

    abnormal_sensors = []

    current_values = {
        "Soil_pH": soil_ph,
        "Soil_Moisture": soil_moisture,
        "Temperature_C": temperature,
        "Humidity": humidity,
        "Rainfall_mm": rainfall,
        "Sunlight_Hours": sunlight,
        "Wind_Speed_kmh": wind_speed,
        "Field_Area_hectare": field_area,
        "Previous_Irrigation_mm": previous_irrigation
    }

    for sensor, value in current_values.items():

        minimum, maximum = sensor_ranges[sensor]

        if value < minimum or value > maximum:
            abnormal_sensors.append(
                (sensor, value, minimum, maximum)
            )

    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("📊 Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Irrigation Requirement",
            prediction
        )

    with result_col2:

        st.metric(
            "Predicted Probability",
            f"{predicted_probability * 100:.2f}%"
        )

    with result_col3:

        st.metric(
            "Recommended Water",
            f"{water_litres:,.0f} L"
        )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.subheader("Prediction Probabilities")

    probability_df = pd.DataFrame({
        "Irrigation Level": classes,
        "Probability": [
            probability_dict[c] * 100
            for c in classes
        ]
    })

    st.bar_chart(
        probability_df.set_index("Irrigation Level")
    )

    # ========================================================
    # WATER RECOMMENDATION
    # ========================================================

    st.subheader("💧 Water Quantity Recommendation")

    water_col1, water_col2 = st.columns(2)

    with water_col1:

        st.write(
            f"**Recommended Water Depth:** {rate_mm} mm"
        )

        st.write(
            f"**Field Area:** {field_area:.2f} hectares"
        )

    with water_col2:

        st.write(
            f"**Water Quantity:** {water_litres:,.0f} litres"
        )

        st.write(
            f"**Water Volume:** {water_m3:,.2f} m³"
        )

    st.info(
        "Water quantity is a rule-based recommendation. "
        "The dataset does not contain a separate water-quantity target."
    )

    # ========================================================
    # SENSOR STATUS
    # ========================================================

    st.subheader("🔍 Sensor Validation")

    if len(abnormal_sensors) == 0:

        st.success(
            "✓ Sensor Status: NORMAL\n\n"
            "No abnormal sensor values detected."
        )

    else:

        st.warning(
            "⚠ Sensor Status: ABNORMAL"
        )

        for sensor, value, minimum, maximum in abnormal_sensors:

            st.write(
                f"**{sensor}:** {value} "
                f"(Expected range: {minimum} – {maximum})"
            )

    # ========================================================
    # CURRENT CONDITIONS
    # ========================================================

    st.subheader("🌱 Current Field Conditions")

    condition_col1, condition_col2 = st.columns(2)

    with condition_col1:

        st.write(f"**Soil Moisture:** {soil_moisture:.2f}")
        st.write(f"**Temperature:** {temperature:.2f} °C")
        st.write(f"**Rainfall:** {rainfall:.2f} mm")
        st.write(f"**Wind Speed:** {wind_speed:.2f} km/h")

    with condition_col2:

        st.write(f"**Crop:** {crop_type}")
        st.write(f"**Growth Stage:** {growth_stage}")
        st.write(f"**Mulching:** {mulching}")
        st.write(f"**Season:** {season}")

    # ========================================================
    # MODEL FACTORS
    # ========================================================

    st.subheader("🧠 Important Model Factors")

    st.write(
        "The following features had the highest global "
        "feature importance in the trained Decision Tree:"
    )

    important_features = {
        "Soil Moisture": 28.41,
        "Temperature": 16.66,
        "Rainfall": 12.25,
        "Wind Speed": 10.24,
        "Crop Growth Stage - Vegetative": 8.88,
        "Mulching - Yes": 5.26,
        "Crop Growth Stage - Flowering": 5.19,
        "Mulching - No": 3.69
    }

    importance_df = pd.DataFrame(
        list(important_features.items()),
        columns=["Feature", "Importance (%)"]
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.caption(
        "Feature importance indicates how much each feature "
        "contributed to the model's decision process; it does "
        "not imply causation."
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    st.divider()

    st.subheader("🎯 Final Decision")

    if prediction == "Low":
        st.success(
            "Low irrigation requirement predicted."
        )

    elif prediction == "Medium":
        st.warning(
            "Medium irrigation requirement predicted."
        )

    else:
        st.error(
            "High irrigation requirement predicted."
        )

    st.write(
        f"**Decision:** {prediction} irrigation requirement"
    )

    st.write(
        f"**Model probability:** "
        f"{predicted_probability * 100:.2f}%"
    )

    st.write(
        f"**Recommended water:** "
        f"{water_litres:,.0f} litres "
        f"({water_m3:,.2f} m³)"
    )