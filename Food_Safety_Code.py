import pickle
import pandas as pd
import streamlit as st

import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Food Product Approval Predictor", page_icon="🍱", layout="centered")

# =====================================================================
# LIVE MOVING GRADIENT BACKGROUND
# =====================================================================
st.markdown("""
    <style>
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: linear-gradient(-45deg, #eef2f3, #8e9eab, #eef2f3, #e0eafc) !important;
        background-size: 400% 400% !important;
        animation: gradientBG 6s ease infinite !important;
    }
    
    [data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)


# =====================================================================
# MODEL INITIALIZATION 
# =====================================================================

@st.cache_resource
def load_model():
    # Looks for the model file in the same GitHub repository folder
    with open("best_food_safety_model.pkl", "rb") as f:
        return pickle.load(f)

# Load the trained model bundle
bundle = load_model()
model = bundle["model"]
encoders = bundle["encoders"]
target_encoder = bundle["target_encoder"]
feature_order = bundle["features"]

st.title("🍱 Food Product Approval Predictor")
st.caption("Enter product details to predict whether it will be Approved or Rejected.")

col1, col2 = st.columns(2)
with col1:
    product_type = st.selectbox("Product type", encoders["product_type"].classes_)
    packaging_type = st.selectbox("Packaging type", encoders["packaging_type"].classes_)
    shelf_life_days = st.number_input("Shelf life (days)", min_value=1, max_value=400, value=150)
    storage_temperature_C = st.number_input("Storage temperature (°C)", min_value=-20.0, max_value=40.0, value=5.0, step=0.1)
    humidity_level_percent = st.number_input("Humidity level (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

with col2:
    moisture_content = st.number_input("Moisture content", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
    preservative_level = st.number_input("Preservative level", min_value=0.0, max_value=10.0, value=3.0, step=0.01)
    pH_level = st.number_input("pH level", min_value=0.0, max_value=14.0, value=5.5, step=0.01)
    inspection_score_0_100 = st.number_input("Inspection score (0-100)", min_value=0, max_value=100, value=85)
    customer_rating_1_5 = st.slider("Customer rating (1-5)", 1, 5, 4)

if st.button("Predict approval", type="primary"):
    # Apply encoders properly for multi-class structure matching train dataset
    row = {
        "product_type": encoders["product_type"].transform([product_type])[0],
        "packaging_type": encoders["packaging_type"].transform([packaging_type])[0],
        "shelf_life_days": shelf_life_days,
        "storage_temperature_C": storage_temperature_C,
        "humidity_level_percent": humidity_level_percent,
        "moisture_content": moisture_content,
        "preservative_level": preservative_level,
        "pH_level": pH_level,
        "inspection_score_0_100": inspection_score_0_100,
        "customer_rating_1_5": customer_rating_1_5,
    }
    
    X = pd.DataFrame([row])[feature_order]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = target_encoder.inverse_transform([pred])[0]

    # Show prediction output nicely
    if label == "Approved":
        st.success(f"Prediction: **{label}** ✅")
    else:
        st.error(f"Prediction: **{label}** ❌")

    # Smart Safety Checks Report
    st.subheader("📋 Crucial Metrics Analysis Report")
    if storage_temperature_C > 10.0:
        st.warning("⚠️ **Temperature Alert:** High storage temperature increases bacterial spoilage risk.")
    else:
        st.info("✅ **Temperature Check:** Storage temperature is safe.")

    if inspection_score_0_100 < 70:
        st.warning(f"⚠️ **Inspection Alert:** Facility inspection score ({inspection_score_0_100}) does not meet safety standards.")
    else:
        st.info("✅ **Inspection Check:** Production facility safety compliance has passed.")

    if pH_level < 4.0 or pH_level > 8.0:
        st.warning(f"⚠️ **Chemical Alert:** Abnormal pH level ({pH_level}) indicates chemical instability.")
    else:
        st.info("✅ **pH Level Check:** Acidity levels are within limits.")

    # Probability Chart
    st.subheader("📊 Prediction Probability Profile")
    proba_df = pd.DataFrame(
        {"Status": target_encoder.classes_, "Probability": proba}
    ).sort_values("Probability", ascending=False)
    st.bar_chart(proba_df.set_index("Status"))

st.divider()
st.caption("Model: Machine Learning Classifier trained on food safety historical records.")



