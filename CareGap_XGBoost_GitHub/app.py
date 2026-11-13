
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="CareGap XGBoost Model Test",
    layout="centered"
)


st.markdown("""
<style>
[data-testid="stNumberInput"] button {
    display: none !important;
}
[data-testid="stNumberInput"] input {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

st.title("CareGap Planner — XGBoost Live Test")
st.write("Predict mammography or colorectal screening prevalence using five inputs.")

screening_type = st.radio(
    "Choose screening type",
    ["Mammography", "Colorectal"],
    horizontal=True
)

model_path = str(BASE_DIR / "models/Mammography_Model.pkl")

with open(model_path, "rb") as f:
    mammography_model = pickle.load(f)

colorectal_model_path = str(BASE_DIR / "models/Colorectal_Model.pkl")
with open(colorectal_model_path, "rb") as f:
    colorectal_model = pickle.load(f)

model = (
    mammography_model
    if screening_type == "Mammography"
    else colorectal_model
)

poverty = st.number_input(
    "Population below 150% poverty threshold (%)",
    value=31.4, min_value=0.0, max_value=100.0
)
uninsured = st.number_input(
    "Uninsured population (%)",
    value=32.1, min_value=0.0, max_value=100.0
)
english = st.number_input(
    "Limited English proficiency (%)",
    value=0.0, min_value=0.0, max_value=100.0
)
vehicle = st.number_input(
    "Households without a vehicle (%)",
    value=6.0, min_value=0.0, max_value=100.0
)
distance = st.number_input(
    "Distance to nearest geocoded mammography facility (km)",
    value=21.44337527566393, min_value=0.0
)

if st.button("Predict Screening Rate", type="primary"):
    inputs = pd.DataFrame([{
        "EP_POV150": poverty,
        "EP_UNINSUR": uninsured,
        "EP_LIMENG": english,
        "EP_NOVEH": vehicle,
        "Nearest_Mammo_Distance_km": distance
    }])

    prediction = float(model.predict(inputs)[0])

    st.metric(f"Predicted {screening_type} Screening", f"{prediction:.2f}%")
    st.caption(
        "Model prediction of CDC PLACES screening prevalence. "
        "Not an individual medical prediction."
    )

    if screening_type == "Mammography":
        st.info(
            "Reference test tract 13019970100: "
            "actual modeled estimate 73.60%. "
            "This comparison applies only to the default inputs."
        )



st.divider()
st.subheader(f"{screening_type} Model Testing Results")

# Use saved test results for the selected model
if screening_type == "Mammography":
    results_path = str(BASE_DIR / "results/Mammography_Test_Results.csv")
else:
    results_path = str(BASE_DIR / "results/Colorectal_Test_Results.csv")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

if Path(results_path).exists():
    test_results = pd.read_csv(results_path)

    actual = test_results["Actual"]
    predicted = test_results["Predicted"]

    st.write(
        f"Evaluation on {len(test_results)} held-out census tracts "
        "from 24 Georgia counties."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mean_absolute_error(actual, predicted):.4f}")
    c2.metric("RMSE", f"{np.sqrt(mean_squared_error(actual, predicted)):.4f}")
    c3.metric("R²", f"{r2_score(actual, predicted):.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(actual, predicted, alpha=0.55)

    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())

    ax.plot(
        [low, high], [low, high],
        "r--", label="Perfect prediction"
    )

    ax.set_xlabel("Actual Screening (%)")
    ax.set_ylabel("Predicted Screening (%)")
    ax.set_title(f"{screening_type}: Actual vs Predicted")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("Saved testing results not found.")

st.caption(
    "These results are based on the reconstructed "
    "county-held-out test split."
)

st.divider()
st.subheader("Test Model Using a CSV Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV containing the five model input features",
    type=["csv"],
    key=f"upload_{screening_type}"
)

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)

    feature_cols = list(model.feature_names_in_)
    missing = [c for c in feature_cols if c not in uploaded_df.columns]

    if missing:
        st.error(f"Missing required features: {missing}")
    else:
        inputs = uploaded_df[feature_cols].copy()
        inputs = inputs.apply(pd.to_numeric, errors="coerce")
        inputs = inputs.replace(-999, np.nan)

        if inputs.isna().any().any():
            st.error("CSV contains missing or invalid feature values.")
        else:
            uploaded_df["Predicted_Screening"] = model.predict(inputs)

            st.success(
                f"Predictions generated for {len(uploaded_df)} records!"
            )

            # Accept only the matching target for selected screening
            if screening_type == "Mammography":
                target_col = "Actual_Mammography_Screening"
            else:
                target_col = "Actual_Colorectal_Screening"

            if target_col in uploaded_df.columns:
                actual = pd.to_numeric(
                    uploaded_df[target_col],
                    errors="coerce"
                )
                predicted = uploaded_df["Predicted_Screening"]

                if actual.notna().all() and len(actual) >= 2:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("MAE", f"{mean_absolute_error(actual, predicted):.4f}")
                    c2.metric("RMSE", f"{np.sqrt(mean_squared_error(actual, predicted)):.4f}")
                    c3.metric("R²", f"{r2_score(actual, predicted):.4f}")

                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.scatter(actual, predicted, alpha=0.55)
                    low = min(actual.min(), predicted.min())
                    high = max(actual.max(), predicted.max())
                    ax.plot([low, high], [low, high], "r--")
                    ax.set_xlabel("Actual Screening (%)")
                    ax.set_ylabel("Predicted Screening (%)")
                    ax.set_title(f"{screening_type}: Uploaded Dataset")
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info(
                    f"Upload '{target_col}' to calculate evaluation metrics. "
                    "Predictions are available without actual values."
                )

            st.dataframe(uploaded_df.head(20), use_container_width=True)

            st.download_button(
                "Download Prediction Results",
                uploaded_df.to_csv(index=False),
                file_name=f"CareGap_{screening_type}_Predictions.csv",
                mime="text/csv"
            )


# =========================================================
# SPATIAL ANALYSIS — GEORGIA SCREENING ACCESS
# =========================================================

st.divider()
st.header("Spatial Analysis — Georgia Screening Access")

st.write(
    "Explore the geographic distribution of mammography "
    "facilities and distance-based screening access "
    "across Georgia."
)

spatial_folder = BASE_DIR / "caregap_dashboard"

spatial_df = pd.read_csv(
    spatial_folder / "caregap_data.csv"
)

facilities_df = pd.read_csv(
    spatial_folder /
    "georgia_mammography_facilities_with_coordinates.csv"
)

# Facility coordinates
facilities_df["Latitude"] = pd.to_numeric(
    facilities_df["Latitude"], errors="coerce"
)
facilities_df["Longitude"] = pd.to_numeric(
    facilities_df["Longitude"], errors="coerce"
)

facilities_map = facilities_df.dropna(
    subset=["Latitude", "Longitude"]
).copy()

st.subheader("Mammography Facility Locations")

st.write(
    f"Displaying {len(facilities_map)} geocoded "
    "mammography facilities in Georgia."
)

st.map(
    facilities_map,
    latitude="Latitude",
    longitude="Longitude",
    zoom=6
)

st.caption(
    "Each point represents a geocoded mammography "
    "facility. This is not a map of colorectal facilities."
)

# Distance distribution
st.subheader("Distance to Nearest Mammography Facility")

distance_col = "Nearest_Mammo_Distance_km"

distances = pd.to_numeric(
    spatial_df[distance_col],
    errors="coerce"
).dropna()

fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(distances, bins=30, edgecolor="white")
ax.set_xlabel("Distance to Nearest Facility (km)")
ax.set_ylabel("Number of Census Tracts")
ax.set_title("Geographic Distribution of Facility Distance")
ax.grid(axis="y", alpha=0.2)

st.pyplot(fig)
plt.close(fig)

c1, c2, c3 = st.columns(3)

c1.metric("Census Tracts", f"{len(spatial_df):,}")
c2.metric("More Than 10 km", f"{(distances > 10).sum():,}")
c3.metric("More Than 20 km", f"{(distances > 20).sum():,}")

st.caption(
    "Distances are approximate straight-line geographic "
    "distances, not road travel distances or travel times."
)
