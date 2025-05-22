import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
# Load the trained model
best_model = joblib.load("crop_recommendation_model.pkl")

# Load X_train
try:
    X_train = pd.read_csv("X_train.csv")
except FileNotFoundError:
    st.error("X_train.csv not found. Please ensure the file is available.")

# Function to generate future weather predictions
def generate_future_weather(X_train, model, days=300):
    last_known_features = np.array([X_train.iloc[-1].values])
    future_predictions = []
    
    for _ in range(days):
        last_known_features = last_known_features[:, :11]  # Ensure correct shape
        last_known_features_df = pd.DataFrame(last_known_features, columns=X_train.columns)
        last_known_features_df = last_known_features_df.astype(float)
        predicted_values = model.predict(last_known_features_df)
        future_predictions.append(predicted_values.flatten())
        last_known_features[0, -3:] = predicted_values.flatten()  # Update last 3 columns
        last_known_features = np.roll(last_known_features, -1)
    
    future_weather_df = pd.DataFrame(
        future_predictions, 
        columns=["Avg Temperature (°C)", "Humidity (%)", "Rainfall (mm)"]
    )
    future_weather_df["Date"] = pd.date_range(start="2025-04-04", periods=days, freq="D")
    
    return future_weather_df[["Date", "Avg Temperature (°C)", "Humidity (%)", "Rainfall (mm)"]]

# Load crop dataset
crop_df = pd.read_csv("Crop ds.csv")

# Extract unique soil types
soil_types = set()
for soil in crop_df["Soil Type"].dropna():
    soil_types.update(map(str.strip, soil.split(',')))
unique_soil_types = sorted(soil_types)

# Streamlit UI
st.title("🌱 Crop Recommendation System")

# User input for soil type
user_soil_type = st.selectbox("Select your soil type:", unique_soil_types)

if st.button("Recommend Crops"):
    # Step 1: Filter crops based on soil type
    filtered_crops = crop_df[crop_df['Soil Type'].str.contains(user_soil_type, case=False, na=False)]
    
    if filtered_crops.empty:
        st.error(f"No crops found for soil type: {user_soil_type}")
    else:
        # Step 2: Generate future weather predictions
        future_weather_df = generate_future_weather(X_train, best_model)

        # Step 3: Define weight factors for each criterion
        #while True:
        #    weights = random.choices([0.1 * i for i in range(1, 10)], k=4)
         #   if round(sum(weights), 1) == 1.0:
         #       break

        w_temp, w_humidity, w_rainfall, w_yield = 0.4,0.3,0.2,0.1

        # Step 4: Define maximum possible differences (adjust as needed)
        max_temp_diff = 30  # Assume max possible temp difference (30°C)
        max_humidity_diff = 50  # Assume max possible humidity difference (50%)
        max_rainfall_diff = 500  # Assume max possible rainfall difference (500 mm)

        # Step 5: Calculate crop suitability scores
        crop_scores = []
        
        for index, crop in filtered_crops.iterrows():
            crop_name = crop['Crop']
            growth_days = crop['Growth Duration (days)']

            # Get the weather data for the required number of growth days
            relevant_weather = future_weather_df.iloc[:growth_days]  

            # Calculate weather condition differences
            temp_diff = abs(relevant_weather["Avg Temperature (°C)"].mean() - crop["Avg Temperature (°C)"])
            humidity_diff = abs(relevant_weather["Humidity (%)"].mean() - crop["Avg Humidity (%)"])
            rainfall_diff = abs(relevant_weather["Rainfall (mm)"].mean() - crop["Avg Rainfall (mm)"])

            # Normalize differences (convert to 0-100 suitability scores)
            temp_score = max(0, (1 - temp_diff / max_temp_diff) * 100)
            humidity_score = max(0, (1 - humidity_diff / max_humidity_diff) * 100)
            rainfall_score = max(0, (1 - rainfall_diff / max_rainfall_diff) * 100)

            # Normalize yield to a scale of 0-100
            # Normalize yield to a scale of 0-100
            max_yield = filtered_crops["Yield (kg/ha)"].max()
            min_yield = filtered_crops["Yield (kg/ha)"].min()

            if max_yield == min_yield:  # Avoid division by zero
                yield_score = 50  # Assign a default mid-range score if all crops have the same yield
            else:
                yield_score = (crop["Yield (kg/ha)"] - min_yield) / (max_yield - min_yield) * 100


            # Suitability score calculation
            score = (
                w_temp * temp_score +
                w_humidity * humidity_score +
                w_rainfall * rainfall_score +
                w_yield * yield_score
            )

            # Store the crop and its calculated score
            crop_scores.append((crop_name, score))

        # Convert scores to DataFrame
        crop_scores_df = pd.DataFrame(crop_scores, columns=["Crop", "Suitability Score"])

        # Sort by highest score
        crop_scores_df = crop_scores_df.sort_values(by="Suitability Score", ascending=False)

        # Step 6: Display top 3 recommended crops
        top_3_crops = crop_scores_df.head(3).copy()
        top_3_crops["Suitability Score"] = top_3_crops["Suitability Score"].round(2).astype(str) + "%"

        # Display results
        st.subheader("🌾 Top 3 Recommended Crops 🌾")
        st.table(top_3_crops)
