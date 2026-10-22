import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ========================================
# 1. PROJECT PATH
# ========================================

project_root = Path(__file__).resolve().parents[2]

processed_file = (
    project_root
    / "data"
    / "processed"
    / "hyderabad_aqi_weather_features.csv"
)


# ========================================
# 2. LOAD DATA
# ========================================

data = pd.read_csv(processed_file)

data["datetime"] = pd.to_datetime(data["datetime"])

data = data.sort_values("datetime")
data = data.reset_index(drop=True)


print("\n========================================")
print("AQI + WEATHER XGBOOST MODEL")
print("========================================")

print("\nDataset loaded:", len(data), "records")


# ========================================
# 3. SELECT FEATURES
# ========================================

features = [
    "hour",
    "day",
    "month",
    "year",
    "day_of_week",
    "is_weekend",

    "aqi_previous_hour",
    "aqi_previous_3_hours",
    "aqi_previous_6_hours",
    "aqi_previous_24_hours",

    "aqi_rolling_6h",
    "aqi_rolling_24h",

    "season",

    "temperature_current",
    "humidity_current",
    "wind_speed_current",
    "wind_direction_current",
    "pressure_current",

    "temperature_previous_hour",
    "humidity_previous_hour",
    "wind_speed_previous_hour",

    "temperature_change",
    "humidity_change",
    "wind_speed_change"
]

target = "target_aqi_next_hour"

X = data[features]
y = data[target]


# ========================================
# 4. TRAIN / TEST SPLIT
# ========================================

# IMPORTANT:
# Never randomly shuffle time-series data.

split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining period:")
print(data["datetime"].iloc[0])
print("to")
print(data["datetime"].iloc[split_index - 1])

print("\nTesting period:")
print(data["datetime"].iloc[split_index])
print("to")
print(data["datetime"].iloc[-1])


# ========================================
# 5. CATEGORICAL FEATURE
# ========================================

categorical_features = [
    "season"
]

numeric_features = [
    "hour",
    "day",
    "month",
    "year",
    "day_of_week",
    "is_weekend",

    "aqi_previous_hour",
    "aqi_previous_3_hours",
    "aqi_previous_6_hours",
    "aqi_previous_24_hours",

    "aqi_rolling_6h",
    "aqi_rolling_24h",

    "temperature_current",
    "humidity_current",
    "wind_speed_current",
    "wind_direction_current",
    "pressure_current",

    "temperature_previous_hour",
    "humidity_previous_hour",
    "wind_speed_previous_hour",

    "temperature_change",
    "humidity_change",
    "wind_speed_change"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "season",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ========================================
# 6. CREATE XGBOOST MODEL
# ========================================

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# ========================================
# 7. CREATE PIPELINE
# ========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ========================================
# 8. TRAIN MODEL
# ========================================

print("\nTraining XGBoost with weather features...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# ========================================
# 9. PREDICTIONS
# ========================================

predictions = pipeline.predict(X_test)


# ========================================
# 10. EVALUATION
# ========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n========================================")
print("XGBOOST + WEATHER PERFORMANCE")
print("========================================")

print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ========================================
# 11. ACTUAL VS PREDICTED
# ========================================

results = pd.DataFrame({
    "datetime": data["datetime"].iloc[split_index:].values,
    "actual_aqi": y_test.values,
    "predicted_aqi": predictions
})


print("\nFirst 10 predictions:")
print(results.head(10))


# ========================================
# 12. SAVE PREDICTIONS
# ========================================

output_file = (
    project_root
    / "data"
    / "processed"
    / "aqi_xgboost_weather_predictions.csv"
)

results.to_csv(
    output_file,
    index=False
)


print("\nPredictions saved to:")
print(output_file)


print("\n========================================")
print("XGBOOST + WEATHER MODEL COMPLETED!")
print("========================================")