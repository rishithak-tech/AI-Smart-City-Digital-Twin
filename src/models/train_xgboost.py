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
    / "hyderabad_air_quality_features.csv"
)


# ========================================
# 2. LOAD DATA
# ========================================

data = pd.read_csv(processed_file)

print("\n========================================")
print("AQI XGBOOST MODEL")
print("========================================")

print(f"\nDataset loaded: {len(data)} records")


# ========================================
# 3. PREPARE DATA
# ========================================

data["datetime"] = pd.to_datetime(data["datetime"])

data = data.sort_values("datetime")
data = data.reset_index(drop=True)


# ========================================
# 4. SELECT FEATURES
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
    "season"
]

target = "target_aqi_next_hour"

X = data[features]
y = data[target]


# ========================================
# 5. TRAIN / TEST SPLIT
# ========================================

# IMPORTANT:
# Do NOT shuffle time-series data.

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
# 6. HANDLE CATEGORICAL FEATURE
# ========================================

categorical_features = ["season"]

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
    "aqi_rolling_24h"
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
# 7. CREATE XGBOOST MODEL
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
# 8. CREATE PIPELINE
# ========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ========================================
# 9. TRAIN MODEL
# ========================================

print("\nTraining XGBoost model...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# ========================================
# 10. MAKE PREDICTIONS
# ========================================

predictions = pipeline.predict(X_test)


# ========================================
# 11. EVALUATE MODEL
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
print("XGBOOST MODEL PERFORMANCE")
print("========================================")

print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ========================================
# 12. SHOW ACTUAL VS PREDICTED
# ========================================

results = pd.DataFrame({
    "datetime": data["datetime"].iloc[split_index:].values,
    "actual_aqi": y_test.values,
    "predicted_aqi": predictions
})


print("\nFirst 10 predictions:")
print(results.head(10))


# ========================================
# 13. SAVE PREDICTIONS
# ========================================

output_file = (
    project_root
    / "data"
    / "processed"
    / "aqi_xgboost_predictions.csv"
)

results.to_csv(
    output_file,
    index=False
)


print("\nPredictions saved to:")
print(output_file)


# ========================================
# 14. FINAL MESSAGE
# ========================================

print("\n========================================")
print("XGBOOST MODEL COMPLETED!")
print("========================================")