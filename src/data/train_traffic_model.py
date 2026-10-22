import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# ============================================
# LOAD DATA
# ============================================

DATA_PATH = (
    "data/raw/idd_traffic_features.csv"
)

df = pd.read_csv(DATA_PATH)


print("\n====================================")
print("TRAFFIC DENSITY MODEL")
print("====================================")

print(
    f"Total samples: {len(df)}"
)


# ============================================
# FEATURES
# ============================================

features = [
    "traffic_pixel_ratio",
    "living_pixel_ratio",
    "drivable_pixel_ratio",
    "construction_pixel_ratio",
    "sky_pixel_ratio"
]


target = "traffic_density_score"


X = df[features]

y = df[target]


# ============================================
# TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================
# MODEL
# ============================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)


print(
    "\nTraining traffic model..."
)


model.fit(
    X_train,
    y_train
)


# ============================================
# PREDICTION
# ============================================

predictions = model.predict(
    X_test
)


# ============================================
# EVALUATION
# ============================================

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


print(
    "\n===================================="
)

print(
    "MODEL RESULTS"
)

print(
    "===================================="
)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================
# SAVE MODEL
# ============================================

MODEL_PATH = (
    "data/raw/traffic_density_model.pkl"
)


joblib.dump(
    model,
    MODEL_PATH
)


print(
    "\nModel saved to:"
)

print(
    MODEL_PATH
)


print(
    "\nTraffic model training complete!"
)