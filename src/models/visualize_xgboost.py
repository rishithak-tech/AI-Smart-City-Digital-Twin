import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ========================================
# 1. PROJECT PATH
# ========================================

project_root = Path(__file__).resolve().parents[2]

results_file = (
    project_root
    / "data"
    / "processed"
    / "aqi_xgboost_predictions.csv"
)


# ========================================
# 2. LOAD PREDICTIONS
# ========================================

data = pd.read_csv(results_file)

data["datetime"] = pd.to_datetime(data["datetime"])


print("\n========================================")
print("XGBOOST MODEL VISUALIZATION")
print("========================================")

print(f"\nTotal predictions: {len(data)}")


# ========================================
# 3. SELECT A SMALL TIME WINDOW
# ========================================

plot_data = data.head(200)


# ========================================
# 4. CREATE GRAPH
# ========================================

plt.figure(figsize=(14, 6))

plt.plot(
    plot_data["datetime"],
    plot_data["actual_aqi"],
    label="Actual AQI",
    linewidth=2
)

plt.plot(
    plot_data["datetime"],
    plot_data["predicted_aqi"],
    label="XGBoost Predicted AQI",
    linewidth=2
)


# ========================================
# 5. GRAPH DETAILS
# ========================================

plt.title(
    "Actual vs Predicted AQI — XGBoost"
)

plt.xlabel("Date and Time")

plt.ylabel("AQI")

plt.legend()

plt.xticks(rotation=45)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()


print("\n========================================")
print("VISUALIZATION COMPLETED!")
print("========================================")