import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================
# 1. LOAD DATASET
# ============================================

file_path = "data/processed/hyderabad_aqi_weather_features.csv"

data = pd.read_csv(file_path)

print("Dataset shape:", data.shape)


# ============================================
# 2. SELECT RICHER FEATURES
# ============================================

features = [
    "aqi",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "surface_pressure",
    "hour",
    "day_of_week",
    "is_weekend",
    "aqi_previous_hour",
    "aqi_previous_3_hours",
    "aqi_previous_6_hours",
    "aqi_previous_24_hours",
    "aqi_rolling_24h"
]

target = "target_aqi_next_hour"


X = data[features].values
y = data[target].values

print("Input shape:", X.shape)
print("Target shape:", y.shape)


# ============================================
# 3. TIME-BASED TRAIN-TEST SPLIT
# ============================================

split_index = int(len(X) * 0.8)

X_train_raw = X[:split_index]
X_test_raw = X[split_index:]

y_train_raw = y[:split_index]
y_test_raw = y[split_index:]

print("\nRaw training shape:", X_train_raw.shape)
print("Raw testing shape:", X_test_raw.shape)


# ============================================
# 4. SCALE USING TRAINING DATA ONLY
# ============================================

scaler_X = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)

X_test_scaled = scaler_X.transform(X_test_raw)


scaler_y = MinMaxScaler()

y_train_scaled = scaler_y.fit_transform(
    y_train_raw.reshape(-1, 1)
).flatten()

y_test_scaled = scaler_y.transform(
    y_test_raw.reshape(-1, 1)
).flatten()


print("\nScaled training shape:", X_train_scaled.shape)
print("Scaled testing shape:", X_test_scaled.shape)


# ============================================
# 5. CREATE 24-HOUR TRAINING SEQUENCES
# ============================================

sequence_length = 24

X_train_sequences = []
y_train_sequences = []

for i in range(sequence_length, len(X_train_scaled)):

    X_train_sequences.append(
        X_train_scaled[i - sequence_length:i]
    )

    y_train_sequences.append(
        y_train_scaled[i]
    )


X_train_sequences = np.array(X_train_sequences)
y_train_sequences = np.array(y_train_sequences)


# ============================================
# 6. CREATE 24-HOUR TEST SEQUENCES
# ============================================

X_test_sequences = []
y_test_sequences = []

for i in range(sequence_length, len(X_test_scaled)):

    X_test_sequences.append(
        X_test_scaled[i - sequence_length:i]
    )

    y_test_sequences.append(
        y_test_scaled[i]
    )


X_test_sequences = np.array(X_test_sequences)
y_test_sequences = np.array(y_test_sequences)


print("\nLSTM training sequences:", X_train_sequences.shape)
print("LSTM testing sequences:", X_test_sequences.shape)


# ============================================
# 7. CONVERT TO PYTORCH TENSORS
# ============================================

X_train_tensor = torch.tensor(
    X_train_sequences,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_sequences,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_sequences,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test_sequences,
    dtype=torch.float32
)


print("\nPyTorch shapes:")
print("X_train:", X_train_tensor.shape)
print("y_train:", y_train_tensor.shape)
print("X_test:", X_test_tensor.shape)
print("y_test:", y_test_tensor.shape)


# ============================================
# 8. DEFINE LSTM MODEL
# ============================================

class AQILSTM(torch.nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout
    ):

        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = torch.nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        # Last time step
        out = out[:, -1, :]

        # Predict AQI
        out = self.fc(out)

        return out


# ============================================
# 9. CREATE MODEL
# ============================================

model = AQILSTM(
    input_size=len(features),
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)

print("\nLSTM Model:")
print(model)


# ============================================
# 10. LOSS FUNCTION
# ============================================

criterion = torch.nn.MSELoss()


# ============================================
# 11. OPTIMIZER
# ============================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================
# 12. TRAIN MODEL
# ============================================

epochs = 20

print("\nStarting training...\n")

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    predictions = model(X_train_tensor)

    loss = criterion(
        predictions.squeeze(),
        y_train_tensor
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {loss.item():.6f}"
    )


print("\nTraining completed! 🎉")


# ============================================
# 13. TEST MODEL
# ============================================

model.eval()

with torch.no_grad():

    test_predictions_scaled = model(
        X_test_tensor
    ).squeeze().numpy()


# ============================================
# 14. CONVERT BACK TO ACTUAL AQI
# ============================================

actual_aqi = scaler_y.inverse_transform(
    y_test_sequences.reshape(-1, 1)
).flatten()

predicted_aqi = scaler_y.inverse_transform(
    test_predictions_scaled.reshape(-1, 1)
).flatten()


# ============================================
# 15. CALCULATE METRICS
# ============================================

mae = mean_absolute_error(
    actual_aqi,
    predicted_aqi
)

rmse = np.sqrt(
    mean_squared_error(
        actual_aqi,
        predicted_aqi
    )
)

r2 = r2_score(
    actual_aqi,
    predicted_aqi
)


# ============================================
# 16. DISPLAY RESULTS
# ============================================

print("\n============================================")
print("UPGRADED LSTM TEST RESULTS")
print("============================================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ============================================
# 17. SAVE PREDICTIONS
# ============================================

results = pd.DataFrame({
    "actual_aqi": actual_aqi,
    "predicted_aqi": predicted_aqi
})

output_file = "data/processed/aqi_lstm_rich_predictions.csv"

results.to_csv(
    output_file,
    index=False
)

print("\nPredictions saved to:")
print(output_file)