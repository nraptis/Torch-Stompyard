# weather_nn_train_test.py

import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


df1 = pd.read_csv("weather_train.csv")  # or weather_train.csv
print(df1["raining"].value_counts())
print("\nTraining Humidity stats by raining:")
print(df1.groupby("raining")["humidity"].describe())

df2 = pd.read_csv("weather_test.csv")  # or weather_train.csv
print(df2["raining"].value_counts())
print("\nTesting Humidity stats by raining:")
print(df2.groupby("raining")["humidity"].describe())


# -----------------------------
# Helper: load a CSV into tensors
# -----------------------------
def load_weather_csv(path):
    df = pd.read_csv(path)

    # Input tensor → humidity
    X = df["humidity"].values.astype("float32").reshape(-1, 1)

    # Output tensor → raining (True/False → 1/0)
    y = (df["raining"] == "True").astype("float32").values.reshape(-1, 1)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    return X_tensor, y_tensor

# -----------------------------
# 1. Load train + test separately
# -----------------------------
X_train, y_train = load_weather_csv("weather_train.csv")
X_test,  y_test  = load_weather_csv("weather_test.csv")

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)

print("train_ds = ", train_ds)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=8)

# -----------------------------
# 2. Define Sequential model
# -----------------------------
model = nn.Sequential(
    nn.Linear(1, 1)
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 3. Train
# -----------------------------
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)

    avg_loss = total_loss / len(train_ds)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs}, loss = {avg_loss:.4f}")

# -----------------------------
# 4. Evaluate on test set
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        logits = model(batch_X)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        correct += (preds == batch_y).sum().item()
        total += batch_y.numel()

accuracy = correct / total if total > 0 else float("nan")
print(f"Test accuracy: {accuracy:.3f}")

# -----------------------------
# 5. Example inference
# -----------------------------
with torch.no_grad():
    humidity_example = torch.tensor([[97.5]])
    prob = torch.sigmoid(model(humidity_example)).item()
    print(f"Humidity 97.5% → P(raining=True): {prob:.3f}")
