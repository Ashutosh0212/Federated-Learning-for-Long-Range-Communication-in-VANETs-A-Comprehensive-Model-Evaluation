import matplotlib.pyplot as plt

# Updated Data for the models
models = ["Simple LSTM", "Basic CNN", "Simple Hybrid", "Enhanced GRU"]

# Rounds data
rounds = [1, 2, 3]

# Updated Loss data for each model
training_loss = {
    "Simple LSTM": [0.00544, 0.00178, 0.00129],
    "Basic CNN": [0.00603, 0.00196, 0.00132],
    "Simple Hybrid": [0.00870, 0.00187, 0.00131],
    "Enhanced GRU": [0.00631, 0.00460, 0.00165],  # Updated GRU values
}

validation_loss = {
    "Simple LSTM": [0.0020, 0.0018, 0.0015],
    "Basic CNN": [0.0028, 0.0020, 0.0015],
    "Simple Hybrid": [0.0021, 0.0016, 0.0014],
    "Enhanced GRU": [0.0022, 0.0016, 0.0014],  # Updated GRU values
}

validation_mae = {
    "Simple LSTM": [0.0247, 0.0206, 0.0184],
    "Basic CNN": [0.0313, 0.0219, 0.0200],
    "Simple Hybrid": [0.0256, 0.0206, 0.0173],
    "Enhanced GRU": [0.0265, 0.0219, 0.0185],  # Updated GRU values
}

# Plotting the trends
plt.figure(figsize=(12, 10))

# Training Loss Plot
plt.subplot(3, 1, 1)
for model in models:
    plt.plot(rounds, training_loss[model], marker='o', label=model)
plt.title("Training Loss Trend Across Rounds")
plt.xlabel("Federated Learning Rounds")
plt.ylabel("Training Loss")
plt.legend()
plt.grid()

# Validation Loss Plot
plt.subplot(3, 1, 2)
for model in models:
    plt.plot(rounds, validation_loss[model], marker='o', label=model)
plt.title("Validation Loss Trend Across Rounds")
plt.xlabel("Federated Learning Rounds")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid()

# Validation MAE Plot
plt.subplot(3, 1, 3)
for model in models:
    plt.plot(rounds, validation_mae[model], marker='o', label=model)
plt.title("Validation MAE Trend Across Rounds")
plt.xlabel("Federated Learning Rounds")
plt.ylabel("Validation MAE")
plt.legend()
plt.grid()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
