import re
import matplotlib.pyplot as plt

# Path to the metrics log
log_path = "/home/francisco/trading-model/data-logs/learning_metrics.log"

val_losses = []
mses = []
r2_scores = []

# Parse metrics log with debugging print statements
with open(log_path, 'r') as file:
    for line in file:
        print(f"Processing line: {line.strip()}")  # Debug print
       # Updated regex to be more flexible with potential number formats
        match = re.search(r"Validation Loss: ([\d.]+), MSE: ([\d.]+), R² Score: ([-+]?[0-9]*\.?[0-9]+)", line)
        if match:
            val_losses.append(float(match.group(1)))
            mses.append(float(match.group(2)))
            r2_scores.append(float(match.group(3)))
            print(f"Matched Line - Validation Loss: {match.group(1)}, MSE: {match.group(2)}, R²: {match.group(3)}")


# Plotting for visual assessment
plt.figure(figsize=(12, 5))

# Plot Validation Loss
plt.subplot(1, 3, 1)
plt.plot(val_losses, label="Validation Loss")
plt.title("Validation Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()

# Plot MSE
plt.subplot(1, 3, 2)
plt.plot(mses, label="Mean Squared Error")
plt.title("MSE Over Time")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()

# Plot R² Score
plt.subplot(1, 3, 3)
plt.plot(r2_scores, label="R² Score")
plt.title("R² Score Over Time")
plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.legend()

plt.tight_layout()
plt.show()
