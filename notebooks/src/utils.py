import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred):
    """Calculates MAE, RMSE, and R² metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def plot_predictions(y_true, y_pred, title="Model Predictions vs Actual"):
    """Plots true vs predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.8)
    plt.plot(y_pred, label='Predicted', alpha=0.8)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Calls')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_metrics(metrics, filename="results/metrics_report.json"):
    """Saves metrics dictionary as a JSON file."""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
