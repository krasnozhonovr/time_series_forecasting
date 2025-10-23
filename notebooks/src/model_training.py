from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_random_forest(X_train, y_train, X_test, y_test):
    """Trains RandomForest and returns metrics."""
    model = RandomForestRegressor(n_estimators=1000, max_depth=None, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    return model, metrics
