import numpy as np
import pandas as pd
from datetime import timedelta

def generate_future_features(start_time, end_time, freq='30T'):
    """Generates a future DataFrame with time-based features for forecasting."""
    future_df = pd.DataFrame({'timestamp': pd.date_range(start=start_time, end=end_time, freq=freq)})
    future_df['hour'] = future_df['timestamp'].dt.hour
    future_df['weekday'] = future_df['timestamp'].dt.weekday

    # Time-based binary features
    future_df['IS_DAY'] = future_df['hour'].between(8, 21).astype(int)
    future_df['IS_LUNCH'] = future_df['hour'].between(12, 13).astype(int) * 0.2
    future_df['IS_WORKTIME'] = future_df['hour'].between(8, 18).astype(int) * 0.2
    future_df['IS_MORNING'] = future_df['hour'].between(8, 9).astype(int) * 0.4

    # Calendar-based features
    future_df['WORKDAYS'] = (future_df['weekday'] < 5).astype(int)
    future_df['HOLIDAYS'] = (future_df['weekday'] >= 5).astype(int)
    future_df['day'] = future_df['timestamp'].dt.day
    future_df['billing_day'] = future_df['day'].isin([5, 15, 25]).astype(int)
    future_df['critical_day'] = future_df['day'].isin([3, 4, 13, 14, 23, 24]).astype(int)

    return future_df.set_index('timestamp')


def make_forecast(model, df, feature_cols, scaler_x, scaler_y, steps=48):
    """Generates forecasts for N future time steps using a trained model."""
    predictions = []
    input_data = df[feature_cols].copy()

    for _ in range(steps):
        scaled_features = scaler_x.transform(input_data.iloc[[-1]])
        scaled_features = scaled_features.reshape(1, 1, -1)
        pred_scaled = model.predict(scaled_features)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        predictions.append(pred)

        # Update the DataFrame for next step
        new_row = input_data.iloc[[-1]].copy()
        new_row['CNT_CALLS'] = pred
        input_data = pd.concat([input_data, new_row]).iloc[1:]

    return np.array(predictions)
