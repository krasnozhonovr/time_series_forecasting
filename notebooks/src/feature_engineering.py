import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds day-of-week, billing, and lag-based features."""
    df['weekday'] = df.index.weekday
    df = df.join(pd.get_dummies(df['weekday'], prefix='weekday'))
    df['critical_day'] = df['DATESTART'].dt.day.isin([3,4,10,13,14,23,24]).astype(int)
    df['billing_day'] = df['DATESTART'].dt.day.isin([5,15,25]).astype(int)

    for i in range(1, 11):
        df[f'lag_{i}'] = df['CNT_CALLS'].shift(i)

    df['rolling_mean_7'] = df['CNT_CALLS'].rolling(7).mean()
    df['rolling_std_3'] = df['CNT_CALLS'].rolling(3).std()

    return df.dropna()
