import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Loads call center data and parses dates."""
    df = pd.read_csv(file_path, sep=';', parse_dates=True)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans numeric and datetime columns, replaces commas, and creates timestamp."""
    df[['IS_LUNCH', 'IS_WORKTIME', 'IS_MORNING']] = (
        df[['IS_LUNCH', 'IS_WORKTIME', 'IS_MORNING']]
        .replace(',', '.', regex=True)
        .astype(float)
    )
    df['MONTH'] = pd.to_datetime(df['MONTH'] + '-01')
    df['DATESTART'] = pd.to_datetime(df['DATESTART'], dayfirst=True)
    df['RAZREZ'] = pd.to_timedelta(df['RAZREZ'] + ':00')
    df['TIMESTAMP'] = df['DATESTART'] + df['RAZREZ']
    return df
