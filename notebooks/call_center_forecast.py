# === 3. Anomaly Detection & Correction ===
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("🔹 Running anomaly detection...")

# Make a copy of the main dataset
anomaly_df = data.copy()

# Visualize original call volume
plt.figure(figsize=(12, 5))
plt.plot(anomaly_df['TIMESTAMP'], anomaly_df['CNT_CALLS'], label='Raw Call Volume', color='steelblue')
plt.title('Inbound Calls Before Cleaning')
plt.xlabel('Date')
plt.ylabel('Number of Calls')
plt.legend()
plt.show()

# Normalize the CNT_CALLS column
scaler = StandardScaler()
scaled_calls = scaler.fit_transform(anomaly_df[['CNT_CALLS']])

# Isolation Forest model for anomaly detection
iso = IsolationForest(contamination=0.001, random_state=42)
anomaly_df['anomaly_flag'] = iso.fit_predict(scaled_calls)

# Convert -1/1 to human-readable
anomaly_df['anomaly_flag'] = anomaly_df['anomaly_flag'].map({1: 'normal', -1: 'anomaly'})

# Visualize detected anomalies
plt.figure(figsize=(12, 5))
plt.plot(anomaly_df['TIMESTAMP'], anomaly_df['CNT_CALLS'], label='Call Volume', color='grey')
plt.scatter(anomaly_df.loc[anomaly_df['anomaly_flag']=='anomaly', 'TIMESTAMP'],
            anomaly_df.loc[anomaly_df['anomaly_flag']=='anomaly', 'CNT_CALLS'],
            color='red', label='Detected Anomalies')
plt.title('Detected Anomalies in Call Data')
plt.xlabel('Date')
plt.ylabel('Calls')
plt.legend()
plt.show()

# Replace anomalies with median value
median_value = anomaly_df.loc[anomaly_df['anomaly_flag']=='normal', 'CNT_CALLS'].median()
anomaly_df.loc[anomaly_df['anomaly_flag']=='anomaly', 'CNT_CALLS'] = median_value

print(f"Replaced {sum(anomaly_df['anomaly_flag'] == 'anomaly')} anomalies with median = {median_value:.2f}")

# Drop helper column
anomaly_df.drop(columns=['anomaly_flag'], inplace=True)

# Use cleaned data going forward
data = anomaly_df.copy()

plt.figure(figsize=(12, 5))
plt.plot(data['TIMESTAMP'], data['CNT_CALLS'], label='Cleaned Call Volume', color='seagreen')
plt.title('Inbound Calls After Cleaning')
plt.xlabel('Date')
plt.ylabel('Number of Calls')
plt.legend()
plt.show()