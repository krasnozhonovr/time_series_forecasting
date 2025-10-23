# 📞 Call Center Time Series Forecasting with Machine Learning

This project implements **time series forecasting** for inbound call volumes in a call center using **LSTM networks** and **Random Forest Regression**.  
It is based on our research presented at the *9th International Conference on Digital Technologies in Education, Science and Industry (DTESI 2024)*, Almaty, Kazakhstan.

## 🧠 Objective
To compare traditional ensemble and deep learning approaches for forecasting call volume and optimizing staffing.

## 🧩 Models Used
- **Random Forest Regressor** — stable performance on small datasets  
- **LSTM Neural Network** — superior accuracy and ability to model sequential dependencies

## ⚙️ Methods
- Feature engineering (lags, rolling statistics, one-hot encoding)
- Anomaly detection using Isolation Forest
- Model evaluation via MAE, RMSE, R²

## 📊 Key Results
| Metric | Random Forest (Daily) | LSTM (Daily) | Random Forest (Half-Hourly) | LSTM (Half-Hourly) |
|---------|----------------------:|-------------:|-----------------------------:|--------------------:|
| MAE | 481 | **387** | 14 | **13** |
| RMSE | 674 | **595** | 26 | **23** |
| R² | 0.8958 | **0.9350** | 0.9686 | **0.9720** |

LSTM models achieved **higher accuracy and better fit** for fine-grained (half-hourly) data.

## 🛠️ Tools
Python, TensorFlow, Scikit-learn, Pandas, Matplotlib

## 🧾 Reference
Krasnozhonov, R., Altaibek, A., Ydyrys, A., Nurtas, M.  
*Time Series Forecast of Inbound Call Volume in Call Center using Machine Learning Methods*  
DTESI 2024, Almaty, Kazakhstan.  
[PDF Summary](https://ceur-ws.org/Vol-3966/W3Paper15.pdf)

## 📁 File Overview

- `src/data_preprocessing.py` – handles data loading, cleaning, and formatting
- `src/feature_engineering.py` – creates time-based, calendar, and lag features
- `src/model_training.py` – trains ML and LSTM models
- `src/forecasting.py` – generates and visualizes future forecasts
- `notebooks/call_center_forecast.ipynb` – full end-to-end workflow
