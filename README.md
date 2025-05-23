# Stock Price Trend Prediction Using LSTM

A complete end-to-end workflow for forecasting Apple Inc. (AAPL) closing prices using a three-layer LSTM network in Google Colab.

---

## 🚀 Project Overview

1. **Data Acquisition**  
   Download historical daily price data for AAPL (2010-01-01 to 2024-01-01) via the Yahoo Finance API (`yfinance`).

2. **Exploratory Data Analysis (EDA)**  
   - Data cleaning & validation  
   - Descriptive statistics  
   - Closing price plot  
   - Technical indicators (MA20, MA50, RSI14)  
   - Correlation heatmap

3. **Data Preparation**  
   - Feature selection (Close, MA20, MA50, RSI14)  
   - MinMax scaling  
   - Rolling 60-day sequence generation  
   - Train/test split (80/20)

4. **Model Building & Training**  
   - Three-layer LSTM (128 → 64 → 32 units) + Dropout(0.2)  
   - RMSprop optimizer (lr = 1e-4)  
   - Callbacks: EarlyStopping, ReduceLROnPlateau  
   - Up to 150 epochs, batch_size = 8, validation_split = 10%

5. **Evaluation & Visualization**  
   - Inverse-scaling predictions  
   - Actual vs. Predicted plot  
   - Metrics: MSE, RMSE, MAE, R², Directional Accuracy

---

## ▶️ Usage

1. **Open** the Colab notebook (`Stock-Price-Trend-Prediction-Using-LSTM.ipynb`) in Google Colab.  
2. **Run** each code cell in order:  
   - Environment Setup  
   - Data Acquisition  
   - EDA (plots & indicators)  
   - Data Preparation  
   - Model Build & Training  
   - Evaluation & Visualization  
3. **Inspect** the saved model weights in `models/lstm_best_tuned.weights.h5`.

---

## 📊 Results

- Training & Validation Loss plot  
- Actual vs. Predicted closing price plot  
- Final metrics displayed in console:  
  - MSE  
  - RMSE  
  - MAE  
  - R²  

---

## 💡 Notes

- Swap `ticker = "AAPL"` with any other valid symbol to forecast a different stock.  
- Adjust `n_steps`, `batch_size`, `epochs`, or the network architecture to experiment.  
- Use Streamlit to build an interactive dashboard around this model.

---

## 👤 Author

**Rohit Kumar Mourya**  
AI & Machine Learning Intern, Elevate Labs  

