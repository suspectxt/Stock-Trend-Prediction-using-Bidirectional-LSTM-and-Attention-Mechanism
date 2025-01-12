# Stock Price Prediction using LSTM and Technical Analysis

## Overview

This project aims to predict the future price direction (up or down) of the Sensex Index using a **Bidirectional LSTM (Long Short-Term Memory)** model. The model leverages **technical analysis indicators** and historical stock price data to forecast price trends for a 3-month period. The prediction is made based on a combination of technical indicators and historical price information, allowing the model to make data-driven predictions about stock price movements.

The performance of the model is evaluated using **rolling-window backtesting** to simulate real-world prediction conditions, ensuring that the model generalizes well to unseen data.

## Key Features

- **Data Used:** Historical stock prices (e.g., Sensex) and related technical indicators.
- **Technical Indicators:**
  - **Exponential Moving Averages (EMA):** EMA for 7, 20, and 111 days.
  - **Relative Strength Index (RSI):** Momentum-based indicator to identify overbought or oversold conditions.
  - **Accumulation/Distribution Line (A/D):** Volume-based indicator assessing buying and selling pressure.
  - **Stock Closing Price and Volume.**

## Approach

### 1. **Data Preprocessing and Feature Engineering**
   - **Resampling:** Data is resampled to monthly frequency, using the closing price and volume for each month.
   - **Technical Indicators Calculation:** The `ta` (Technical Analysis) library is used to compute various technical indicators (EMA, RSI, A/D).
   - **Normalization:** The data is normalized using **MinMaxScaler** to scale the features to a range between 0 and 1.

### 2. **Sequence Generation for LSTM**
   - Sequences of past `N` months are created to be used as input for training the LSTM model. 
   - The model predicts the next month's closing price based on these sequences, aiming to forecast the price direction over a 3-month horizon.

### 3. **Model Architecture**
   - The model uses **Bidirectional LSTM** layers to capture dependencies in both the past and future price trends.
   - An **Attention Layer** is applied to allow the model to focus on important time steps, improving prediction accuracy.
   - The output layer produces a single predicted value (the next month's closing price).

### 4. **Backtesting Approach**
   - **Rolling-Window Backtesting:** The model is trained on historical data up to a certain point and tested on the following 3 months.
   - The backtesting procedure simulates real-world predictions where the model is continuously updated with new data as time progresses.
   - **Early Stopping:** To prevent overfitting, the model training process uses early stopping based on validation loss.

### 5. **Prediction**
   - The model outputs a predicted closing price for the next month. 
   - The model’s performance is evaluated by comparing predicted values with actual closing prices.

### 6. **Performance Evaluation**
   - **Accuracy in Predicting Negative Months:** This metric measures how accurately the model can predict stock price drops over a 3-month period.
   - **Mean Squared Error (MSE):** Used to assess the model's performance by evaluating the difference between predicted and actual closing prices.
   - The accuracy achieved in predicting whether the stock will go up or down over the next 3 months is an important measure of the model's predictive power.

## Results

- **Best Accuracy:** The highest accuracy obtained for predicting whether the stock will be in a positive or negative trend over the 3-month period was **77%**.
- **Monthly Returns Prediction:** The model successfully predicts the monthly returns direction (positive or negative) with a reasonable level of accuracy.

## File Structure

- **`stock_prediction_model.ipynb`**: Jupyter notebook containing the model implementation and backtesting.
- **`sensex.csv`**: Historical stock data of Sensex (used for training and testing).
- **`predictions_and_actuals.csv`**: CSV file containing the predicted and actual stock prices.
- **`monthly_returns.csv`**: CSV file containing monthly returns based on the actual and predicted values.
- **`model.pkl`**: Serialized model file for saving and loading the trained model.

## How to Use

1. **Install Dependencies:**  
   To run the model, make sure you have the following dependencies installed:
   ```bash
   pip install pandas numpy tensorflow scikit-learn ta matplotlib
   ```

2. **Data Preparation:**  
   Download (from [yfinance](https://finance.yahoo.com))or prepare the stock data in CSV format with columns `Date`, `Close`, and `Volume`.

3. **Running the Model:**
   - Run the Jupyter notebook or Python script to start training the model.
   - The model will train on the historical data and then perform rolling-window backtesting to evaluate its performance.

4. **Results:**  
   After running the model, the predicted and actual values will be saved in CSV files for further analysis.

## Conclusion

This project demonstrates the application of **Bidirectional LSTM** and **technical analysis** for stock price prediction. The model, when backtested with a rolling window approach, is able to predict stock price movements with a high degree of accuracy, particularly in predicting negative price movements. This makes it a valuable tool for making informed investment decisions.

## Future Work

- **Hyperparameter Tuning:** The model can be improved further by optimizing hyperparameters such as the number of LSTM units, dropout rates, and learning rates.
- **Additional Features:** More technical indicators and external data (such as news sentiment or economic indicators) can be added to improve prediction accuracy.
- **Evaluation Metrics:** Explore additional evaluation metrics like Precision, Recall, and F1-Score to measure the model's performance in different scenarios.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/suspectxt/Stock-Trend-Prediction-using-Bidirectional-LSTM-and-Attention-Mechanism/blob/main/LICENSE) file for details.
