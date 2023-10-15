import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import date, timedelta

# Load your LSTM model
model = load_model('your_lstm_model')

# Function to predict stock prices
def predict_stock_prices(test_data, time_step=15, pred_days=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    test_data = scaler.fit_transform(test_data.reshape(-1, 1))

    x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = time_step
    i = 0

    while i < pred_days:
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1

    return scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]

# Function to generate future dates
def generate_future_dates(start_date, num_days):
    future_dates = [start_date + timedelta(days=i) for i in range(num_days)]
    return future_dates

# Streamlit app
def main():
    st.title('Stock Price Prediction App')
    st.sidebar.title('Settings')

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with historical stock prices:", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader('Data Preview:')
        st.write(data.head())

        time_step = st.sidebar.slider("Select the time step:", min_value=1, max_value=30, value=15)
        pred_days = st.sidebar.slider("Select the number of days to predict:", min_value=1, max_value=365, value=30)
        future_date = st.sidebar.date_input("Select a future date for prediction:", date.today() + timedelta(days=pred_days))

        if st.sidebar.button("Predict Stock Prices"):
            st.subheader('Predicted Stock Prices:')
            test_data = data['Close'].values
            predicted_prices = predict_stock_prices(test_data, time_step, pred_days)
            
            future_dates = generate_future_dates(date.today(), pred_days)
            future_prices_df = pd.DataFrame({'Date': future_dates, 'Predicted Prices': predicted_prices})
            
            st.write(f"Predicted Stock Prices for {future_date}:")
            st.dataframe(future_prices_df[future_prices_df['Date'] == future_date])

if __name__ == '__main__':
    main()