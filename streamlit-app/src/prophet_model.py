from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# Function to prepare data for Prophet
def prepare_data_for_prophet(data, symbol, lookback_days=90):
    end_date = data.index.max()
    start_date = end_date - timedelta(days=lookback_days)

    # Handle both 'Close' and 'close' column names
    close_col = 'Close' if 'Close' in data.columns else 'close'
    prophet_data = data[(data.index >= start_date) & 
                        (data['symbol'] == symbol)][[close_col]].reset_index()
    prophet_data.columns = ['ds', 'y']
    prophet_data['ds'] = prophet_data['ds'].dt.tz_localize(None)
    return prophet_data


# Function to run Prophet model and generate plots
# def run_prophet_model(data, symbol, forecast_days=30):
#     model = Prophet(daily_seasonality=True)
#     model.fit(data)
#
#     future_dates = model.make_future_dataframe(periods=forecast_days)
#     forecast = model.predict(future_dates)
#
#     fig1, ax1 = plt.subplots()
#     model.plot(forecast, ax=ax1)
#     ax1.set_title(f'{symbol} Forecast')
#
#     fig2, ax2 = plt.subplots()
#     model.plot_components(forecast, ax=ax2)
#     ax2.set_title(f'{symbol} Forecast Components')
#
#     return forecast, fig1, fig2

def run_prophet_model(data, symbol, forecast_days=30):
    model = Prophet(daily_seasonality=True)
    model.fit(data)

    future_dates = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future_dates)

    fig1 = model.plot(forecast)
    plt.title(f'{symbol} Forecast')

    fig2 = model.plot_components(forecast)
    plt.suptitle(f'{symbol} Forecast Components')

    return forecast, fig1, fig2
