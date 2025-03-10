from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import timedelta


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
    
    # Remove weekends
    prophet_data = prophet_data[prophet_data['ds'].dt.weekday < 5]
    return prophet_data


# Function to run Prophet model and generate plots
def run_prophet_model(data, symbol, forecast_days=30):
    model = Prophet(daily_seasonality=True)
    model.fit(data)

    future_dates = model.make_future_dataframe(periods=forecast_days)
    future_dates = future_dates[future_dates['ds'].dt.weekday < 5]
    forecast = model.predict(future_dates)

    fig1 = model.plot(forecast)
    plt.title(f'{symbol} Forecast')

    fig2 = model.plot_components(forecast)
    plt.suptitle(f'{symbol} Forecast Components')

    return forecast, fig1, fig2
