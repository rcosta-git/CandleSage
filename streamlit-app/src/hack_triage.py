import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

def get_fear_greed_index(limit=15):
    url = "https://api.alternative.me/fng/"
    params = {
        'limit': limit,
        'format': 'json',
        'date_format': 'us'
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['value'] = df['value'].astype(int)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df[['timestamp', 'value', 'value_classification']]
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def plot_lr_channel(data, ticker, period, std_dev=2):
    # Access the 'Close' price for the specified ticker
    df = data[data['symbol'] == ticker].query(f"period == {period}")  # Extract the ticker data
    x = np.arange(len(df))
    y = df['Close'].values  # Access the 'Close' price directly

    coeffs = np.polyfit(x, y, 1)
    trendline = np.polyval(coeffs, x)

    residuals = y - trendline
    std = np.std(residuals)
    upper_band = trendline + std_dev * std
    lower_band = trendline - std_dev * std

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax.plot(df.index, trendline, label='Trendline', color='red')
    ax.fill_between(df.index, lower_band, upper_band, color='blue', alpha=0.2, 
                    label='Std Dev Bands')

    ax.set_title(f"{ticker} Price Chart with Linear Regression Channel")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    return fig
