from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import scipy.stats as stats
import requests
import keys
import numpy as np
# import matplotlib.pyplot as plt
import neattext.functions as nfx
import re

# Box1(Data Prep) -- Starts
def fetch_market_data(symbols_periods, end_date=None):
    """
    Fetches historical market data for both stocks and cryptocurrencies
    Accepts symbols in format {'BTC': [15,30,90], 'AAPL': [15,30,90]}
    Returns combined DataFrame with timestamp, symbol, period, and price data
    """
    
    api_key = keys.APCA_API_KEY_ID
    api_secret = keys.APCA_API_SECRET_KEY
    
    crypto_client = CryptoHistoricalDataClient(api_key,api_secret )
    stock_client = StockHistoricalDataClient(api_key,api_secret)
    end_date = end_date or datetime.now()
    all_data = []
    
    for symbol, periods in symbols_periods.items():
        for days in periods:
            start_date = end_date - timedelta(days=days)
            
            try:
                # Determine asset type
                if symbol.upper() in {'BTC', 'ETH', 'USDT'}:  # Add crypto symbols as needed
                    request = CryptoBarsRequest(
                        symbol_or_symbols=[f"{symbol}/USD"],
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date
                    )
                    bars = crypto_client.get_crypto_bars(request)
                else:
                    request = StockBarsRequest(
                        symbol_or_symbols=[symbol],
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date
                    )
                    bars = stock_client.get_stock_bars(request)
                
                # Process and label data
                df = bars.df.reset_index()
                df['period'] = days
                df['symbol'] = symbol
                all_data.append(df)
                
            except Exception as e:
                print(f"Error fetching {symbol} ({days}d): {str(e)}")
    
    return pd.concat(all_data).set_index('timestamp') if all_data else pd.DataFrame()
# Box1 -- Ends
# Box2 (Models) -- Starts
def calculate_statistics(combined_df):
    """
    Calculates volatility metrics from fetched data
    Returns DataFrame with statistical analysis for each symbol/period combination
    """
    results = []
    
    for (symbol, period), group in combined_df.groupby(['symbol', 'period']):
        closes = group['close'].ffill().bfill()
        if len(closes) < 2:
            continue
            
        mean = closes.mean()
        std_dev = closes.std()
        variance = closes.var()
        
        metrics = {
            'symbol': symbol,
            'days': period,
            'close': closes.iloc[-1],
            'mean': mean,
            'std_dev': std_dev,
            'variance': variance,
            'cv': std_dev / mean,
            'skewness': stats.skew(closes),
            'kurtosis': stats.kurtosis(closes),
            'z_score': (closes.iloc[-1] - mean) / std_dev
        }
        
        # Calculate confidence intervals
        df = len(closes) - 1
        t70 = stats.t.ppf(0.85, df)
        t95 = stats.t.ppf(0.975, df)
        
        metrics.update({
            't_70_start': mean - t70 * std_dev,
            't_70_end': mean + t70 * std_dev,
            't_95_start': mean - t95 * std_dev,
            't_95_end': mean + t95 * std_dev
        })
        
        results.append(metrics)
    
    cols = ['symbol', 'days', 'close', 'mean', 'std_dev', 'variance',
            'cv', 'skewness', 'kurtosis', 't_70_start', 't_70_end',
            't_95_start', 't_95_end', 'z_score']
    
    return pd.DataFrame(results)[cols].round(2)


# Box4 (AI powered Suggestions)- Starts
def AIopinion(messages, model = 'sonar'):
    from openai import OpenAI

    YOUR_API_KEY = st.secrets["OPENAI_API_KEY"]
    
    import re
    from IPython.display import clear_output
    
    client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")
    
    try:
        response_stream = client.chat.completions.create(
            model=model, #sonar-pro #sonar-reasoning-pro #sonar #sonar-reasoning
            messages=messages,
            stream=True,
        )
    
        full_response = []
        link_map = {}
        link_counter = 1
        buffer = ""
    
        for chunk in response_stream:
            if hasattr(chunk, 'citations') and chunk.citations:
                # Enumerate URLs starting at index 1 for proper citation numbering
                link_map = {str(i): url for i, url in enumerate(chunk.citations, start=1)}
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                buffer += content
                
                # Process URLs incrementally
                while True:
                    match = re.search(r'https?://[^\s\)\]\}]+', buffer)
                    if not match:
                        break
                        
                    url = match.group()
                    if url not in link_map:
                        link_map[url] = link_counter
                        link_counter += 1
                    
                    # Replace URL with reference marker
                    buffer = buffer.replace(url, f'[{link_map[url]}]', 1)
                
                # Print processed content and clear buffer
                print(buffer, end='', flush=True)
                full_response.append(buffer)
                buffer = ""
    
        # Print remaining buffer content
        if buffer:
            print(buffer, end='', flush=True)
            full_response.append(buffer)
    
        # Add references appendix
        print("\n\n--- References ---")
        for url, num in sorted(link_map.items(), key=lambda x: x[1]):
            print(f"[{num}] {url}")

        # Add references appendix
        full_response.append("\n\n--- References ---")
        for url, num in sorted(link_map.items(), key=lambda x: x[1]):
            full_response.append(f"[{num}] {url}")
        return "\n".join(full_response)

    except Exception as e:
        clear_output()
        print(f"Error occurred: {str(e)}")

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


import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def plot_emas_streamlit(df):
    """
    Plots the Close Price along with 20, 50, and 100 EMAs for each symbol in the DataFrame,
    and displays the plots on a Streamlit app.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing columns ['symbol', 'close'] with a datetime index.
    """
    # Define EMA spans
    ema_spans = [20, 50, 100]

    # Group data by symbol
    grouped_data = df.groupby('symbol')

    # Process each symbol separately
    for symbol, data in grouped_data:
        # Ensure data is sorted by timestamp
        data = data.sort_index()

        # Calculate EMAs
        for span in ema_spans:
            data[f'{span} EMA'] = data['close'].ewm(span=span, adjust=False).mean()

        # Plot Close Price and EMAs
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['close'], label=f'{symbol} Close Price', color='blue')
        for span in ema_spans:
            plt.plot(data.index, data[f'{span} EMA'], label=f'{span} EMA')
        
        # Customize plot
        plt.title(f'{symbol} Close Price and EMAs')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid()

        # Display the plot in Streamlit
        st.pyplot(plt)
        plt.close()
    return data


def analyze_data(stats_df, fear_greed_df, emas_df,user_input):
    messages = [{
        "role": "user",
        "content": f"""
        Looking at fear greed index , statistics, kurtosis, cv , Skewness, and 20-50-100 EMAs, Predict the price movement would be upward
        or downward with what chance of confidence, can you say that.
        
        Also Suggest 3 strategies as below, Options or otherwise , 
        1. Low Risk, 2.Medium Risk, 3.High Risk
        
        {user_input}
        
        ## Fear & Greed Index:
        {fear_greed_df.to_markdown()}.

        ## EMAs:
        This chart shows the Close Price along with the 20, 50, and 100 EMAs.
        You can talk about it, if it is not empty:
        {emas_df.to_markdown()}

        ##
        {stats_df.to_markdown()}
 
        """
    }]
    output_of_ai = AIopinion(messages=messages)
    return output_of_ai

def clean_text(text):
    cleaned_text = nfx.remove_stopwords(text)
    cleaned_text = nfx.remove_punctuations(cleaned_text)
    cleaned_text = nfx.remove_special_characters(cleaned_text)
    return cleaned_text
# Function to clean and format text for markdown rendering
def clean_text_old(text):
    # Fix non-breaking spaces
    text = text.replace("\xa0", " ")

    # Normalize dashes
    text = text.replace("\u2010", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")

    # Handle price ranges like "$330-$325" or "$330 to $325"
    text = re.sub(r'(\d)-\n(\d)', r'\1-\2', text)
    text = re.sub(r'(\d)\n-\n(\d)', r'\1-\2', text)

    # Handle currency symbols without breaking them from the number
    text = re.sub(r'(\$|€|£|¥)(\d)', r'\1\2', text)

    # Correct cases where words are split across lines
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'(\w)\n(\w)', r'\1\2', text)

    # Fix spacing around punctuation and hyphenated words
    text = re.sub(r'\s*-', ' -', text)
    text = re.sub(r'(\d)\s*-([^\d])', r'\1 -\2', text)

    # Ensure spacing before and after parentheses
    text = re.sub(r'\s*(\(|\))\s*', r'\1', text)

    # Fix cases with missing spaces around punctuation
    text = re.sub(r'\s*(,|\.\s*)', r'\1 ', text)
    text = re.sub(r'\s*(\(|\))\s*', r'\1', text)

    # Merge lines that were unnecessarily split
    text = re.sub(r'(\S)\n', r'\1 ', text)

    # Remove extra spaces between sentences or words
    text = re.sub(r'\s+', ' ', text)

    # Fix spacing around parentheses and symbols
    text = re.sub(r'\(\s*', r'(', text)  # Remove space after opening paren
    text = re.sub(r'\s*\)', r')', text)  # Remove space before closing paren
    text = re.sub(r'(?<!\s)\(', r' (', text)  # Add space before opening paren
    text = re.sub(r'\)(?!\s|\.|,|\))', r') ', text)  # Add space after closing

    # Fix decimal numbers and currency formatting
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)  # Fix decimals
    text = re.sub(r'\$\s+', r'$', text)  # Remove space after dollar sign
    text = re.sub(r'(\$)\s*\n+\s*(\d)', r'\1\2', text)  # Fix split currency
    text = re.sub(r'(?<!\n)(\d)\s*\n+\s*(?=[\d\.])', r'\1', text)
    text = re.sub(r'(?<!\n|\*)(\d+(?:\.\d+)?)\s*\n+\s*(?=%|\w)', r'\1 ', text)

    # Handle colons in titles and sections
    text = re.sub(
        r'([^\n:]+:)\s+(?=\w)', r'\1 ', text)  # Normal text after :
    text = re.sub(
        r'([^\n:]+:)\s*(?=\d+\.\s|[-–]\s)', r'\1\n\n', text)  # List

    # Handle numbered lists by adding newlines
    text = re.sub(r'(?:^|\n|\s{2,})(?<!\d)(\d+)\s*\.\s+(?!\d)', r'\n\1. ', text)

    # Handle hyphenated lists
    text = re.sub(r'(?:^|\n|\s{2,})[-–]\s+', r'\n- ', text)

    # Add newlines between list items
    list_separator_pattern = (
        r'(?<!\d)(?<=\.)\s+(?=\d(?!\d*\.))|'  # After period, before non-decimal
        r'(?<=\))\s+(?=\d(?!\d*\.))|'  # After parenthesis, before non-decimal
        r'(?<=\.)\s+(?=[-–])|'  # After period, before hyphen
        r'(?<=\))\s+(?=[-–])'  # After parenthesis, before hyphen
    )
    text = re.sub(list_separator_pattern, r'\n\n', text)

    # Handle list items
    text = re.sub(r'(?<!\d)(\d+\.)\s+', r'\1 ', text)  # Fix spacing after lists
    text = re.sub(r'(?<!\n)\s+(?<!\d)(\d+\.)', r'\n\n\1', text)  # Add newlines

    # Clean up excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Ensure proper spacing after headers and before lists
    text = re.sub(r'(###[^\n]+)\n\n', r'\1\n', text)
    text = re.sub(r'\n\n\n+', '\n\n', text)

    # Ensure sections are well-separated
    text = re.sub(r'(?<=\S)\s*\n(?=#+\s)', r'\n\n', text)

    # Add line breaks before key sections
    # Case-insensitive replacements for main headers
    text = re.sub(
        r'(?i)trend analysis\s*:',
        "\n### Trend Analysis\n",
        text
    )
    text = re.sub(
        r'(?i)trading strategies\s*:',
        "\n### Trading Strategies\n",
        text
    )

    # Case-sensitive replacements
    text = text.replace("Analysis:", "\n### Analysis:\n")
    text = text.replace("Analysis :", "\n### Analysis:\n")
    text = text.replace("strategies:", "\n### strategies:\n")
    text = text.replace("strategies :", "\n### strategies:\n")

    # Case-insensitive replacements for strategy sections
    text = re.sub(
        r'(?i)low\s*-\s*risk strategy*:',
        "\n### Low-Risk Strategy to Collect Premiums\n",
        text
    )
    text = re.sub(
        r'(?i)high\s*-\s*risk strategy*:',
        "\n### High-Risk Strategy for a Potential Rapid Move\n",
        text
    )

    # Remove unwanted line breaks not surrounded by words
    text = re.sub(r'(?<!\w)\n(?!\w)', ' ', text)

    # Remove unnecessary spaces around newlines and punctuation
    text = text.strip()

    return text

def plot_lr_channel(data, ticker, period, std_dev=2):
    df = data[data['symbol'] == ticker].query(f"period == {period}")
    x = np.arange(len(df))
    y = df['close'].values

    coeffs = np.polyfit(x, y, 1)
    trendline = np.polyval(coeffs, x)

    residuals = y - trendline
    std = np.std(residuals)
    upper_band = trendline + std_dev * std
    lower_band = trendline - std_dev * std

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['close'], label='Close Price', color='blue')
    ax.plot(df.index, trendline, label='Trendline', color='red')
    ax.fill_between(df.index, lower_band, upper_band, color='blue', alpha=0.2, label='Std Dev Bands')

    ax.set_title(f"{ticker} Price Chart with Linear Regression Channel")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    return fig


