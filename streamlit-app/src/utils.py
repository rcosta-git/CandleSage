# Authors: Robert Costa and Shreyas Mahimkar

import os
import shutil
import pickle
import re
import time

from PIL import Image
import pytesseract

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

from openai import OpenAI
import yfinance as yf
import streamlit as st

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import mplfinance as mpf

def calculate_student_t_distribution(symbol, df=None):
    """Calculate comprehensive volatility metrics for securities"""
    if df is None:
        asset = yf.Ticker(ticker=symbol)
        hist = asset.history(period=f"{period}d", interval="1d")

        # Data cleaning and preparation
        hist = hist.ffill().bfill()
        closes = hist['Close']
    else:
        # Use the provided dataframe
        hist = df
        closes = hist['Close']

    # Check if we have any data
    if len(hist) == 0 or len(closes) == 0:
        return {
            'symbol': symbol,
            'period': 0,
            'error': f'No data available for {symbol}'
        }

    mean_price = closes.mean()
    # Get latest closing price
    latest_close = closes.iloc[-1]  # Last entry in the closes series

    # IQR calculations
    Q1 = closes.quantile(0.25)
    Q3 = closes.quantile(0.75)
    IQR = Q3 - Q1
    
    # Volatility metrics
    std_dev = closes.std()
    variance = closes.var()
    
    # Distribution shape metrics
    cv = std_dev / mean_price  # Coefficient of Variation
    skewness = float(stats.skew(closes))  # Skewness, convert to float
    kurtosis = float(stats.kurtosis(closes))  # Kurtosis, convert to float
    z_score = (latest_close - mean_price) / std_dev
    
    # Student's t-distribution parameters
    df_param = len(closes) - 1
    t_value_70 = stats.t.ppf(0.85, df_param)
    t_value_90 = stats.t.ppf(0.975, df_param)
    
    # Price ranges
    t_70_start = mean_price - t_value_70 * std_dev
    t_70_end = mean_price + t_value_70 * std_dev
    t_95_start = mean_price - t_value_90 * std_dev
    t_95_end = mean_price + t_value_90 * std_dev

    return {
        'symbol': symbol,
        'close' : latest_close,
        'period': len(closes),
        'mean': round(mean_price, 2),
        'median': round(closes.median(), 2),
        'Q_start': round(Q1 - 1.5*IQR, 2),
        'Q1': round(Q1, 2),
        'Q3': round(Q3, 2),
        'Q_end': round(Q3 + 1.5*IQR, 2),
        'std_dev': round(std_dev, 2),
        'variance': round(variance, 2),
        'cv': round(cv, 4),
        'skewness': round(skewness, 4),
        'kurtosis': round(kurtosis, 4),
        't_70_start': round(t_70_start, 2),
        't_70_end': round(t_70_end, 2),
        't_95_start': round(t_95_start, 2),
        't_95_end': round(t_95_end, 2),
        'z_score': z_score
    }

# Function to clean and format text for markdown rendering
def clean_text(text):
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
        r'(?<=\))\s+(?=[-–])'   # After parenthesis, before hyphen
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


def smart_data_fetch(ticker, image_path, target_days, interval):
    # warm up data for short period
    if target_days > 365:
        df = fetch_and_plot_data(ticker, image_path, days=target_days, interval=interval)
    else:
        df_large = fetch_and_plot_data(ticker, image_path, days=365, interval=interval)
        end_date = df_large.index.max()
        start_date = end_date - pd.Timedelta(days=target_days)
        df = df_large[df_large.index >= start_date].copy()

    return df

def fetch_and_plot_data(symbol, img_path, days=330, interval="1d", ema_periods=[20, 50, 100]):
    print(f"\nFetching data for {symbol} with interval {interval} for {days} days...")
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=days)

    # Fetch data with specified interval
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    print(f"Downloaded data with interval {interval}, shape:", data.shape)

    # Check if we have any data
    if len(data) == 0 or 'Close' not in data.columns or len(data['Close']) == 0:
        plt.figure(figsize=(12, 6))
        plt.text(
            0.5, 0.5,
            f'No closing price data available for {symbol} with interval {interval}',
            horizontalalignment='center',
            verticalalignment='center',
            transform=plt.gca().transAxes
        )
        plt.savefig(img_path)
        plt.close()
        return None

    # Calculate EMAs and additional metrics
    print("\nCalculating metrics...")
    data['Returns'] = data['Close'].diff()
    data.columns = data.columns.get_level_values(0)  # Reset column index

    # Adjust EMA periods based on interval for more meaningful analysis
    if interval in ["1m", "2m", "5m", "15m", "30m", "60m"]:
        if len(data) < 100: 
            ema_periods = [5, 10, 20]
        else:
            ema_periods = [8, 21, 55]

    print("After adding metrics, columns:", data.columns)
    for period in ema_periods:
        data[f'{period} EMA'] = (
            data['Close']
            .ewm(span=period, adjust=False)
            .mean()
        )

    # Create a candlestick chart if mplfinance is available and we have OHLC data
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):

        # Prepare EMA data for mplfinance
        ema_data = []
        ema_colors = ['orange', 'green', 'red', 'purple']
        ema_styles = []
        
        ema_styles.append(mpf.make_addplot(
            data['Close'],
            color='blue',
            width=1.5,
            label='Close Price'
        ))

        for i, period in enumerate(ema_periods):
            ema_data.append(data[f'{period} EMA'])
            ema_styles.append(mpf.make_addplot(
                data[f'{period} EMA'], 
                color=ema_colors[i % len(ema_colors)],
                width=1
            ))
        
        # Set up style for the chart
        mc = mpf.make_marketcolors(
            up='green', down='red',
            wick={'up':'green', 'down':'red'},
            edge={'up':'green', 'down':'red'},
            volume={'up':'green', 'down':'red'}
        )

        s = mpf.make_mpf_style(
            marketcolors=mc, 
            gridstyle='--', 
            y_on_right=True,
            facecolor='white'
        )
        
        # Create the candlestick chart
        fig, axes = mpf.plot(
            data, 
            type='candle', 
            style=s,
            addplot=ema_styles,
            title=f'Close Price & EMAs',
            ylabel='Price (USD)',
            figsize=(12, 6),
            returnfig=True
        )
        
        # Add legend for EMAs
        ax = axes[0]
        lines = ax.get_lines()
        ema_lines = lines[-len(ema_periods):]       
        legend_labels = [f'{period} EMA' for period in ema_periods]
        ax.legend(ema_lines, legend_labels, loc='lower left')
        
        # Save the figure
        plt.savefig(img_path)
        plt.close(fig)
    
    # Return the data with all calculated metrics
    print("\nReturning data with shape:", data.shape)
    print("Final columns:", data.columns)
    return data

def save_cookies(url, username, password):
    # Set up WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")  # Run in background
    options.add_argument("--start-maximized")  # Open window maximized
    options.add_argument("--disable-notifications")  # Disable notifications
    
    # Clear cached ChromeDriver files
    cache_dir = os.path.expanduser('~/.wdm')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    driver = webdriver.Chrome(
        service=Service(
                ChromeDriverManager().install()
            ), options=options
    )
    print("Using ChromeDriver at:", ChromeDriverManager().install())

    # Open the chart URL
    driver.get(url)

    # Wait for the chart to load
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'canvas[data-name="pane-canvas"]')
        )
    )

    # Click on the sign-in button
    sign_in_button = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, '.button-U2jIw4km > svg:nth-child(1)')
        )
    )
    sign_in_button.click()

    # Click on the email sign-in option
    email_sign_in_option = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, 'div.joinItem-U2jIw4km:nth-child(13)')
        )
    )
    email_sign_in_option.click()

    # Click on the Email button
    email_button = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, '.emailButton-nKAw8Hvt')
        )
    )
    email_button.click()

    # Enter credentials
    email_input = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#id_username"))
    )
    password_input = driver.find_element(By.CSS_SELECTOR, "#id_password")

    email_input.send_keys(username)
    password_input.send_keys(password)

    # Click the Sign In button using the correct CSS selector
    sign_in_button = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, '.submitButton-LQwxK8Bm')
        )
    )
    sign_in_button.click()

    # Wait for manual captcha solving (uncomment if needed)
    # print("Please solve the captcha manually and press Enter to continue...")
    # input()

    # Save cookies to a file
    with open("cookies.pkl", "wb") as file:
        pickle.dump(driver.get_cookies(), file)

    driver.quit()

def persist_chart_for_analysis(url, image_path):
    # Ensure the directory for saving the image exists
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    # Check if cookies file exists
    if not os.path.exists("cookies.pkl"):
        print("Cookies file not found. Running save_cookies to create it.")
        save_cookies(url, st.secrets["tv_email"], st.secrets["tv_password"])

    # Set up WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")  # Run in background
    options.add_argument("--start-maximized")  # Open window maximized
    options.add_argument("--disable-notifications")  # Disable notifications
    
    # Clear cached ChromeDriver files
    cache_dir = os.path.expanduser('~/.wdm')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    driver = webdriver.Chrome(
        service=Service(
                ChromeDriverManager().install()
            ), options=options
    )
    print("Using ChromeDriver at:", ChromeDriverManager().install())

    # Open the chart URL
    driver.get(url)

    # Load cookies from the file
    with open("cookies.pkl", "rb") as file:
        cookies = pickle.load(file)
        for cookie in cookies:
            driver.add_cookie(cookie)

    # Refresh the page to apply cookies
    driver.refresh()

    # Wait for the chart to load
    print("Waiting for the chart canvas to load...")
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'canvas[data-name="pane-canvas"]')
        )
    )
    
    # Add additional wait time for indicators to load
    print("Waiting for indicators to load...")
    time.sleep(3)  # Give time for indicators to start appearing
    
    # Check for loading spinners and wait for them to disappear
    spinners = driver.find_elements(By.CSS_SELECTOR, '.loading-indicator')
    if spinners:
        try:
            WebDriverWait(driver, 10).until(
                EC.invisibility_of_element(
                    (By.CSS_SELECTOR, '.loading-indicator')
                )
            )
        except:
            print("Warning: Loading indicators still present after timeout")
    
    # Wait for the specific indicator elements to load
    print("Waiting for the first indicator element to load...")
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'div.sources-l31H9iuA > div:nth-child(6)')
        )
    )
    print("Waiting for the Volume indicator to load...")
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'div.valueValue-l31H9iuA[title="Volume"]')
        )
    )
    print("Waiting for the EMA indicator to load...")
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'div.valueValue-l31H9iuA[title="EMA"]')
        )
    )

    # Take full-page screenshot
    print("Taking screenshot...")
    driver.save_screenshot(image_path)
    driver.quit()
    print(f"Screenshot saved to {image_path}")

def image_to_analysis(image_path):
    # Use pytesseract to extract text from the image
    image = Image.open(image_path)
    processed_data = pytesseract.image_to_string(image)
    return processed_data

def analyze_data(processed_data, statistics="empty"):
    messages = [{
        "role": "user",
        "content": f"""Analyze this data, describing the trends and EMAs and 
        suggest a low-risk options trading strategy to profit from collecting 
        premiums, as well as a high-risk one to profit from a rapid move if it 
        seems possible, or no trade if that is the best option for either case.
        
        ## Processed Data:
        {processed_data}

        ## Statistics: this has 95% confidence interval of the student's t-
        distribution, mean and standard deviation. You can talk about it, if
        it is not empty:
        {statistics}
        """
    }]
    return AIopinion(messages=messages)

def AIopinion(messages):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"], 
                    base_url="https://api.perplexity.ai")
    
    try:
        response_stream = client.chat.completions.create(
            model="sonar-pro",  # sonar-pro #sonar-reasoning-pro #sonar 
            # sonar-reasoning
            messages=messages,
            stream=True,
        )
    
        full_response = []
        link_map = {}
        link_counter = 1
        buffer = ""
    
        for chunk in response_stream:
            if hasattr(chunk, 'citations') and chunk.citations:
                # Enumerate URLs starting at index 1 for proper citation nums
                link_map = {str(i): url for i, url in enumerate(chunk.citations,
                                                                start=1)}
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
                
                # Append processed content and clear buffer
                full_response.append(buffer)
                buffer = ""
    
        # Append remaining buffer content
        if buffer:
            full_response.append(buffer)
    
        # Add references appendix
        full_response.append("\n\n--- References ---")
        for url, num in sorted(link_map.items(), key=lambda x: x[1]):
            full_response.append(f"[{num}] {url}")
    
        return "\n".join(full_response)
    
    except Exception as e:
        return f"Error occurred: {str(e)}"

def generate_saved_chart_url(symbol, interval="1d"):
    chart_id = "CvaXuplE"  # Your saved TradingView chart ID
    
    # Strip out dashes and convert '=F' to '%21'
    symbol = symbol.replace("-", "").replace("=F", "1%21")
    
    return (f"https://www.tradingview.com/chart/{chart_id}/?symbol={symbol}"
            f"&interval={interval}")

def get_exchange(ticker: str) -> str:
    """
    Returns the exchange where the given ticker is listed.
    
    :param ticker: Stock or asset ticker symbol (e.g., "AAPL", "ES=F").
    :return: The exchange name as a string.
    """
    exchange_map = {
        "NMS": "NASDAQ",  # Convert Yahoo's "NMS" to "NASDAQ"
        "NYQ": "NYSE",
        "ASE": "AMEX",
        "NYM": "NYMEX",
        "CBT": "CBOT",
        "CME": "CME_MINI",
    }
    
    try:
        # Check for hyphen in the ticker
        if "-" in ticker:
            exchange = "BITSTAMP"
        else:
            stock = yf.Ticker(ticker)
            exchange = stock.info.get("exchange", "BITSTAMP")
        
        print("Exchange: ", exchange)
        return exchange_map.get(exchange, exchange)  # Convert if in map
    except Exception as e:
        return f"Error: {e}"

def plot_lr_channel(data, ticker, period, std_dev=2):
    # Use the provided DataFrame directly without filtering
    # The period parameter is kept for API compatibility but not used for filtering
    df = data.copy()  # Make a copy to avoid modifying the original
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
