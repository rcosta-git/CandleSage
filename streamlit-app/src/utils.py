# Authors: Robert Costa and Shreyas Mahimkar

import os
import pickle
import re
import requests

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
from bs4 import BeautifulSoup

def save_cookies(url, username, password):
    # Set up WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in background
    options.add_argument("--start-maximized")  # Open window maximized
    options.add_argument("--disable-notifications")  # Disable notifications
    driver = webdriver.Chrome(
        service=Service(
                ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
            ), options=options
    )

    # Open the chart URL
    driver.get(url)

    # Wait for the chart to load
#     WebDriverWait(driver, 30).until(
#         EC.presence_of_element_located(
#             (By.CSS_SELECTOR, 'canvas[data-name="pane-canvas"]')
#         )
#     )

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

    # Wait for manual captcha solving
    #print("Please solve the captcha manually and press Enter to continue...")
    #input()

    # Save cookies to a file
    with open("cookies.pkl", "wb") as file:
        pickle.dump(driver.get_cookies(), file)

    driver.quit()

def persist_chart_for_analysis(url, image_path):
    # Ensure the directory for saving the image exists
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    # Check if cookies file exists
#     if not os.path.exists("cookies.pkl"):
#         print("Cookies file not found. Running save_cookies to create it.")
#         save_cookies(url, st.secrets["tv_email"], st.secrets["tv_password"])

    # Set up WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in background
    options.add_argument("--start-maximized")  # Open window maximized
    options.add_argument("--disable-notifications")  # Disable notifications
    driver = webdriver.Chrome(
        service=Service(
                ChromeDriverManager(
                    #chrome_type=ChromeType.CHROMIUM
                    ).install()
            ), options=options
    )

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
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'canvas[data-name="pane-canvas"]')
        )
    )
    
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

def analyze_data(processed_data):
    messages = [{
        "role": "user",
        "content": f"""Analyze this data, describing the trends and EMAs and 
        suggest a low-risk options trading strategy to profit from collecting 
        premiums, as well as a high-risk one to profit from a rapid move if it 
        seems possible, or no trade if that is the best option for either case.
        
        ## 
        {processed_data}
        """
    }]
    return AIopinion(messages=messages)

def AIopinion(messages):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"], base_url="https://api.perplexity.ai")
    
    try:
        response_stream = client.chat.completions.create(
            model="sonar-pro",  # sonar-pro #sonar-reasoning-pro #sonar #sonar-reasoning
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

def generate_saved_chart_url(symbol):
    chart_id = "CvaXuplE"  # Your saved TradingView chart ID
    return f"https://www.tradingview.com/chart/{chart_id}/?symbol={symbol}"

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
        "CME": "Chicago Mercantile Exchange",
        "NYM": "NYMEX",
        "CBT": "CBOT",
        "ICE": "ICE Futures",
    }
    
    try:
        stock = yf.Ticker(ticker)
        exchange = stock.info.get("exchange", "Unknown")
        return exchange_map.get(exchange, exchange)  # Convert if in map
    except Exception as e:
        return f"Error: {e}"
