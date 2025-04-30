# Authors: Robert Costa and Shreyas Mahimkar

from utils import *
from hmm import *
from prophet_model import *
import streamlit as st
import pandas as pd
import os
import sys
import hashlib
import time
import webbrowser
import threading

# Add the current directory to path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our custom modules
from auth.db import init_database, verify_user, register_user
from chatbot.responses import get_chatbot_response

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "page" not in st.session_state:
    st.session_state.page = "Market Analysis"

try:
    allow_AI_suggestions = st.secrets.get("AI_suggestions", False) 
except Exception:
    allow_AI_suggestions = False  # Default value when secrets are not available

# Initialize database
if "db_initialized" not in st.session_state:
    if init_database():
        st.session_state.db_initialized = True
    else:
        # For development, we continue even if DB initialization fails
        st.session_state.db_initialized = "warning"

# Login page
def show_login_page():
    # Add copyright header
    st.markdown(
        """<div style='text-align: left; font-size: 0.8em;'>
        © 2025 Acadia Analytics. All rights reserved.
        </div>""", 
        unsafe_allow_html=True
    )
    
    # Add extra space after copyright
    st.markdown("\n")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        logo_path = os.path.join(os.path.dirname(__file__), "logo.jpeg")
        if os.path.exists(logo_path):
            st.image(logo_path, width=200)
    
    st.title("CandleSage Login")
    
    # Tabs for login and registration
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    # Login tab
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    success, user_data = verify_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.email = user_data.get("email", "")
                        st.success("Login successful!")
                        time.sleep(0.5)  # Brief pause for user feedback
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
    
    # Register tab
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email (optional)")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_submitted = st.form_submit_button("Register")
            
            if register_submitted:
                if not new_username or not new_password:
                    st.error("Username and password are required")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    success, message = register_user(new_username, new_email, new_password)
                    if success:
                        st.success(f"Registration successful! You can now login.")
                    else:
                        st.error(message)
    
    # Demo credentials help
    with st.expander("Need demo credentials?"):
        st.info("""
        **For demonstration purposes:**
        - Username: demo
        - Password: demo123
        
        Login to access all features including the AI chatbot assistant.
        """)
    
    # Warning if database initialization failed
    if st.session_state.get("db_initialized") == "warning":
        st.warning("""
        Database connection warning. The app is running in demo mode with limited functionality.
        MySQL database might not be properly configured.
        """)

# Chatbot interface
def show_chatbot():
    st.title("CandleSage AI Assistant")
    st.write(f"Welcome, {st.session_state.username}!")
    
    st.markdown("""
    Ask questions about market analysis, trading strategies, or how to use the CandleSage platform.
    This AI assistant can help with interpreting chart patterns, understanding indicators, and more.
    """)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask me about trading or market analysis...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process the message and generate a response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = get_chatbot_response(user_input)
                    if response:
                        st.write(response)
                    else:
                        st.warning("Unable to get a response from the AI assistant. Please try again later.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Using fallback response...")
                    response = "I apologize, but I'm having trouble processing your request right now. Please try rephrasing your question or try again later."
                    st.write(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Market Analysis page
def show_market_analysis():
    # Add copyright header
    st.markdown(
        """<div style='text-align: left; font-size: 0.8em;'>
        © 2025 Acadia Analytics. All rights reserved.
        </div>""", 
        unsafe_allow_html=True
    )
    
    # Add extra space after copyright
    st.markdown("\n")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        logo_path = os.path.join(os.path.dirname(__file__), "logo.jpeg")
        st.image(logo_path, width=200)
    st.title("CandleSage Ticker Analysis", anchor="center")
    
    # Rest of your existing application code
    st.markdown(
        """
        This demo of our trading system was first developed by [Robert Costa](
        https://www.linkedin.com/in/rob-costa-541b5576/) and [Shreyas Mahimkar](
        https://www.linkedin.com/in/shreyas-mahimkar/) at the Tufts University
        JumboHack 2025. For more background information, visit [Rob's page](
        https://rcosta-git.github.io/) on GitHub, or take a look at our [code](
        https://github.com/rcosta-git/CandleSage/tree/main/streamlit-app).
        Also supported by [Helena Fu](https://www.linkedin.com/in/helena-fu-ml/)
        and [Qiwen Zeng](https://www.linkedin.com/in/qiwen-zeng/)'s contribution
        to [Hidden Markov models](https://www.youtube.com/watch?v=kqSzLo9fenk).

        In this app, you can input a ticker symbol in the provided input field.
        The application will process the data, analyze trends, and can suggest
        both low-risk and high-risk trading strategies based on the analysis.
        """
    )
    
    st.header("Stock and Cryptocurrency Analysis")
    
    ticker = st.text_input("Enter symbol, add -USD for crypto, =F for futures:",
                           placeholder="AAPL, BTC-USD, ES=F, SPY if left blank")
    if not ticker:
        ticker = "SPY"  # Default to SPY if no ticker is entered
    
    # Continue with your existing code
    # Add interval selection
    interval_options = {
        "1m": "1 Minute", 
        "2m": "2 Minutes",
        "5m": "5 Minutes",
        "15m": "15 Minutes",
        "30m": "30 Minutes",
        "60m": "1 Hour",
        "1d": "1 Day",
        "5d": "5 Days",
        "1wk": "1 Week",
        "1mo": "1 Month",
        "3mo": "3 Months"
    }
    
    # ... Rest of your existing code follows here ...
    # The existing functionality for your app remains unchanged

# Main Streamlit app
def main():
    # Check if user is authenticated
    if not st.session_state.authenticated:
        show_login_page()
        return
    
    # Add sidebar with navigation and logout
    with st.sidebar:
        st.write(f"### Welcome, {st.session_state.username}!")
        
        # Add logout button at the bottom of sidebar
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.chat_history = []
            st.rerun()
    
    # Create tabs for navigation
    tab1, tab2 = st.tabs(["Market Analysis", "AI Assistant"])
    
    with tab1:
        show_market_analysis()
    
    with tab2:
        show_chatbot()

if __name__ == "__main__":
    main()