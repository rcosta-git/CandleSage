# Authors: Robert Costa and Shreyas Mahimkar

from utils import *
from hmm import *
from prophet_model import *
import streamlit as st
import pandas as pd
import os

allow_AI_suggestions = st.secrets.get("AI_suggestions", False)

# Main Streamlit app
def main():
    # Add copyright header
    st.markdown(
        """<div style='text-align: left; font-size: 0.8em;'>
        © 2025 <a href='https://costacapital.org/' target='_blank'>Costa Capital
        Technologies</a>. All rights reserved.
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
    # Add description text
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
    
    interval = st.selectbox(
        "Select data interval:",
        options=list(interval_options.keys()),
        format_func=lambda x: interval_options[x],
        index=6  # Default to "1d"
    )

    # Add note about interval limitations
    if interval in ["1m", "2m", "5m", "15m", "30m", "60m"]:
        st.info(
            "Note: Intraday data (minute/hour intervals) is limited to 7 days for US equities and 60 days for other markets by Yahoo Finance API. "
            "Period has been automatically limited based on the selected interval."
        )

    default_periods = {
        "1m": 7,
        "2m": 7,
        "5m": 7,
        "15m": 7,
        "30m": 7,
        "60m": 7, 
        "1d": 90,
        "5d": 365,
        "1wk": 365 * 2,  
        "1mo": 365 * 5,  
        "3mo": 365 * 5,  
    }
    
    # Define maximum period based on interval
    max_periods = {
        "1m": 7,
        "2m": 7,
        "5m": 7,
        "15m": 7,
        "30m": 7,
        "60m": 7, 
        "1d": 365 * 2,
        "5d": 365 * 5,
        "1wk": 365 * 10,  
        "1mo": 365 * 20,  
        "3mo": 365 * 30,  
    }
    
    # Set default and max period based on interval
    default_period = default_periods.get(interval, 90)
    max_period = max_periods.get(interval, 365)
    
    period = st.number_input(
        "Enter period (days):", 
        min_value=1, 
        max_value=max_period,
        value=default_period
    )

    # Add Select All checkbox
    select_all = st.checkbox("Select All", value=False)

    # Add checkboxes for each feature
    generate_hmm = st.checkbox(
        "Hidden Markov Model",
        value=select_all,
        help="Enable/disable HMM-based market state analysis"
    )
    generate_lr = st.checkbox(
        "Linear Regression Channel",
        value=select_all,
        help="Enable/disable Linear Regression Channel analysis"
    )
    generate_prophet = st.checkbox(
        "Prophet Model Forecast",
        value=select_all,
        help="Enable/disable Prophet Model forecast"
    )
    generate_ai = st.checkbox(
        "Generate AI Options Trading Suggestions",
        value=select_all and allow_AI_suggestions,
        disabled=not allow_AI_suggestions,
        help="Enable/disable AI-powered trading suggestions"
    )
    use_tradingview = False # Too many things that go wrong with this

    if generate_hmm:
        n_states = st.number_input("Number of HMM Model Hidden States",
                                   min_value=1, max_value=10, value=3)

    if use_tradingview:
        tv_interval = st.number_input("Enter TradingView interval (minutes):",
                                      min_value=1, max_value=1440, value=30)

    # Ensure images directory exists
    images_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(images_dir, exist_ok=True)
    image_path = os.path.join(images_dir, "chart.png")
    analysis_result = "### AI-generated trading suggestions are turned off."

    if st.button("Analyze"):
        if ticker:
            st.session_state.analyze_clicked = True
            st.session_state.ticker = ticker
            st.session_state.period = period
            st.session_state.interval = interval
            st.session_state.generate_hmm = generate_hmm
            st.session_state.generate_lr = generate_lr
            st.session_state.generate_prophet = generate_prophet
            st.session_state.use_tradingview = use_tradingview
            st.session_state.generate_ai = generate_ai

    if st.session_state.get("analyze_clicked", False):
        ticker = st.session_state.ticker
        period = st.session_state.period
        interval = st.session_state.get('interval', interval)
        generate_hmm = st.session_state.generate_hmm
        generate_lr = st.session_state.generate_lr
        generate_prophet = st.session_state.generate_prophet
        use_tradingview = st.session_state.use_tradingview
        generate_ai = st.session_state.generate_ai

        df = fetch_and_plot_data(ticker, image_path, days=period, interval=interval)
        if df is None:
            st.error(
                f"No data available for {ticker}. "
                "Please check if the symbol is correct."
            )
            return
            
        # Add 'period' and 'symbol' columns to the DataFrame
        df['period'] = period
        df['symbol'] = ticker

        # Display the saved image
        st.header("Price Chart with Exponential Moving Averages")
        st.image(image_path, caption=f"Chart for {ticker} ({interval} interval, {period} days)")

        # Calculate student t-distribution statistics using the same df
        student_t_dict = calculate_student_t_distribution(ticker, df=df)
        if 'error' in student_t_dict:
            st.error(student_t_dict['error'])
            return
        columns = [
            'symbol', 'days', 'close', 'mean', 
            'std_dev', 'variance', 'cv', 'skewness', 'kurtosis',
            't_70_start', 't_70_end', 't_95_start', 't_95_end', 
            'z_score'
        ]
        statistics_df = pd.DataFrame([student_t_dict])

        # Add a header for the statistics
        st.header("Statistics")
        # Display the DataFrame without the index
        st.dataframe(statistics_df.set_index(statistics_df.columns[0]))
        
        # Generate HMM analysis if requested
        if generate_hmm:
            st.header("Hidden Markov Model Analysis")

            df = smart_data_fetch(ticker, image_path, target_days=period, interval=interval)
            # Generate HMM analysis
            fig, state_df, trans_mat = generate_hmm_analysis(df, interval=interval, n_states=n_states)
            
            # Display results
            st.write("### Market States Over Time")
            st.pyplot(fig)
            
            st.write("### State Analysis")
            st.dataframe(state_df)
            
            st.write("### Transition Matrix")
            st.dataframe(trans_mat)

        # Linear Regression Channel
        if generate_lr:
            st.header('Linear Regression Channel')
            std_dev = st.slider('Standard Deviation', 1.0, 3.0, 2.0, .1,
                                key='std_dev')
            lr_fig = plot_lr_channel(df, ticker, period, std_dev)
            st.pyplot(lr_fig)

        if generate_prophet:
            st.header('Prophet Model Forecast')
            
            # Prepare data for Prophet Model
            prophet_df = prepare_data_for_prophet(df, ticker, period)
            
            # Run Prophet Model
            forecast, forecast_fig, components_fig = run_prophet_model(
                prophet_df, ticker
            )
            
            # Display the forecast plot
            st.subheader('Forecast')
            
            # Explain the forecast plot
            st.markdown('''
            The Prophet forecast plot shows:
            - Historical data points (black dots)
            - Predicted values (blue line)
            - Uncertainty intervals (light blue shaded area)
            
            Prophet is a forecasting tool developed by Facebook. It is designed to 
            handle time series data with daily observations that display patterns 
            on different time scales. Prophet uses an additive model to fit 
            non-linear trends with yearly, weekly, and daily seasonality, plus 
            holiday effects. It works best with time series that have strong 
            seasonal effects and several seasons of historical data.
            
            Due to the built-in Fourier cyclical nature, Prophet may show 
            oscillating behavior if there isn't enough data. This is because the 
            model tries to fit the seasonal components even when the data is 
            insufficient, leading to overfitting and oscillations in the forecast.
            
            The uncertainty intervals give you an idea of how confident the model 
            is in its predictions. Wider intervals indicate more uncertainty.
            ''')
            
            st.pyplot(forecast_fig)
            
            # Display the components plot
            st.subheader('Forecast Components')
            
            # Explain the components plot
            st.markdown('''
            The components plot shows the individual components that make up the 
            forecast. These components are additive, meaning that they are added 
            together to form the final forecast. The components include the trend, 
            seasonality, and holidays. The values shown in the components plot are 
            not actual prices, but rather the contribution of each component to 
            the final forecast.
            ''')
            
            st.pyplot(components_fig)

        if use_tradingview:
            exchange = get_exchange(ticker)
            symbol = f"{exchange}%3A{ticker}"
            url = generate_saved_chart_url(symbol, tv_interval)

            # Add a header for the TradingView chart
            st.header("Candlestick Chart")

            # Persist chart for analysis
            persist_chart_for_analysis(url, image_path)
            
            # Display the saved image
            st.image(image_path, caption=f"TradingView Chart for {ticker}")


        # Analyze the data directly
        if generate_ai:
            analysis_result = analyze_data(
                df.to_markdown(), statistics_df.to_markdown()
            )
        # Add a header for the analysis result
        st.header("AI Analysis and Options Trading Recommendation")

        # Add a disclaimer
        st.markdown("""
        **Disclaimer:** All trading involves risk. This analysis is intended 
        for educational and informational purposes only and is not certified 
        financial advice. Past performance is not a guarantee of future returns.
        """)

        # Clean and display the analysis result with proper markdown
        if generate_ai:
            st.text(clean_text(analysis_result))
        else:
            st.markdown(analysis_result)

if __name__ == "__main__":
    main()
