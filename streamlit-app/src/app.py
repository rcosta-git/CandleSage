# Authors: Robert Costa and Shreyas Mahimkar

from utils import *
from PIL import Image
import re
import streamlit as st
import pandas as pd
import os

allow_tradingview = False
allow_AI_suggestions = True

# Main Streamlit app
def main():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        logo_path = os.path.join(os.path.dirname(__file__), "logo.jpeg")
        st.image(logo_path, width=200)
    st.title("CandleSage Ticker Analysis", anchor="center")
    # Add description text
    st.markdown(
        """
        Welcome to CandleSage, a free application initially developed by
        [Robert Costa](https://www.linkedin.com/in/rob-costa-541b5576/),
        [Shreyas Mahimkar](https://www.linkedin.com/in/shreyas-mahimkar/) and
        [Helena Fu](https://www.linkedin.com/in/helena-fu-ml/) at the Tufts
        JumboHack 2025. For more information, visit [Rob's GitHub page on
        Financial Mathematics](https://rcosta-git.github.io/), and our
        [code](https://github.com/rcosta-git/CandleSage/tree/main/streamlit-app)
        for the app on GitHub, licensed under the GNU General Public License.

        In this app, you can input a ticker symbol in the provided input field.
        The application will process the data, analyze trends, and can suggest
        both low-risk and high-risk trading strategies based on the analysis.
        """
    )
    st.header("Stock and Cryptocurrency Analysis")
    
    ticker = st.text_input("Enter symbol:", placeholder="AAPL, BTC-USD, etc.")
    period = st.number_input("Enter period (days):", min_value=1, value=90)

    # Add HMM analysis toggle
    generate_hmm = st.checkbox(
        "Hidden Markov Model",
        value=False,
        help="Enable/disable HMM-based market state analysis"
    )

    # Add TradingView toggle if enabled
    if allow_tradingview:
        use_tradingview = st.checkbox(
            "TradingView 30 minute Candlestick Chart",
            value=False,
            help="Enable/disable TradingView chart with 30 minute candlesticks"
        )
    else:
        use_tradingview = False

    # Add AI suggestions toggle if enabled
    if allow_AI_suggestions:
        generate_ai = st.checkbox(
            "Generate AI Analysis Suggestions",
            value=False,
            help="Enable/disable AI-powered analysis suggestions"
        )
    else:
        generate_ai = False
    
    # Ensure images directory exists
    images_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(images_dir, exist_ok=True)
    image_path = os.path.join(images_dir, "chart.png")
    analysis_result = "### AI-generated analysis suggestions are turned off."
    
    if st.button("Analyze"):
        if ticker:
            df = fetch_and_plot_data(ticker, image_path, days=period)
            if df is None:
                st.error(
                    f"No data available for {ticker}. "
                    "Please check if the symbol is correct."
                )
                return
            
            # Display the saved image
            st.image(image_path, caption=f"Chart for {ticker}")

            # Calculate student t-distribution statistics using the same dataframe
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
                
                # Input for number of hidden states
                n_states = 4
                
                # Generate HMM analysis
                fig, state_df = generate_hmm_analysis(df, n_states)
                
                # Display results
                st.write("### Market States Over Time")
                st.pyplot(fig)
                
                st.write("### State Analysis")
                st.dataframe(state_df)

            if use_tradingview:
                exchange = get_exchange(ticker)
                symbol = f"{exchange}:{ticker}"
                url = generate_saved_chart_url(symbol)

                # Add a header for the TradingView chart
                st.header("Candlestick Chart with 30 minute Candles")

                # Persist chart for analysis
                persist_chart_for_analysis(url, image_path)
                
                # Display the saved image
                st.image(image_path, caption=f"TradingView Chart for {ticker}")

            # Analyze the data directly
            if generate_ai:
                analysis_result = analyze_data(
                    df.to_markdown(), statistics_df.to_markdown(),
                    image_to_analysis(image_path) if use_tradingview else None
                )
        else:
            st.warning("Please enter a ticker symbol.")

        # Add a header for the analysis result
        st.header("AI Analysis and Trading Recommendation")

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
