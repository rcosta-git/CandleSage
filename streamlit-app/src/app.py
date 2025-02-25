# Authors: Robert Costa and Shreyas Mahimkar

from utils import *
from PIL import Image
import re
import streamlit as st
import pandas as pd

use_tradingview = False
use_AI_suggestions = False

# Main Streamlit app
def main():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("logo.jpeg", width=200)
    st.title("CandleSage Ticker Analysis", anchor="center")
    # Add description text
    st.markdown(
        """
        In this app, you can input a ticker symbol in the provided input field. 
        The application will process the data, analyze trends, and can suggest
        both low-risk and high-risk trading strategies based on the analysis.

        Developed by Robert Costa and Shreyas Mahimkar at JumboHack 2025, at 
        Tufts University. For more information, visit Rob's GitHub page
        [Financial Mathematics](https://rcosta-git.github.io/), and look at our
        [app](https://github.com/rcosta-git/CandleSage/tree/main/streamlit-app)
        source code on GitHub, licensed under the GNU General Public License.
        """
    )
    st.header("Stock and Cryptocurrency Analysis")
    ticker = st.text_input("Enter symbol:", placeholder="AAPL, BTC, etc")
    period = st.number_input("Enter period (days):", min_value=1, value=90)
    image_path = f"images/chart.png"
    analysis_result = "### AI-generated analysis suggestions are turned off."
    
    if st.button("Analyze"):
        if ticker and use_tradingview:
            exchange = get_exchange(ticker)
            symbol = f"{exchange}:{ticker}"
            url = generate_saved_chart_url(symbol)

            # Persist chart for analysis
            persist_chart_for_analysis(url, image_path)

            # Process the image for analysis
            processed_data = image_to_analysis(image_path)

            # Analyze the data
            if use_AI_suggestions:
                analysis_result = analyze_data(processed_data)
        elif ticker:
            df = fetch_and_plot_data(ticker, image_path, days=period)

            # Calculate student t-distribution statistics
            student_t_dict = calculate_student_t_distribution(ticker, period)
            columns = [
                'symbol', 'days', 'close', 'mean', 
                'std_dev', 'variance', 'cv', 'skewness', 'kurtosis',
                't_70_start', 't_70_end', 't_95_start', 't_95_end', 
                'z_score'
            ]
            statistics_df = pd.DataFrame(
                [student_t_dict], columns=student_t_dict
            )

            # Add a header for the statistics
            st.header("Statistics")
            # Display the DataFrame without the index
            st.dataframe(statistics_df.set_index(statistics_df.columns[0]))

            # Analyze the data directly
            if use_AI_suggestions:
                analysis_result = analyze_data(
                    df.to_markdown(), statistics_df.to_markdown()
                )
        else:
            st.warning("Please enter a ticker symbol.")

        # Display the saved image
        st.image(image_path, caption=f"Chart for {ticker}")

        # Add a header for the analysis result
        st.header("AI Analysis and Trading Recommendation")

        # Add a disclaimer
        st.markdown("""
        **Disclaimer:** All trading involves risk. This analysis is intended 
        for educational and informational purposes only and is not certified 
        financial advice. Past performance is not a guarantee of future returns.
        """)

        # Clean and display the analysis result with proper markdown
        if use_AI_suggestions:
            st.text(clean_text(analysis_result))
        else:
            st.markdown(analysis_result)

if __name__ == "__main__":
    main()
