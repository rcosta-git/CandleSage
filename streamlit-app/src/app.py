# Authors: Robert Costa and Shreyas Mahimkar

from utils import *
from PIL import Image
import re
import streamlit as st
import pandas as pd

use_tradingview = False

# Main Streamlit app
def main():
    st.title("CandleSage Ticker Analysis")

    ticker_symbol = st.text_input("Enter Ticker Symbol:", "")
    image_path = f"images/{ticker_symbol}_chart.png"
    analysis_result = "Nothing to analyze yet."
    
    if st.button("Analyze"):
        if ticker_symbol and use_tradingview:
            exchange = get_exchange(ticker_symbol)
            symbol = f"{exchange}:{ticker_symbol}"
            url = generate_saved_chart_url(symbol)

            # Persist chart for analysis
            persist_chart_for_analysis(url, image_path)

            # Display the saved image
            st.image(image_path, caption=f"Chart for {ticker_symbol}")

            # Process the image for analysis
            processed_data = image_to_analysis(image_path)

            # Analyze the data
            analysis_result = analyze_data(processed_data)
        elif ticker_symbol:
            df = fetch_and_plot_data([ticker_symbol])

            # Display the saved image
            st.image(image_path, caption=f"Chart for {ticker_symbol}")

            # Calculate student t-distribution statistics
            student_t_dict = calculate_student_t_distribution(ticker_symbol)
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
            analysis_result = analyze_data(
                df.to_markdown(), statistics_df.to_markdown()
            )
        else:
            st.warning("Please enter a ticker symbol.")

        # Add a header for the analysis result
        st.header("AI Analysis and Trading Recommendation")

        # Add a disclaimer
        st.markdown("""
        **Disclaimer:** All trading involves risk. This analysis is intended 
        for educational and informational purposes only and is not certified 
        financial advice.
        """)

        # Clean and display the analysis result with proper markdown
        st.text(clean_text(analysis_result))

if __name__ == "__main__":
    main()
