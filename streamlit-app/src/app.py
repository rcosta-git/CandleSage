# Authors: Robert Costa and Shreyas Mahimkar

from utils import *
from PIL import Image
import re
import streamlit as st

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
            print(df)

            # Display the saved image
            st.image(image_path, caption=f"Chart for {ticker_symbol}")

            # Analyze the data directly
            analysis_result = analyze_data(df.to_markdown())            
        else:
            st.warning("Please enter a ticker symbol.")

        # Clean and display the analysis result with proper markdown
        st.text(clean_text(analysis_result))

if __name__ == "__main__":
    main()
