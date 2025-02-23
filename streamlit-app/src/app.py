# Authors: Robert Costa and Shreyas Mahimkar

from utils import (
    persist_chart_for_analysis,
    image_to_analysis,
    analyze_data,
    get_exchange,
    generate_saved_chart_url
)
from PIL import Image
import re
import streamlit as st

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

    # Add line breaks before key sections
    text = text.replace("Analysis:", "\n### Analysis:\n")
    text = text.replace(
        "Low-risk strategy to collect premiums:",
        "\n### Low-risk strategy to collect premiums:\n"
    )
    text = text.replace(
        "High-risk strategy for a potential rapid move:",
        "\n### High-risk strategy for a potential rapid move:\n"
    )

    # Remove unnecessary spaces around newlines and punctuation
    text = text.strip()

    return text

# Main Streamlit app
def main():
    st.title("CandleSage Ticker Analysis")

    ticker_symbol = st.text_input("Enter Ticker Symbol:", "")
    
    if st.button("Analyze"):
        if ticker_symbol:
            exchange = get_exchange(ticker_symbol)
            symbol = f"{exchange}:{ticker_symbol}"
            url = generate_saved_chart_url(symbol)
            image_path = f"images/{ticker_symbol}_chart.png"

            # Persist chart for analysis
            persist_chart_for_analysis(url, image_path)

            # Display the saved image
            st.image(image_path, caption=f"Chart for {ticker_symbol}")

            # Process the image for analysis
            processed_data = image_to_analysis(image_path)

            # Analyze the data
            analysis_result = analyze_data(processed_data)
            
            # Clean and display the analysis result with proper markdown
            st.text(clean_text(analysis_result))
        else:
            st.warning("Please enter a ticker symbol.")

if __name__ == "__main__":
    main()
