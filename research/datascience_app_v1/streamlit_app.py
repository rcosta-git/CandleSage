import streamlit as st
from hack_triage import (
    fetch_market_data,
    calculate_statistics,
    get_fear_greed_index,
    plot_emas_streamlit,
    analyze_data,
    plot_lr_channel,
    clean_text)
from prophet_model import prepare_data_for_prophet, run_prophet_model

# Configuration
st.set_page_config(page_title="Market Analysis", layout="wide")

# Initialize session state
if "forecast_output" not in st.session_state:
    st.session_state.forecast_output = None
if "analysis_output" not in st.session_state:
    st.session_state.analysis_output = None

# Sidebar
st.sidebar.title("Analysis Parameters")

# Default Inputs for Symbols and Periods
default_symbols = ['BTC', 'ETH']
default_periods = [15, 30, 90,]

selected_symbols = st.sidebar.multiselect(
    'Select Symbols',
    options=['BTC', 'ETH', 'VOO', 'TQQQ', 'MARA'],
    default=default_symbols
)

selected_periods = st.sidebar.multiselect(
    'Select Periods (Days)',
    options=[15, 30, 90, 180, 365],
    default=default_periods
)

# Linear Regression Channel Parameters
st.sidebar.subheader("Linear Regression Channel")
lr_symbol = st.sidebar.selectbox("Select Symbol for LR Channel", selected_symbols)
lr_period = st.sidebar.selectbox("Select Period for LR Channel", selected_periods)
std_dev = st.sidebar.slider("Standard Deviation", 1.0, 3.0, 2.0, 0.1)

# Main content
st.title('Finance Analysis Dashboard')

# Fetch and calculate data based on selected inputs
symbols = {symbol: selected_periods for symbol in selected_symbols}
market_data = fetch_market_data(symbols)
results = calculate_statistics(market_data)

# Display statistics
st.header('Market Statistics')
st.dataframe(results)

# Fear and Greed Index
st.header("Fear and Greed Index")
fgi_df = get_fear_greed_index(limit=5)
st.dataframe(fgi_df)

# EMAs
st.header("20-50-100 EMAs")
ema_df = plot_emas_streamlit(market_data)

# Linear Regression Channel
st.header("Linear Regression Channel")

lr_fig = plot_lr_channel(market_data, lr_symbol, lr_period, std_dev)
st.pyplot(lr_fig)

# Forecast section
st.header("Price Forecast")
selected_symbol = st.selectbox('Select Symbol for Forecast', selected_symbols)
lookback_days = st.slider('Lookback Days', 30, 365, 90)
forecast_days = st.slider('Forecast Days', 7, 90, 30)

if st.button('Run Forecast'):
    prophet_data = prepare_data_for_prophet(market_data, selected_symbol, lookback_days)
    forecast, fig1, fig2 = run_prophet_model(prophet_data, selected_symbol, forecast_days)
    st.session_state.forecast_output = {
        "symbol": selected_symbol,
        "forecast": forecast,
        "fig1": fig1,
        "fig2": fig2,
    }

if st.session_state.forecast_output:
    output = st.session_state.forecast_output
    st.subheader(f'{output["symbol"]} Price Forecast')
    st.pyplot(output["fig1"])
    st.subheader(f'{output["symbol"]} Forecast Components')
    st.pyplot(output["fig2"])
    st.subheader('Forecast Data')
    st.dataframe(output["forecast"][['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# AI Opinion Section
st.header("AI Opinion")
st.caption("Disclaimer: These AI recommendations are subjective and should be taken with caution.")
user_input = st.text_area("Enter your question or topic for AI analysis:", height=100)

if st.button("Analyze"):
    analysis_text = analyze_data(stats_df=results, fear_greed_df=fgi_df, emas_df=ema_df, user_input=user_input)
    st.session_state.analysis_output = clean_text(analysis_text)

if st.session_state.analysis_output:
    st.write(st.session_state.analysis_output)

# # Add custom CSS for scrollable right column
# st.markdown("""
#     <style>
#     .stApp [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
#         height: calc(100vh - 100px);
#         overflow-y: auto;
#     }
#     </style>
# """, unsafe_allow_html=True)


# Add custom CSS for better styling
st.markdown("""
    <style>
    .stApp [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
            height: calc(100vh - 100px);
            overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)
