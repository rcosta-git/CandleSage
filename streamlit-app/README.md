# Streamlit Ticker Analysis Application

This project is a Streamlit web application that allows users to input a ticker 
symbol and perform data analysis on it. The application utilizes various 
functions to process chart images, analyze financial data, and suggest trading 
strategies based on the analysis.

## Authors

- Robert Costa
- Shreyas Mahimkar

## Project Structure

```
streamlit-app
├── src
│   ├── app.py          # Main entry point of the Streamlit application
│   ├── utils.py        # Utility functions for data analysis and processing
│   └── types
│       └── index.py    # Custom types and data structures
├── requirements.txt    # List of dependencies for the project
└── README.md           # Documentation for the project
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit application, execute the following command in your terminal:
```
cd src
streamlit run app.py
```

Once the application is running, you can input a ticker symbol in the provided 
input field. The application will process the data, analyze trends, and suggest 
both low-risk and high-risk trading strategies based on the analysis.

## Using TradingView and Saving Cookies (necessary for premium indicators)

To save cookies for TradingView, follow these steps:

1. Update your TradingView username and password into a `secrets.toml` file.
2. Set the `use_tradingview` flag to `True` in the `app.py` file.
3. Rerun the application and enter a ticker.
4. Manually solve the captcha when prompted.
5. Press Enter to continue and save the cookies to `cookies.pkl`.

## Using AI-generated trading suggestions

To enable AI-suggested trading, follow these steps:
1. Update your OpenAI API key into a `secrets.toml` file.
2. Set the `use_AI_suggestions` flag to `True` in the `app.py` file.
3. Rerun the application and enter a ticker.

## Features

- Input a ticker symbol for analysis
- Display processed chart images
- Analyze trends and EMAs
- Suggest low-risk and high-risk options trading strategies (with AI turned on)

## Contributing

Contributions are welcome! If you have suggestions for improvements or new 
features, please open an issue or submit a pull request.

## License

This project is licensed under the GNU General Public License v3.0. See the
LICENSE file for more details.
