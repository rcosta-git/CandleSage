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

## Saving Cookies

To save cookies for TradingView, follow these steps:

1. Open a Python shell or script.
2. Import the `save_cookies` function from `utils.py`.
3. Run the `save_cookies` function with the appropriate parameters:
    ```python
    from utils import save_cookies
    from apikey import tv_email, tv_password

    url = "https://www.tradingview.com/chart/"
    save_cookies(url, tv_email, tv_password)
    ```

4. Manually solve the captcha when prompted.
5. Press Enter to continue and save the cookies to `cookies.pkl`.

## Features

- Input a ticker symbol for analysis
- Display processed chart images
- Analyze trends and EMAs
- Suggest low-risk and high-risk options trading strategies

## Contributing

Contributions are welcome! If you have suggestions for improvements or new 
features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more 
details.