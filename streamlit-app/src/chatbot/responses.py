import random
import os
import sys
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Add the current directory to path so we can import our modules
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.warning("Groq package not installed. Please install it with: pip install groq")

# System prompt for the AI assistant
SYSTEM_PROMPT = """You are CandleSage, an AI assistant specializing in market analysis, trading strategies, and financial insights. 
Your role is to:
1. Provide accurate and helpful responses about trading and market analysis
2. Explain technical indicators and chart patterns
3. Help users understand market trends and trading strategies
4. Be concise but informative in your responses
5. When discussing specific stocks or cryptocurrencies, always include appropriate disclaimers about investment risks

Remember to:
- Be professional and clear in your explanations
- Use technical terms appropriately but explain them when needed
- Provide balanced perspectives on market analysis
- Always include risk disclaimers when discussing specific investments
"""

#random response placeholder
# Basic market analysis phrases
ANALYSIS_PHRASES = [
    "Based on technical indicators, the market appears to be {tre       nd}.",
    "The {timeframe} chart suggests a potential {pattern} pattern forming.",
    "Consider the recent {indicator} crossover as a signal for possible {direction} movement.",
    "Market sentiment appears to be {sentiment} based on recent volume and price action."
]

# Trend types
TREND_TYPES = ["bullish", "bearish", "consolidating", "showing volatility", "in a sideways pattern"]

# Chart patterns
CHART_PATTERNS = [
    "head and shoulders", "double top", "double bottom", "cup and handle", 
    "ascending triangle", "descending triangle", "symmetrical triangle",
    "flag", "pennant", "wedge"
]

# Technical indicators
INDICATORS = [
    "MACD", "RSI", "moving average", "Bollinger Band", "stochastic", 
    "OBV", "ADX", "Ichimoku Cloud", "ATR"
]

# Market directions
DIRECTIONS = ["upward", "downward", "breakout", "breakdown", "reversal"]

# Market sentiment
SENTIMENTS = ["bullish", "bearish", "neutral", "mixed", "cautiously optimistic", "increasingly pessimistic"]

# Timeframes
TIMEFRAMES = ["daily", "hourly", "weekly", "monthly", "intraday", "long-term"]

def generate_market_analysis():
    """Generate a random market analysis"""
    analysis = random.choice(ANALYSIS_PHRASES)
    return analysis.format(
        trend=random.choice(TREND_TYPES),
        pattern=random.choice(CHART_PATTERNS),
        indicator=random.choice(INDICATORS),
        direction=random.choice(DIRECTIONS),
        sentiment=random.choice(SENTIMENTS),
        timeframe=random.choice(TIMEFRAMES)
    )

def get_groq_response(query):
    """Get a response from Groq API"""
    if not GROQ_AVAILABLE:
        st.warning("Groq API is not available. Using fallback responses.")
        return None
    
    try:
        # Try to get API key from environment variables
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            # Try to get from Streamlit secrets
            try:
                api_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                pass
            
            if not api_key:
                st.warning("Groq API key not found. Please set GROQ_API_KEY in your environment variables or Streamlit secrets.")
                return None
        
        client = Groq(api_key=api_key)
        
        # Get chat history from session state
        chat_history = st.session_state.get("chat_history", [])
        
        # Prepare messages for the API
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add chat history if available
        for msg in chat_history[-4:]:  # Keep last 4 messages for context
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add the current query
        messages.append({"role": "user", "content": query})
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling Groq API: {str(e)}")
        return None

def get_chatbot_response(query):
    """Generate a response based on the user's query"""
    # Try to get a response from Groq first
    groq_response = get_groq_response(query)
    if groq_response:
        return groq_response
    
    # Fall back to rule-based responses if Groq is not available
    query_lower = query.lower()
    
    # Basic responses to common queries
    if any(greeting in query_lower for greeting in ["hello", "hi", "hey", "greetings"]):
        return "Hello! How can I help you with market analysis today?"
    
    elif any(word in query_lower for word in ["bye", "goodbye", "see you"]):
        return "Goodbye! Feel free to return when you need more trading insights."
    
    elif any(word in query_lower for word in ["thanks", "thank you"]):
        return "You're welcome! Is there anything else you'd like to know?"
    
    # Technical analysis queries
    elif any(pattern in query_lower for pattern in ["pattern", "chart pattern", "technical"]):
        return """
        Chart patterns are specific formations that can indicate potential price movements.
        Common patterns include:
        - Head and Shoulders (reversal pattern)
        - Double Tops/Bottoms (reversal pattern)
        - Triangles (continuation or reversal)
        - Flags and Pennants (continuation patterns)
        
        Would you like me to explain any specific pattern in detail?
        """
    
    elif any(word in query_lower for word in ["hmm", "hidden markov", "markov"]):
        return """
        The Hidden Markov Model (HMM) in CandleSage helps identify different market states.
        
        The model works by:
        1. Analyzing price movements to identify hidden states (bullish, bearish, etc.)
        2. Calculating transition probabilities between these states
        3. Helping predict potential state changes
        
        This can be valuable for understanding the current market regime.
        """
    
    elif any(word in query_lower for word in ["prophet", "forecast"]):
        return """
        Prophet is a time series forecasting tool developed by Facebook that CandleSage uses for price predictions.
        
        Key features:
        - Decomposes data into trend, seasonality, and holidays
        - Handles missing data and outliers effectively
        - Provides prediction intervals to gauge uncertainty
        
        The forecast should be used as one of many tools in your trading decisions.
        """
    
    elif "help" in query_lower:
        return """
        I can help you with:
        - Understanding chart patterns and technical indicators
        - Explaining CandleSage's features (Prophet forecasting, HMM analysis)
        - Providing general market analysis concepts
        - Discussing trading strategies
        
        What would you like to learn more about?
        """
    
    else:
        # Generate a random market analysis for other queries
        return generate_market_analysis() + "\n\nIs there anything specific about this analysis you'd like me to explain?"