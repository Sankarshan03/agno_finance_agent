import os
import time
import re
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import logging
import pickle
from datetime import datetime, timedelta
from agno.agent import Agent
from agno.models.google import Gemini
from agno.storage.sqlite import SqliteStorage
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

# Configure logging to capture errors in 'app.log' for debugging
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file (e.g., GOOGLE_API_KEY)
load_dotenv()

# Initialize SQLite storage for agent sessions
storage = SqliteStorage(table_name="agent_sessions", db_file="agents.db")

# Define cache file and duration (24 hours) for storing ticker data
CACHE_FILE = "cache.pkl"
CACHE_DURATION = timedelta(hours=24)

# Initialize the Agent with Gemini model and yfinance tools for financial data
agent = Agent( 
    session_id="financial_analysis_ui",  # Unique session ID for tracking
    user_id="user_ui",  # User identifier
    model=Gemini(  # Configure Gemini LLM for natural language processing
        id="gemini-2.0-flash",
        api_key=os.environ.get("GOOGLE_API_KEY"),
        vertexai=False  # Use non-Vertex AI endpoint
    ),
    storage=storage,  # Persist agent data in SQLite
    tools=[
        YFinanceTools(  # Enable yfinance tools for stock data
            stock_price=True,
            analyst_recommendations=False,  # Disabled due to API limitations
            company_info=True,
            company_news=True,
            historical_prices=True
        )
    ],
    markdown=True  # Enable markdown formatting for responses
)

# Load cached data from file to avoid repeated API calls
def load_cache():
    try:
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        return cache
    except (FileNotFoundError, pickle.PickleError):
        return {}  # Return empty dict if cache file doesn't exist or is invalid

# Save cache to file for persistence
def save_cache(cache):
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        logging.error(f"Failed to save cache: {e}")  # Log cache save errors

# Clear cache entries for a specific ticker to force fresh data fetch
def clear_cache_for_ticker(ticker):
    cache = load_cache()
    for key in list(cache.keys()):
        if ticker in key:
            del cache[key]
    save_cache(cache)

# Extract stock tickers from text using regex
def extract_tickers(text):
    # Match uppercase tickers (1-5 letters, optional exchange suffix)
    pattern = r'\b(?!\$)[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b'
    potential_tickers = re.findall(pattern, text)
    
    # Filter out common non-ticker words
    common_words = {'THE', 'AND', 'FOR', 'YOU', 'ARE', 'THIS', 'THAT', 'WITH', 'HAVE', 'FROM', 'NS'}
    tickers = [ticker for ticker in potential_tickers if ticker.split('.')[0] not in common_words]
    
    return list(set(tickers))  # Return unique tickers

# Resolve ticker symbols, including exchange suffixes for international stocks
def get_proper_ticker_symbol(ticker, max_retries=5, delay=2):
    cache = load_cache()
    cache_key = f"ticker_{ticker}"
    # Check if ticker is cached and not expired
    if cache_key in cache and cache[cache_key]['timestamp'] > datetime.now() - CACHE_DURATION:
        return cache[cache_key]['value']
    
    # Return ticker if it already has a suffix
    if '.' in ticker:
        return ticker
    
    # Try NASDAQ tickers for AMZN and META explicitly
    nasdaq_tickers = {'AMZN': 'AMZN', 'META': 'META'}
    if ticker in nasdaq_tickers:
        for attempt in range(max_retries):
            try:
                info = yf.Ticker(nasdaq_tickers[ticker]).info
                if info.get('regularMarketPrice') is not None:
                    cache[cache_key] = {'value': nasdaq_tickers[ticker], 'timestamp': datetime.now()}
                    save_cache(cache)
                    return nasdaq_tickers[ticker]
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
    
    # Try original ticker without suffix
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info
            if info.get('regularMarketPrice') is not None:
                cache[cache_key] = {'value': ticker, 'timestamp': datetime.now()}
                save_cache(cache)
                return ticker
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
    
    # Try international exchange suffixes (excluding .PA as requested)
    suffixes = ['.NS', '.BO', '.L', '.TO', '.AX', '.DE', '.MI', '.AS']
    for suffix in suffixes:
        full_ticker = ticker + suffix
        for attempt in range(max_retries):
            try:
                info = yf.Ticker(full_ticker).info
                if info.get('regularMarketPrice') is not None:
                    cache[cache_key] = {'value': full_ticker, 'timestamp': datetime.now()}
                    save_cache(cache)
                    return full_ticker
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
    
    # Fallback: return original ticker and clear cache for freshness
    clear_cache_for_ticker(ticker)
    cache[cache_key] = {'value': ticker, 'timestamp': datetime.now()}
    save_cache(cache)
    return ticker

# Validate ticker by checking for price or historical data
def is_valid_ticker(ticker, max_retries=5, delay=2):
    cache = load_cache()
    cache_key = f"valid_{ticker}"
    if cache_key in cache and cache[cache_key]['timestamp'] > datetime.now() - CACHE_DURATION:
        return cache[cache_key]['value']
    
    nasdaq_tickers = {'AMZN': 'AMZN', 'META': 'META'}
    ticker_to_try = nasdaq_tickers.get(ticker, ticker)
    
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker_to_try).info
            if info.get('regularMarketPrice') is not None:
                cache[cache_key] = {'value': True, 'timestamp': datetime.now()}
                save_cache(cache)
                return True
            
            hist = yf.Ticker(ticker_to_try).history(period="1d")
            if not hist.empty:
                cache[cache_key] = {'value': True, 'timestamp': datetime.now()}
                save_cache(cache)
                return True
            
            logging.warning(f"No price or historical data available for {ticker_to_try}")
            cache[cache_key] = {'value': False, 'timestamp': datetime.now()}
            save_cache(cache)
            return False
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Invalid ticker {ticker_to_try}: {str(e)}")
                clear_cache_for_ticker(ticker)
                cache[cache_key] = {'value': False, 'timestamp': datetime.now()}
                save_cache(cache)
                return False

# Run queries with retry logic for Gemini API errors
def run_query_with_retry(query, max_retries=5, delay=2):
    for attempt in range(max_retries):
        try:
            response = agent.run(query)
            return response
        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "Service Unavailable" in error_str:
                st.warning(f"Gemini API unavailable, retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                st.warning(f"Gemini API rate limited, retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Query error: {e}")
                st.error(f"Error: {e}")
                return type('obj', (object,), {
                    'content': f"I encountered an error while processing your request: {str(e)}. Please try a different query or try again later."
                })
    logging.error(f"Failed after {max_retries} attempts")
    return type('obj', (object,), {
        'content': "I'm sorry, but I'm experiencing technical difficulties with the Gemini API. Please try your query again in a few minutes."
    })

# Fetch financial metrics for a given stock ticker
def get_financial_metrics(symbol: str, max_retries=5, delay=2):
    cache = load_cache()
    cache_key = f"metrics_{symbol}"
    if cache_key in cache and cache[cache_key]['timestamp'] > datetime.now() - CACHE_DURATION:
        return cache[cache_key]['value'], cache[cache_key]['display']
    
    nasdaq_tickers = {'AMZN': 'AMZN', 'META': 'META'}
    ticker_to_try = nasdaq_tickers.get(symbol, symbol)
    
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(ticker_to_try)
            info = ticker.info
            
            # Fallbacks for current price
            current_price = info.get('regularMarketPrice')
            if current_price is None:
                current_price = info.get('currentPrice')
            if current_price is None:
                try:
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                except:
                    current_price = None
            
            # Store numeric market cap for calculations
            market_cap = info.get('marketCap', None)
            
            metrics = {
                "symbol": symbol,
                "current_price": current_price,
                "pe_ratio": info.get('trailingPE', None),
                "forward_pe": info.get('forwardPE', None),
                "market_cap": market_cap,
                "eps": info.get('trailingEps', None),
                "dividend_yield": info.get('dividendYield', None),
                "price_to_book": info.get('priceToBook', None),
                "beta": info.get('beta', None),
                "52_week_high": info.get('fiftyTwoWeekHigh', None),
                "52_week_low": info.get('fiftyTwoWeekLow', None)
            }
            
            # Format metrics for display (e.g., '11.42T' for market cap)
            display_metrics = metrics.copy()
            if isinstance(market_cap, (int, float)):
                if market_cap >= 1e12:
                    display_metrics['market_cap'] = f"{market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    display_metrics['market_cap'] = f"{market_cap/1e9:.2f}B"
                elif market_cap >= 1e6:
                    display_metrics['market_cap'] = f"{market_cap/1e6:.2f}M"
                else:
                    display_metrics['market_cap'] = f"{market_cap:,.0f}"
            else:
                display_metrics['market_cap'] = 'N/A'
            
            # Format other fields for display
            for key in ['current_price', 'pe_ratio', 'forward_pe', 'eps', 'dividend_yield', 'price_to_book', 'beta', '52_week_high', '52_week_low']:
                if display_metrics[key] is None:
                    display_metrics[key] = 'N/A'
                elif key == 'dividend_yield' and isinstance(display_metrics[key], (int, float)):
                    display_metrics['market_cap'] = f"{market_cap/1e12:.2f}T"
                elif isinstance(display_metrics[key], (int, float)):
                    display_metrics[key] = f"{display_metrics[key]:.2f}"
            
            cache[cache_key] = {'value': metrics, 'display': display_metrics, 'timestamp': datetime.now()}
            save_cache(cache)
            return metrics, display_metrics
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Failed to get financial metrics for {ticker_to_try}: {e}")
                clear_cache_for_ticker(symbol)
                return {"error": f"Failed to get financial metrics: {e}"}, None

# Streamlit UI setup
st.set_page_config(page_title="AI Financial Analyst", layout="wide")
st.title("📊 AI-powered Financial Analysis Agent")

# Display information about supported stocks and limitations
with st.expander("ℹ️ Note about supported stocks"):
    st.write("""
    This tool works best with US-listed stocks (e.g., AMZN, META). For international stocks, 
    some data might not be available through our data sources.
    
    For international stocks, try using the exchange suffix:
    - `.NS` for NSE (National Stock Exchange of India)
    - `.BO` for BSE (Bombay Stock Exchange)
    - `.L` for London Stock Exchange
    - `.TO` for Toronto Stock Exchange
    - `.AX` for Australian Stock Exchange
    
    Example: "TCS.NS" instead of just "TCS"
    """)

# Input field for user queries
query = st.text_input("Ask a financial question:", placeholder="e.g. What is the stock price of AMZN? or META stock analysis")

if st.button("Run Query") and query:
    with st.spinner("Analyzing..."):
        try:
            # Clean query by removing $ prefix to avoid invalid ticker extraction
            cleaned_query = re.sub(r'\$[A-Z]+', lambda m: m.group(0)[1:], query)
            response = run_query_with_retry(cleaned_query, max_retries=5, delay=2)
            st.subheader("🤖 Agent Response")
            st.markdown(response.content)

            # Extract tickers from query or response
            tickers = extract_tickers(cleaned_query)
            if not tickers:
                tickers = extract_tickers(response.content)
            
            # Limit to 3 tickers to avoid UI overload
            tickers = tickers[:3]
            
            for t in tickers:
                st.markdown("---")
                
                # Resolve proper ticker symbol
                proper_ticker = get_proper_ticker_symbol(t)
                
                # Validate ticker
                if not is_valid_ticker(proper_ticker):
                    st.warning(f"⚠️ No data available for {proper_ticker}. This ticker may be delisted, not supported, or the data source is temporarily unavailable.")
                    st.info(f"Try searching for a different stock (e.g., AMZN, META) or use a supported exchange suffix like .NS for Indian stocks. You can also try again later.")
                    continue
                
                st.subheader(f"📈 {proper_ticker} Financial Metrics")
                
                try:
                    metrics, display_metrics = get_financial_metrics(proper_ticker)
                    
                    if isinstance(metrics, dict) and "error" in metrics:
                        st.warning(f"Could not fetch data for {proper_ticker}: {metrics['error']}")
                        continue
                        
                    # Create numeric DataFrame for internal use
                    display_data = {k: v for k, v in metrics.items() if k != 'symbol'}
                    df = pd.DataFrame([display_data]).T.rename(columns={0: "Value"})
                    
                    # Create formatted DataFrame for display
                    display_data_formatted = {k: v for k, v in display_metrics.items() if k != 'symbol'}
                    df_formatted = pd.DataFrame([display_data_formatted]).T.rename(columns={0: "Value"})
                    
                    # Force string type to avoid Arrow serialization errors
                    df_formatted['Value'] = df_formatted['Value'].astype(str)
                    
                    # Display DataFrame with stretch width
                    st.dataframe(df_formatted, width='stretch')

                    # Chart 1: 52-week high/low bar chart
                    if (metrics.get("52_week_high") is not None and 
                        metrics.get("52_week_low") is not None and
                        metrics["52_week_high"] != metrics["52_week_low"]):
                        try:
                            fig, ax = plt.subplots()
                            ax.bar(["52W Low", "52W High"], 
                                   [metrics["52_week_low"], metrics["52_week_high"]], 
                                   color=["red", "green"])
                            ax.set_title(f"{proper_ticker} - 52 Week Range")
                            st.pyplot(fig)
                        except Exception as e:
                            logging.error(f"Error plotting 52-week chart for {proper_ticker}: {e}")
                            pass

                    # Chart 2: 6-month historical price line chart
                    try:
                        cache_key = f"hist_{proper_ticker}_6mo"
                        cache = load_cache()
                        if cache_key in cache and cache[cache_key]['timestamp'] > datetime.now() - CACHE_DURATION:
                            hist = cache[cache_key]['value']
                        else:
                            hist = yf.Ticker(proper_ticker).history(period="6mo")
                            cache[cache_key] = {'value': hist, 'timestamp': datetime.now()}
                            save_cache(cache)
                        if not hist.empty:
                            st.subheader(f"📉 {proper_ticker} - Last 6 Months Price")
                            st.line_chart(hist["Close"])
                        else:
                            st.info(f"No historical price data available for {proper_ticker}")
                    except Exception as e:
                        logging.error(f"Error plotting historical chart for {proper_ticker}: {e}")
                        st.warning(f"Could not fetch historical prices for {proper_ticker}: {str(e)}")

                    # Chart 3: 1-month candlestick chart
                    try:
                        cache_key = f"hist_{proper_ticker}_1mo"
                        cache = load_cache()
                        if cache_key in cache and cache[cache_key]['timestamp'] > datetime.now() - CACHE_DURATION:
                            hist = cache[cache_key]['value']
                        else:
                            hist = yf.Ticker(proper_ticker).history(period="1mo", interval="1d")
                            cache[cache_key] = {'value': hist, 'timestamp': datetime.now()}
                            save_cache(cache)
                        if not hist.empty and len(hist) > 5:
                            st.subheader(f"🕯️ {proper_ticker} - 1 Month Candlestick Chart")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            mpf.plot(hist, type="candle", style="charles", ax=ax, volume=False)
                            st.pyplot(fig)
                        else:
                            st.info(f"Insufficient data for candlestick chart for {proper_ticker}")
                    except Exception as e:
                        logging.error(f"Error plotting candlestick chart for {proper_ticker}: {e}")
                        pass
                        
                except Exception as e:
                    logging.error(f"Error processing data for {proper_ticker}: {e}")
                    st.warning(f"Error processing data for {proper_ticker}: {str(e)}")
        
        except Exception as e:
            logging.error(f"General error in query processing: {e}")
            st.error(f"❌ Error: {e}")