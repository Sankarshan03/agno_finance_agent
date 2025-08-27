AI-Powered Financial Analysis Agent
Overview
This Streamlit application provides an AI-powered financial analysis tool that answers user queries about stocks (e.g., AMZN, META) and displays financial metrics, 52-week price ranges, 6-month historical price charts, and 1-month candlestick charts. It uses the Gemini 2.0 Flash model for natural language processing and yfinance for stock data, with caching to optimize performance.
Features

Natural Language Queries: Ask questions like "What is the stock price of AMZN?" or "META stock analysis".
Financial Metrics: Displays metrics like current price, PE ratio, market cap, EPS, dividend yield, beta, and 52-week high/low.
Visualizations:
Bar chart for 52-week high/low.
Line chart for 6-month historical prices.
Candlestick chart for 1-month price movements.


Robust Error Handling: Retries for Gemini API errors (e.g., 503, 429) and yfinance failures, with cache clearing for stale data.
International Stock Support: Handles tickers with exchange suffixes (e.g., .NS, .BO, .L).

Prerequisites

Python 3.8+
A Google Cloud account with an API key for Gemini (free tier available).
Optional: API keys for alternative LLMs (e.g., Hugging Face, OpenRouter) for free usage.

Installation

Clone the Repository:
git clone <repository-url>
cd <repository-directory>


Install Dependencies:
pip install streamlit yfinance pandas matplotlib mplfinance python-dotenv

Note: The agno library (for Agent, Gemini, YFinanceTools, SqliteStorage) is assumed to be proprietary or custom. Ensure it is installed or available in your environment.

Set Up Environment Variables:Create a .env file in the project root:
GOOGLE_API_KEY=your-google-api-key

Obtain a Google API key from Google Cloud Console.

Clear Cache (Optional):Delete cache.pkl to ensure fresh data for tickers like AMZN and META.


Usage

Run the Application:
streamlit run main.py

The app will open in your browser at http://localhost:8501.

Enter Queries:

Input financial questions (e.g., "What is the stock price of AMZN?" or "Analyze META stock").
Click "Run Query" to see the AI response, financial metrics, and charts.


Supported Tickers:

Best for US stocks (e.g., AMZN, META).
For international stocks, use suffixes like .NS (e.g., TCS.NS for Tata Consultancy Services).



Troubleshooting

Gemini API Errors (503, 429):
Check your API key and quota in Google Cloud Console.
Ensure GOOGLE_API_KEY is set correctly in .env.
The app retries 5 times with exponential backoff. If errors persist, contact Google Cloud support or try an alternative LLM (see below).


DataFrame Serialization Errors:
If errors like "Could not convert '11.42T' to double" appear, verify that df_formatted['Value'] = df_formatted['Value'].astype(str) is in the code.
Update Streamlit and pyarrow to the latest versions: pip install --upgrade streamlit pyarrow.


Ticker Not Recognized:
Check app.log for errors (e.g., "symbol may be delisted").
Test the ticker at https://finance.yahoo.com/quote/<TICKER> (e.g., AMZN, META).
Delete cache.pkl to clear stale data.


No Data Available:
Ensure the ticker is valid and supported by yfinance.
Try adding an exchange suffix (e.g., TCS.NS instead of TCS).



Alternative Free LLM APIs
To avoid Gemini API issues (e.g., 503 errors, quota limits), consider these free LLM APIs:

Hugging Face Serverless Inference API:
Models: Mistral 7B, LLaMA 2.
Free tier: ~1000 requests/day.
Setup: Sign up at https://huggingface.co/, get an API key, and modify the Agent to use huggingface_hub.InferenceClient.


OpenRouter:
Models: LLaMA 3, Mistral Nemo.
Free tier: Limited credits (~$1 worth of requests).
Setup: Register at https://openrouter.ai/, get an API key, and adapt the Agent for OpenRouter's chat completion API.


Cerebras Inference API:
Models: LLaMA 3.1, Qwen.
Free tier: Limited credits for testing.
Setup: Sign up at https://cerebras.ai/, get an API key, and update the Agent for Cerebras' API.



To integrate an alternative LLM, create a custom model wrapper for the Agent class. Example for Hugging Face:
from huggingface_hub import InferenceClient
class HuggingFaceModel:
    def __init__(self, api_key, model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.client = InferenceClient(api_key=api_key, model=model_id)
    def run(self, query):
        try:
            response = self.client.text_generation(prompt=query, max_new_tokens=500)
            return type('obj', (object,), {'content': response})
        except Exception as e:
            return type('obj', (object,), {'content': f"Error: {str(e)}"})
agent = Agent(
    session_id="financial_analysis_ui",
    user_id="user_ui",
    model=HuggingFaceModel(api_key=os.environ.get("HUGGINGFACE_API_KEY")),
    storage=storage,
    tools=[YFinanceTools(...)],
    markdown=True
)

Notes

Performance: Gemini 2.0 Flash is optimized for financial queries, but free alternatives like Mistral 7B may have lower performance for complex analysis.
Rate Limits: Free LLM APIs have strict limits. For production, consider paid tiers (e.g., Hugging Face Pro, SuperGrok at https://x.ai/grok).
Caching: The app caches data for 24 hours to reduce API calls. Delete cache.pkl to refresh data if tickers fail.
Dependencies: Ensure agno library is available. If proprietary, contact the provider for documentation.

License
This project is unlicensed. Ensure compliance with the licenses of yfinance (Apache 2.0) and any LLM models used (e.g., Apache 2.0 for Mistral, custom for LLaMA).
Contact
For issues or feature requests, open a GitHub issue or contact the maintainer.