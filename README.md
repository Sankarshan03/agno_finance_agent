# AI-Powered Financial Analysis Agent

## Overview

An intelligent Streamlit application that provides AI-powered financial analysis for stocks. The tool answers natural language queries about stocks, displays key financial metrics, and provides interactive visualizations of price movements using the Gemini 2.0 Flash model and yfinance data.

## Features

- **Natural Language Queries**: Ask questions like "What is the stock price of AMZN?" or "META stock analysis"
- **Comprehensive Financial Metrics**: Current price, PE ratio, market cap, EPS, dividend yield, beta, and 52-week high/low
- **Interactive Visualizations**:
  - Bar chart for 52-week high/low comparison
  - Line chart for 6-month historical price trends
  - Candlestick chart for 1-month price movements
- **Robust Error Handling**: Automatic retries for API errors with exponential backoff
- **Global Stock Support**: Handles international tickers with exchange suffixes (e.g., .NS, .BO, .L)
- **Intelligent Caching**: Optimized performance with 24-hour data caching

## Prerequisites

- Python 3.8+
- Google Cloud account with Gemini API access (free tier available)
- Optional: API keys for alternative LLMs (Hugging Face, OpenRouter) for free usage

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**
   ```bash
   pip install streamlit yfinance pandas matplotlib mplfinance python-dotenv
   ```
   
   Note: The `agno` library (for Agent, Gemini, YFinanceTools, SqliteStorage) is required. Ensure it's installed in your environment.

3. **Environment Setup**
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your-google-api-key
   # Optional alternative API keys:
   # HUGGINGFACE_API_KEY=your-huggingface-key
   # OPENROUTER_API_KEY=your-openrouter-key
   ```

4. **Obtain API Keys**
   - Google Gemini: Get from [Google Cloud Console](https://cloud.google.com/)
   - Optional alternatives: [Hugging Face](https://huggingface.co/), [OpenRouter](https://openrouter.ai/)

## Usage

1. **Launch the Application**
   ```bash
   streamlit run main.py
   ```
   The app will open in your browser at `http://localhost:8501`.

2. **Enter Your Queries**
   - Input financial questions in natural language
   - Examples: "What is the stock price of AMZN?", "Analyze META stock"
   - Click "Run Query" to generate analysis

3. **Supported Ticker Formats**
   - US stocks: AMZN, META, AAPL
   - International stocks: TCS.NS (NSE), RELIANCE.BO (BSE), VOD.L (LSE)

## Troubleshooting

### Common Issues & Solutions

**Gemini API Errors (503, 429)**
- Verify API key in Google Cloud Console
- Check quota usage in Google Cloud dashboard
- Application automatically retries 5 times with exponential backoff

**DataFrame Serialization Errors**
- Update dependencies: `pip install --upgrade streamlit pyarrow`
- Ensure proper data type conversion in code

**Ticker Not Recognized**
- Verify ticker format and exchange suffix
- Check validity at [Yahoo Finance](https://finance.yahoo.com/)
- Delete `cache.pkl` to clear stale data: `rm cache.pkl`

**No Data Available**
- Confirm ticker is supported by yfinance
- Try alternative exchange suffixes for international stocks

## Alternative LLM Options

To avoid Gemini API limitations, consider these free alternatives:

### Hugging Face Inference API
- Models: Mistral 7B, LLaMA 2
- Free tier: ~1000 requests/day
- Setup: Sign up at [Hugging Face](https://huggingface.co/)

### OpenRouter
- Models: LLaMA 3, Mistral Nemo
- Free tier: Limited credits
- Setup: Register at [OpenRouter](https://openrouter.ai/)

### Implementation Example (Hugging Face)
```python
from huggingface_hub import InferenceClient

class HuggingFaceModel:
    def __init__(self, api_key, model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.client = InferenceClient(api_key=api_key, model=model_id)
    
    def run(self, query):
        try:
            response = self.client.text_generation(
                prompt=query, 
                max_new_tokens=500
            )
            return type('obj', (object,), {'content': response})
        except Exception as e:
            return type('obj', (object,), {'content': f"Error: {str(e)}"})

# Update agent initialization
agent = Agent(
    session_id="financial_analysis_ui",
    user_id="user_ui",
    model=HuggingFaceModel(api_key=os.environ.get("HUGGINGFACE_API_KEY")),
    storage=storage,
    tools=[YFinanceTools(...)],
    markdown=True
)
```

## Performance Notes

- **Gemini 2.0 Flash**: Optimized for financial queries with best performance
- **Free Alternatives**: May have lower performance for complex analysis
- **Rate Limits**: Free tiers have usage restrictions; consider paid options for heavy usage
- **Caching**: Data cached for 24 hours; delete `cache.pkl` to force refresh

## License

This project utilizes:
- yfinance (Apache 2.0 License)
- Various LLM models (check respective licenses)
- Ensure compliance with all dependent libraries' licenses

## Support

For issues or feature requests:
1. Check the troubleshooting guide above
2. Review application logs in `app.log`
3. Open a GitHub issue with detailed error information
4. Contact maintainers for critical issues

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with comprehensive tests
4. Ensure code follows existing style and patterns

---

*Note: This application is for educational and informational purposes only. Not financial advice. Always conduct your own research before making investment decisions.*
