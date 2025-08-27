# AI-Powered Financial Analysis Agent

## Overview

An intelligent financial analysis tool that provides AI-powered insights into stock performance. This Streamlit application answers natural language queries about stocks, displays key financial metrics, and provides interactive visualizations of price movements using the Gemini 2.0 Flash model and yfinance data.

## Features

- **Natural Language Queries**: Ask questions like "What is the stock price of AMZN?" or "META stock analysis"
- **Comprehensive Financial Metrics**: Current price, PE ratio, market cap, EPS, dividend yield, beta, and 52-week high/low
- **Interactive Visualizations**:
  - Bar chart for 52-week high/low comparison
  - Line chart for 6-month historical price trends
  - Candlestick chart for 1-month price movements
- **Global Stock Support**: Handles international tickers with exchange suffixes (e.g., .NS, .BO, .L)
- **Intelligent Caching**: Optimized performance with 24-hour data caching
- **Robust Error Handling**: Automatic retries for API errors with exponential backoff

## Prerequisites

- Python 3.8+
- Google Cloud account with Gemini API access (free tier available)
- Docker (optional, for containerized deployment)

## Installation

### Method 1: Using Python Virtual Environment

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
   ```

### Method 2: Using Docker

1. **Build the Docker image**
   ```bash
   docker build -t financial-analysis-app .
   ```

2. **Run the container**
   ```bash
   # Create cache file
   touch cache.pkl
   
   # Run container
   docker run -d -p 8501:8501 \
     -e GOOGLE_API_KEY=your_google_api_key_here \
     -v $(pwd)/cache.pkl:/app/cache.pkl \
     --name financial-app \
     financial-analysis-app
   ```

### Method 3: Using Docker Compose (Recommended)

1. **Set up environment file**
   ```bash
   echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
   ```

2. **Start the application**
   ```bash
   docker-compose up -d
   ```

## Usage

1. **Start the application**
   ```bash
   # If using Python directly
   streamlit run main.py
   
   # If using Docker
   # Application will be available at http://localhost:8501
   ```

2. **Ask financial questions**
   - Input questions like "What is the stock price of AMZN?" or "Analyze META stock"
   - Click "Run Query" to generate analysis

3. **Supported Ticker Formats**
   - US stocks: AMZN, META, AAPL
   - International stocks: TCS.NS (NSE), RELIANCE.BO (BSE), VOD.L (LSE)

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
# Optional alternative API keys:
# HUGGINGFACE_API_KEY=your_huggingface_key
# OPENROUTER_API_KEY=your_openrouter_key
```

### API Keys

- **Google Gemini**: Obtain from [Google Cloud Console](https://cloud.google.com/)
- **Hugging Face**: Optional alternative, obtain from [Hugging Face](https://huggingface.co/)
- **OpenRouter**: Optional alternative, obtain from [OpenRouter](https://openrouter.ai/)

## Troubleshooting

### Common Issues

1. **Gemini API Errors (503, 429)**
   - Verify API key in Google Cloud Console
   - Check quota usage in Google Cloud dashboard
   - Application automatically retries 5 times with exponential backoff

2. **Ticker Not Recognized**
   - Verify ticker format and exchange suffix
   - Check validity at [Yahoo Finance](https://finance.yahoo.com/)
   - Delete `cache.pkl` to clear stale data

3. **No Data Available**
   - Confirm ticker is supported by yfinance
   - Try alternative exchange suffixes for international stocks

4. **Docker Deployment Issues**
   - Ensure Docker Desktop is running
   - Check file permissions for cache.pkl
   - Use absolute paths for volume mounts on Windows

### Logs

Application logs are stored in `app.log` for debugging purposes.

## Project Structure

```
.
├── main.py                 # Main application code
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── .env                   # Environment variables (create this)
├── cache.pkl              # Data cache (auto-generated)
├── agents.db              # Agent sessions database (auto-generated)
└── app.log                # Application logs (auto-generated)
```

## API Alternatives

To avoid Gemini API limitations, consider these free alternatives:

- **Hugging Face Inference API**: ~1000 requests/day free tier
- **OpenRouter**: Limited free credits available
- **Cerebras Inference API**: Limited free credits for testing

See the code for implementation examples of alternative LLM providers.

## License

This project utilizes:
- yfinance (Apache 2.0 License)
- Various LLM models (check respective licenses)
- Ensure compliance with all dependent libraries' licenses

## Support

For issues or questions:
1. Check the troubleshooting guide above
2. Review application logs in `app.log`
3. Ensure all dependencies are properly installed
4. Verify API keys are correctly configured

---

*Note: This application is for educational and informational purposes only. Not financial advice. Always conduct your own research before making investment decisions.*