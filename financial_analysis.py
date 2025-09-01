import streamlit as st
import tempfile
import os
import lancedb
from agno.agent import Agent
from agno.media import File
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.url import UrlKnowledge
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.reranker.sentence_transformer import SentenceTransformerReranker
from agno.storage.sqlite import SqliteStorage as SqliteAgentStorage
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up the Streamlit page
st.set_page_config(
    page_title="Financial Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Test SentenceTransformerReranker connection
def test_reranker_connection():
    try:
        reranker = SentenceTransformerReranker(model="BAAI/bge-reranker-v2-m3")
        test_text = [{"content": "Test document"}]  # Format as list of dicts
        reranker.rerank("Test query", test_text)
        st.success("✅ Successfully initialized SentenceTransformerReranker")
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize SentenceTransformerReranker: {str(e)}")
        return False

# Initialize the Gemini model
def initialize_model():
    try:
        return Gemini(
            id="gemini-2.0-flash",  # Stable model
            api_key=os.getenv("GEMINI_API_KEY"),  # Replace with a valid key
            search=True,
            include_thoughts=True
        )
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {str(e)}")
        return None

# Create the financial analysis agent
def create_agent():
    model = initialize_model()
    if not model:
        return None
# Create a storage backend using the Sqlite database
    storage = SqliteAgentStorage(
        # store sessions in the ai.sessions table
        table_name="agent_sessions",
        # db_file: Sqlite database file
        db_file="tmp/data.db",
    )
    return Agent(
        model=model,
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                key_financial_ratios=True,
                historical_prices=True
            )
        ],
        markdown=True,
        search_knowledge=True,
        storage=storage,
        add_history_to_messages=True,
        description="Financial analyst agent specializing in PDF report analysis and real-time market data.",
        instructions=[
            "Use yfinance tools to augment PDF analysis with real-time stock data",
            "Compare PDF financial metrics with current market conditions",
            "Format responses with tables and markdown for clarity",
            "Provide investment recommendations based on the analysis",
            "Answer any financial questions the user might have",
            "Always search the knowledge base before answering and include sources",
            "Use the SentenceTransformerReranker to prioritize the most relevant documents"
        ],
        debug_mode=True
    )

# Process uploaded files
def process_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return None

# Create and load knowledge base
def create_knowledge_base(pdf_urls, pdf_paths, website_urls):
    try:
        # Define embedder and reranker
        embedder = SentenceTransformerEmbedder(id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        reranker = SentenceTransformerReranker(model="BAAI/bge-reranker-v2-m3")

        # Check if tantivy-py is installed for hybrid search
        try:
            import tantivy
            search_type = SearchType.vector
            st.info("Using vector search with tantivy-py.")
        except ImportError:
            st.warning("tantivy-py not installed. Falling back to vector search. Install with 'pip install tantivy-py' for hybrid search.")
            search_type = SearchType.vector

        # Initialize LanceDb
        vector_db = LanceDb(
            table_name="financial_docs",
            uri="C:\\Temp\\financial_db",
            search_type=search_type,
            embedder=embedder
        )

        # Check if table exists
        db = lancedb.connect("C:\\Temp\\financial_db")
        table_exists = "financial_docs" in db.table_names()

        knowledge_bases = []

        # Handle PDFs (URLs and uploaded files)
        pdf_files = []
        if pdf_urls:
            pdf_files.extend(pdf_urls)
        if pdf_paths:
            pdf_files.extend(pdf_paths)
        if pdf_files:
            kb_pdf = PDFUrlKnowledgeBase(
                urls=pdf_files,
                vector_db=vector_db,
                embedder=embedder,
                reranker=reranker,
                chunk_size=512,
                chunk_overlap=50
            )
            knowledge_bases.append(kb_pdf)

        # Handle websites
        if website_urls:
            kb_website = UrlKnowledge(
                urls=website_urls,
                vector_db=vector_db,
                embedder=embedder,
                reranker=reranker,
                chunk_size=512,
                chunk_overlap=50
            )
            knowledge_bases.append(kb_website)

        if not knowledge_bases:
            st.warning("No valid URLs or files provided for knowledge base.")
            return None

        # Combine knowledge bases
        if len(knowledge_bases) > 1:
            combined_kb = CombinedKnowledgeBase(sources=knowledge_bases)
        else:
            combined_kb = knowledge_bases[0]

        # Load embeddings only if necessary
        if not table_exists:
            st.info("Creating new LanceDB table 'financial_docs' and generating embeddings...")
            combined_kb.load(recreate=True)
            st.success("✅ Knowledge base created and embeddings loaded!")
        else:
            st.info("Reusing existing LanceDB table 'financial_docs'...")
            combined_kb.load(recreate=False)  # Reuse existing table
            st.success("✅ Knowledge base loaded from existing table!")

        return combined_kb
    except Exception as e:
        st.error(f"Error creating knowledge base: {str(e)}")
        return None

# Inspect LanceDB table (for debugging)
def inspect_lancedb_table():
    try:
        db = lancedb.connect("C:\\Temp\\financial_db")
        if "financial_docs" in db.table_names():
            table = db.open_table("financial_docs")
            st.write("**LanceDB Table Contents**")
            st.write(f"Schema: {table.schema}")
            st.write(f"Number of records: {len(table)}")
            st.write("Sample data (first 5 records):")
            st.write(table.to_pandas().head())
        else:
            st.warning("No 'financial_docs' table found in LanceDB.")
    except Exception as e:
        st.error(f"Error inspecting LanceDB table: {str(e)}")

# Display chat history
def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Main application
def main():
    st.title("📊 Financial Analysis Agent with RAG")
    st.markdown("Analyze financial documents with enhanced RAG using SentenceTransformerReranker and real-time market insights")

    # Test reranker connection
    if not test_reranker_connection():
        return

    # Initialize agent
    if "agent" not in st.session_state:
        st.session_state.agent = create_agent()
        if not st.session_state.agent:
            st.error("Agent initialization failed. Check API key and try again.")
            return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    display_chat_history()

    # Sidebar for inputs
    with st.sidebar:
        st.header("Input Options")
        pdf_url = st.text_input(
            "Enter PDF URL:",
            value="https://www.bseindia.com/xml-data/corpfiling/AttachHis/468b09a3-a212-4066-bbaa-4b0ba524d2ce.pdf",
            help="URL of a financial PDF document to analyze"
        )
        uploaded_file = st.file_uploader(
            "Or upload a PDF file:",
            type=["pdf"],
            help="Upload a financial document for analysis"
        )
        ticker = st.text_input(
            "Stock Ticker Symbol:",
            value="RELIANCE.NS",
            help="Enter the stock ticker symbol (e.g., AAPL, MSFT, RELIANCE.NS)"
        )
        website_url = st.text_input(
            "Website URL (optional):",
            help="Enter a website URL to include in the analysis"
        )
        
        st.subheader("Quick Analysis Tasks")
        task_options = {
            "Comprehensive Analysis": f"Analyze the attached financial documents for {ticker}. Extract key financial metrics, perform ratio analysis, and provide investment recommendations using RAG.",
            "Financial Metrics": f"Extract and analyze key financial metrics from the attached documents for {ticker}. Focus on profitability, liquidity, leverage, and efficiency ratios using RAG.",
            "Investment Recommendation": f"Based on the attached financial documents and current market data for {ticker}, provide a detailed investment recommendation with price targets and risk assessment using RAG.",
            "Risk Assessment": f"Perform a comprehensive risk assessment for {ticker} based on the attached documents. Identify financial, operational, and market risks with mitigation strategies using RAG.",
            "Custom Task": "Enter your own task below"
        }
        
        selected_task = st.selectbox("Select a task:", list(task_options.keys()))
        custom_task = ""
        if selected_task == "Custom Task":
            custom_task = st.text_area(
                "Enter your custom task:",
                height=100,
                help="Describe what you want the agent to do with the provided documents and data"
            )
        
        analyze_btn = st.button("Execute Task", type="primary")
        inspect_btn = st.button("Inspect LanceDB Table", type="secondary")

    # Inspect LanceDB table if requested
    if inspect_btn:
        inspect_lancedb_table()

    # Handle ad-hoc questions
    if prompt := st.chat_input("Or ask a question about the data..."):
        pdf_urls = [pdf_url] if pdf_url else []
        pdf_paths = [process_uploaded_file(uploaded_file)] if uploaded_file else []
        website_urls = [website_url] if website_url else []

        knowledge_base = create_knowledge_base(pdf_urls, pdf_paths, website_urls)
        if knowledge_base:
            st.session_state.agent.knowledge = knowledge_base
            st.session_state.agent.search_knowledge = True

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            files = []
            if pdf_url:
                files.append(File(url=pdf_url))
            if uploaded_file:
                file_path = process_uploaded_file(uploaded_file)
                if file_path:
                    files.append(File(path=file_path))
            if website_url:
                files.append(File(url=website_url))
            
            with st.spinner("Processing your request..."):
                try:
                    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
                    def run_agent_with_retry(prompt, files):
                        return st.session_state.agent.run(prompt, files=files)
                    
                    response = run_agent_with_retry(prompt, files)
                    full_response = response.content
                    if hasattr(response, 'thoughts') and response.thoughts:
                        full_response += "\n\n---\n\n**Agent Thoughts:**\n\n" + str(response.thoughts)
                except Exception as e:
                    full_response = f"An error occurred: {str(e)}. If it's a 503 error, the model may be overloaded. Please try again later or reduce input size."
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Handle task execution
    if analyze_btn:
        if selected_task == "Custom Task" and custom_task:
            task_prompt = custom_task
        else:
            task_prompt = task_options[selected_task]
        
        if website_url:
            task_prompt += f"\nAlso consider the information from this website: {website_url}"
        
        pdf_urls = [pdf_url] if pdf_url else []
        pdf_paths = [process_uploaded_file(uploaded_file)] if uploaded_file else []
        website_urls = [website_url] if website_url else []

        knowledge_base = create_knowledge_base(pdf_urls, pdf_paths, website_urls)
        if knowledge_base:
            st.session_state.agent.knowledge = knowledge_base
            st.session_state.agent.search_knowledge = True

        st.session_state.messages.append({"role": "user", "content": task_prompt})
        with st.chat_message("user"):
            st.markdown(task_prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            files = []
            if pdf_url:
                files.append(File(url=pdf_url))
            if uploaded_file:
                file_path = process_uploaded_file(uploaded_file)
                if file_path:
                    files.append(File(path=file_path))
            if website_url:
                files.append(File(url=website_url))
            
            with st.spinner("Processing your request..."):
                try:
                    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
                    def run_agent_with_retry(task_prompt, files):
                        return st.session_state.agent.run(task_prompt, files=files)
                    
                    response = run_agent_with_retry(task_prompt, files)
                    full_response = response.content
                    if hasattr(response, 'thoughts') and response.thoughts:
                        full_response += "\n\n---\n\n**Agent Thoughts:**\n\n" + str(response.thoughts)
                except Exception as e:
                    full_response = f"An error occurred: {str(e)}. If it's a 503 error, the model may be overloaded. Please try again later or reduce input size."
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()