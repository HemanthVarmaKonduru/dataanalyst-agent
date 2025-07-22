"""
Configuration file for Data Analyst Agent
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google API Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')

# Application Configuration
APP_TITLE = "Data Analyst Agent"
APP_ICON = "📊"
PAGE_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# File Upload Configuration
MAX_FILE_SIZE_MB = 1
ALLOWED_FILE_TYPES = ['csv']

# Model Configuration
MODEL_TEMPERATURE = 0.1  # Low temperature for accurate responses
MAX_TOKENS = 4000
MODEL_TIMEOUT = 30  # seconds

# Analysis Configuration
CORRELATION_THRESHOLD = 0.7  # Threshold for high correlations
MAX_CATEGORICAL_VALUES = 10  # Max values to show in categorical plots
MAX_NUMERIC_COLUMNS_PLOT = 6  # Max numeric columns to plot at once

# Visualization Configuration
FIGURE_SIZE_LARGE = (15, 10)
FIGURE_SIZE_MEDIUM = (12, 6)
FIGURE_SIZE_SMALL = (10, 6)
DPI = 100

# Chat Configuration
MAX_CHAT_HISTORY = 50  # Maximum number of chat messages to keep
CHAT_MESSAGE_LIMIT = 2000  # Maximum characters per chat message

# Data Processing Configuration
SAMPLE_SIZE_LARGE_DATASET = 10000  # Sample size for large datasets
MAX_ROWS_DISPLAY = 1000  # Maximum rows to display in raw data view

# Error Messages
ERROR_MESSAGES = {
    'no_api_key': "Please set your GOOGLE_API_KEY in the .env file",
    'file_too_large': f"File size exceeds {MAX_FILE_SIZE_MB}MB limit",
    'invalid_file_type': f"Only {', '.join(ALLOWED_FILE_TYPES)} files are supported",
    'api_error': "Error connecting to Google API. Please check your API key and internet connection",
    'processing_error': "Error processing your data. Please check the file format",
    'query_error': "Error processing your query. Please try again with a different question"
}

# Success Messages
SUCCESS_MESSAGES = {
    'file_uploaded': "✅ File uploaded successfully!",
    'analysis_complete': "✅ Analysis complete!",
    'api_connected': "✅ Connected to Google API successfully"
}

# CSS Styles
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .info-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
"""

# Example Queries for User Guidance
EXAMPLE_QUERIES = [
    "What is the average price of cars in this dataset?",
    "Which car brand has the highest horsepower?",
    "Show me the correlation between price and horsepower",
    "What are the top 5 most expensive cars?",
    "How many cars are there for each fuel type?",
    "What is the distribution of car prices?",
    "Which cars have the fastest 0-100 km/h time?",
    "What is the relationship between engine size and performance?",
    "How many seats do most cars have?",
    "What insights can you find about electric vs petrol cars?"
]

# Data Quality Checks
DATA_QUALITY_CHECKS = {
    'missing_values_threshold': 0.5,  # 50% missing values threshold
    'outlier_threshold': 3,  # Standard deviations for outlier detection
    'duplicate_threshold': 0.1,  # 10% duplicate threshold
    'data_type_consistency': True,  # Check for data type consistency
    'value_range_check': True,  # Check for reasonable value ranges
}

# Performance Settings
PERFORMANCE_SETTINGS = {
    'enable_caching': True,
    'cache_ttl': 3600,  # 1 hour cache TTL
    'max_concurrent_requests': 5,
    'request_timeout': 30,
    'enable_progress_bars': True,
}

# Export Settings
EXPORT_SETTINGS = {
    'csv_encoding': 'utf-8',
    'include_timestamp': True,
    'compression': False,
    'max_export_rows': 10000,
}

# Validation Functions
def validate_api_key():
    """Validate that the API key is set and not a placeholder"""
    if not GOOGLE_API_KEY:
        return False, "No API key found"
    
    if GOOGLE_API_KEY == "your_google_api_key_here":
        return False, "API key is still a placeholder"
    
    return True, "API key is valid"

def validate_file_size(file_size_bytes):
    """Validate file size"""
    file_size_mb = file_size_bytes / (1024 * 1024)
    return file_size_mb <= MAX_FILE_SIZE_MB

def validate_file_type(file_name):
    """Validate file type"""
    file_extension = file_name.split('.')[-1].lower()
    return file_extension in ALLOWED_FILE_TYPES

# Default Analysis Settings
DEFAULT_ANALYSIS_SETTINGS = {
    'include_correlations': True,
    'include_outliers': True,
    'include_missing_analysis': True,
    'include_data_type_analysis': True,
    'include_summary_statistics': True,
    'include_visualizations': True,
    'include_insights': True,
    'max_insights': 5,
    'correlation_method': 'pearson',
    'outlier_method': 'iqr',  # or 'zscore'
} 