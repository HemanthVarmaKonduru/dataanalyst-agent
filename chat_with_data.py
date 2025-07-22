import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import json
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure Google API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
    max_tokens=4000
)

# Page configuration
st.set_page_config(
    page_title="Chat with Your Data",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for chat interface
st.markdown("""
<style>
    .chat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .chat-header {
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .chat-header h1 {
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .chat-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0.5rem 0;
    }
    .message-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        backdrop-filter: blur(10px);
    }
    .message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 1rem;
        animation: fadeIn 0.5s ease-in;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
        border-bottom-right-radius: 0.3rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
        border-bottom-left-radius: 0.3rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .data-insight {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .input-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 1rem;
        backdrop-filter: blur(10px);
        margin-top: 1rem;
    }
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border: none;
        border-radius: 2rem;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 2rem;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    .data-summary {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white;
    }
    .suggestion-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    .suggestion-chip {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .suggestion-chip:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: scale(1.05);
    }
    .visualization-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: white;
        font-style: italic;
    }
    .typing-dot {
        width: 8px;
        height: 8px;
        background: white;
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
    }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_analysis' not in st.session_state:
    st.session_state.data_analysis = None
if 'data_quality_score' not in st.session_state:
    st.session_state.data_quality_score = 0
if 'is_typing' not in st.session_state:
    st.session_state.is_typing = False

def detect_file_encoding(file_content):
    """Detect file encoding using chardet"""
    import chardet
    try:
        result = chardet.detect(file_content)
        return result['encoding']
    except:
        return 'utf-8'

def safe_convert_to_serializable(obj):
    """Safely convert objects to JSON serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {str(key): safe_convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert_to_serializable(item) for item in obj]
    else:
        return str(obj)

def calculate_data_quality_score(df):
    """Calculate data quality score"""
    score = 100
    
    # Check for missing values
    missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    score -= missing_percentage * 2
    
    # Check for duplicates
    duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
    score -= duplicate_percentage * 1.5
    
    # Check for data type consistency
    object_columns = df.select_dtypes(include=['object']).columns
    if len(object_columns) > len(df.columns) * 0.8:
        score -= 10
    
    return max(0, min(100, score))

def analyze_data_for_chat(df):
    """Analyze data specifically for chat context"""
    analysis = {
        'basic_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicates': df.duplicated().sum()
        },
        'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': {
            'total': df.isnull().sum().sum(),
            'percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'by_column': df.isnull().sum().to_dict()
        },
        'numeric_summary': {},
        'categorical_summary': {},
        'quality_score': calculate_data_quality_score(df)
    }
    
    # Numeric analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        analysis['numeric_summary'] = {
            'columns': list(numeric_cols),
            'statistics': df[numeric_cols].describe().to_dict(),
            'correlations': df[numeric_cols].corr().to_dict() if len(numeric_cols) > 1 else {}
        }
    
    # Categorical analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        analysis['categorical_summary'] = {
            'columns': list(categorical_cols),
            'unique_counts': {col: df[col].nunique() for col in categorical_cols},
            'top_values': {col: df[col].value_counts().head(3).to_dict() for col in categorical_cols}
        }
    
    return analysis

def chat_with_data(df, user_message, analysis_results):
    """Enhanced chat function with better data context"""
    
    # Prepare comprehensive data context
    data_context = f"""
    You are a helpful data analyst assistant. You have access to a dataset with the following information:
    
    DATASET OVERVIEW:
    - Shape: {df.shape[0]} rows × {df.shape[1]} columns
    - Total memory usage: {analysis_results['basic_info']['memory_usage']:.2f} MB
    - Data quality score: {analysis_results['quality_score']:.1f}/100
    - Duplicate rows: {analysis_results['basic_info']['duplicates']}
    
    COLUMNS AND DATA TYPES:
    {chr(10).join([f"- {col}: {dtype}" for col, dtype in analysis_results['data_types'].items()])}
    
    MISSING VALUES:
    - Total missing: {analysis_results['missing_values']['total']} ({analysis_results['missing_values']['percentage']:.2f}%)
    - Columns with missing values: {[col for col, count in analysis_results['missing_values']['by_column'].items() if count > 0]}
    
    SAMPLE DATA (first 3 rows):
    {df.head(3).to_string()}
    
    NUMERIC COLUMNS ({len(analysis_results['numeric_summary'].get('columns', []))}):
    """
    
    if analysis_results['numeric_summary']:
        data_context += f"""
        {chr(10).join([f"- {col}" for col in analysis_results['numeric_summary']['columns']])}
        
        STATISTICAL SUMMARY:
        {df[analysis_results['numeric_summary']['columns']].describe().to_string()}
        """
    
    if analysis_results['categorical_summary']:
        data_context += f"""
        
        CATEGORICAL COLUMNS ({len(analysis_results['categorical_summary']['columns'])}):
        {chr(10).join([f"- {col}: {analysis_results['categorical_summary']['unique_counts'][col]} unique values" for col in analysis_results['categorical_summary']['columns']])}
        """
    
    # Create the enhanced prompt
    prompt = f"""
    {data_context}
    
    USER QUESTION: {user_message}
    
    INSTRUCTIONS:
    1. Answer the question based ONLY on the provided data
    2. If the question cannot be answered with the available data, say so clearly
    3. Provide specific numbers, percentages, and insights from the data
    4. Do not make assumptions or hallucinate information
    5. If you need to perform calculations, show your work
    6. Be conversational and helpful, like a friendly data analyst
    7. If the data quality score is low (< 70), mention this in your response
    8. Provide actionable insights when possible
    9. If the user asks for visualizations, describe what you would create
    10. Keep responses concise but thorough
    
    Please provide your analysis in a conversational tone:
    """
    
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"I encountered an error while processing your question. Please try rephrasing or ask a different question about your data."

def create_visualization_for_query(df, query, analysis_results):
    """Create visualizations based on user query"""
    try:
        # Simple keyword-based visualization selection
        query_lower = query.lower()
        
        if 'correlation' in query_lower or 'relationship' in query_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Correlation Heatmap')
                return fig
        
        elif 'distribution' in query_lower or 'histogram' in query_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(15, 5))
                if len(numeric_cols) == 1:
                    axes = [axes]
                for i, col in enumerate(numeric_cols[:3]):
                    df[col].hist(ax=axes[i], bins=30, alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                plt.tight_layout()
                return fig
        
        elif 'missing' in query_lower:
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_data[missing_data > 0].plot(kind='bar', ax=ax)
                ax.set_title('Missing Values by Column')
                ax.set_xlabel('Columns')
                ax.set_ylabel('Missing Count')
                plt.xticks(rotation=45)
                return fig
        
        return None
    except:
        return None

def get_suggested_questions(df, analysis_results):
    """Generate suggested questions based on the data"""
    suggestions = []
    
    # Basic questions
    suggestions.append("What are the main insights from this dataset?")
    suggestions.append("Show me a summary of the data")
    
    # Numeric questions
    if analysis_results['numeric_summary']:
        suggestions.append("What are the correlations between numeric variables?")
        suggestions.append("Show me the distribution of numeric columns")
    
    # Categorical questions
    if analysis_results['categorical_summary']:
        suggestions.append("What are the most common categories?")
        suggestions.append("Show me the unique values in categorical columns")
    
    # Quality questions
    if analysis_results['missing_values']['total'] > 0:
        suggestions.append("How much missing data do we have?")
    
    # Specific questions based on data
    if len(df.columns) > 5:
        suggestions.append("Which columns have the most variation?")
    
    return suggestions[:6]  # Limit to 6 suggestions

def main():
    # Main chat interface
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h1>💬 Chat with Your Data</h1>
            <p>Upload your CSV file and start a conversation with your data!</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown("### 📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file (max 1MB)",
            type=['csv'],
            help="Upload a CSV file to start chatting with your data"
        )
        
        if uploaded_file is not None:
            # Check file size
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size > 1:
                st.error(f"File size ({file_size:.2f}MB) exceeds 1MB limit")
                return
            
            # Check file type
            file_name = uploaded_file.name.lower()
            if not file_name.endswith('.csv'):
                st.error("Please upload a CSV file")
                return
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            try:
                # Read file content for encoding detection
                file_content = uploaded_file.read()
                uploaded_file.seek(0)
                
                # Try to detect encoding first
                import chardet
                detected_encoding = chardet.detect(file_content)['encoding']
                
                # Try different encodings
                encodings = [detected_encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                encodings = list(dict.fromkeys(encodings))
                
                df = None
                successful_encoding = None
                
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        successful_encoding = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        continue
                
                if df is None:
                    st.error("❌ Could not read file with any supported encoding")
                    return
                
                if successful_encoding and successful_encoding != 'utf-8':
                    st.info(f"📝 File read using {successful_encoding} encoding")
                
                st.session_state.uploaded_data = df
                
                # Analyze data for chat
                with st.spinner("🔍 Analyzing your data..."):
                    st.session_state.data_analysis = analyze_data_for_chat(df)
                    st.session_state.data_quality_score = st.session_state.data_analysis['quality_score']
                
                st.success(f"✅ Data loaded! {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Show data summary
                st.markdown("### 📊 Data Summary")
                st.write(f"**Quality Score:** {st.session_state.data_quality_score:.1f}/100")
                st.write(f"**Memory Usage:** {st.session_state.data_analysis['basic_info']['memory_usage']:.2f} MB")
                st.write(f"**Missing Values:** {st.session_state.data_analysis['missing_values']['total']}")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
    
    # Main chat area
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        analysis = st.session_state.data_analysis
        
        # Data quality indicator
        quality_score = st.session_state.data_quality_score
        if quality_score < 70:
            st.warning(f"⚠️ Data quality score is {quality_score:.1f}/100. Consider cleaning your data for better analysis.")
        
        # Chat messages container
        st.markdown('<div class="message-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="message user-message"><strong>You:</strong> {message["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message assistant-message"><strong>Data Assistant:</strong> {message["content"]}</div>', 
                          unsafe_allow_html=True)
                
                # Show visualization if available
                if 'visualization' in message and message['visualization'] is not None:
                    st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                    st.pyplot(message['visualization'])
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Show typing indicator
        if st.session_state.is_typing:
            st.markdown("""
            <div class="message assistant-message">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    Analyzing your data...
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Suggested questions
        if len(st.session_state.chat_history) == 0:
            st.markdown("### 💡 Suggested Questions")
            suggestions = get_suggested_questions(df, analysis)
            
            st.markdown('<div class="suggestion-chips">', unsafe_allow_html=True)
            for suggestion in suggestions:
                if st.button(suggestion, key=f"suggest_{suggestion[:20]}"):
                    # Add user message
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': suggestion
                    })
                    
                    # Set typing indicator
                    st.session_state.is_typing = True
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g., What are the main insights from this dataset?",
                key="chat_input"
            )
        
        with col2:
            if st.button("Send", key="send_button"):
                if user_input.strip():
                    # Add user message
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_input
                    })
                    
                    # Set typing indicator
                    st.session_state.is_typing = True
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process chat if typing
        if st.session_state.is_typing:
            # Get AI response
            ai_response = chat_with_data(df, user_input, analysis)
            
            # Create visualization if relevant
            visualization = create_visualization_for_query(df, user_input, analysis)
            
            # Add AI response
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': ai_response,
                'visualization': visualization
            })
            
            # Clear typing indicator
            st.session_state.is_typing = False
            
            # Clear input
            st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    else:
        # Welcome message when no file is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 4rem;">
            <h2>Welcome to Chat with Your Data! 🚀</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Upload a CSV file to start a conversation with your data.
            </p>
            <div style="margin-top: 2rem;">
                <h3>🎯 How it works:</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    <div style="background: white; padding: 1.5rem; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h4>📁 Upload Data</h4>
                        <p>Upload your CSV file (max 1MB)</p>
                    </div>
                    <div style="background: white; padding: 1.5rem; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h4>💬 Start Chatting</h4>
                        <p>Ask questions in natural language</p>
                    </div>
                    <div style="background: white; padding: 1.5rem; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h4>📊 Get Insights</h4>
                        <p>Receive data-driven answers and visualizations</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 