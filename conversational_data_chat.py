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

# Initialize Gemini model with higher temperature for more conversational responses
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,  # Higher temperature for more conversational responses
    max_tokens=4000
)

# Page configuration
st.set_page_config(
    page_title="DataChat - Talk to Your Data",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-modern chat CSS
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 0;
        margin: 0;
    }
    
    /* Chat header */
    .chat-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        color: white;
    }
    
    .chat-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .chat-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Chat messages area */
    .chat-messages {
        height: calc(100vh - 200px);
        overflow-y: auto;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Message bubbles */
    .message-bubble {
        margin: 1rem 0;
        padding: 1rem 1.5rem;
        border-radius: 1.5rem;
        max-width: 80%;
        word-wrap: break-word;
        animation: slideIn 0.3s ease-out;
        position: relative;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .assistant-bubble {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: auto;
        border-bottom-left-radius: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .system-bubble {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        margin: 1rem auto;
        text-align: center;
        border-radius: 1rem;
        font-style: italic;
        max-width: 60%;
    }
    
    /* Input area */
    .input-area {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 1000;
    }
    
    .input-container {
        display: flex;
        gap: 1rem;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Enhanced input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border: none;
        border-radius: 2rem;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 2rem;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        min-width: 120px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Quick actions */
    .quick-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
        justify-content: center;
    }
    
    .quick-action {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.3);
        font-size: 0.9rem;
    }
    
    .quick-action:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: scale(1.05);
    }
    
    /* Data insights */
    .data-insight {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4facfe;
    }
    
    /* Visualization container */
    .viz-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: white;
        font-style: italic;
        padding: 1rem;
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
    
    @keyframes slideIn {
        from { 
            opacity: 0; 
            transform: translateY(20px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    /* Welcome message */
    .welcome-container {
        text-align: center;
        padding: 4rem 2rem;
        color: white;
    }
    
    .welcome-container h2 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .feature-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
    }
    
    .feature-card p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.9rem;
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
if 'is_typing' not in st.session_state:
    st.session_state.is_typing = False
if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False

def detect_file_encoding(file_content):
    """Detect file encoding using chardet"""
    import chardet
    try:
        result = chardet.detect(file_content)
        return result['encoding']
    except:
        return 'utf-8'

def analyze_data_for_conversation(df):
    """Analyze data specifically for conversational context"""
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
        'quality_score': 0
    }
    
    # Calculate quality score
    score = 100
    missing_percentage = analysis['missing_values']['percentage']
    score -= missing_percentage * 2
    duplicate_percentage = (analysis['basic_info']['duplicates'] / len(df)) * 100
    score -= duplicate_percentage * 1.5
    analysis['quality_score'] = max(0, min(100, score))
    
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

def conversational_data_chat(df, user_message, analysis_results, chat_history):
    """Enhanced conversational chat function"""
    
    # Build conversation context
    conversation_context = ""
    if len(chat_history) > 0:
        # Include recent conversation history
        recent_messages = chat_history[-3:]  # Last 3 messages
        conversation_context = "Recent conversation:\n"
        for msg in recent_messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n"
        conversation_context += "\n"
    
    # Prepare data context
    data_context = f"""
    You are a friendly and knowledgeable data analyst assistant. You're having a conversation with a user about their dataset.
    
    DATASET INFORMATION:
    - Size: {df.shape[0]} rows × {df.shape[1]} columns
    - Memory: {analysis_results['basic_info']['memory_usage']:.2f} MB
    - Quality Score: {analysis_results['quality_score']:.1f}/100
    - Missing Data: {analysis_results['missing_values']['total']} values ({analysis_results['missing_values']['percentage']:.1f}%)
    
    COLUMNS:
    {chr(10).join([f"- {col} ({dtype})" for col, dtype in analysis_results['data_types'].items()])}
    
    SAMPLE DATA (first 2 rows):
    {df.head(2).to_string()}
    """
    
    if analysis_results['numeric_summary']:
        data_context += f"""
        
        NUMERIC COLUMNS ({len(analysis_results['numeric_summary']['columns'])}):
        {chr(10).join([f"- {col}" for col in analysis_results['numeric_summary']['columns']])}
        """
    
    if analysis_results['categorical_summary']:
        data_context += f"""
        
        CATEGORICAL COLUMNS ({len(analysis_results['categorical_summary']['columns'])}):
        {chr(10).join([f"- {col}" for col in analysis_results['categorical_summary']['columns']])}
        """
    
    # Create the conversational prompt
    prompt = f"""
    {data_context}
    
    {conversation_context}
    
    USER'S LATEST MESSAGE: {user_message}
    
    INSTRUCTIONS:
    1. Respond in a conversational, friendly tone like you're chatting with a friend
    2. Answer based ONLY on the provided data - don't make assumptions
    3. If you can't answer with the available data, say so clearly but helpfully
    4. Provide specific numbers and insights when possible
    5. Be encouraging and suggest follow-up questions
    6. If the data quality is low, mention it gently
    7. Keep responses concise but thorough
    8. Use natural language, not technical jargon unless asked
    9. Show enthusiasm for data analysis!
    10. If appropriate, suggest what visualizations could help
    
    Respond as a friendly data analyst:
    """
    
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"I'm having trouble processing that right now. Could you try rephrasing your question? I'm here to help you explore your data!"

def get_conversation_starters(df, analysis_results):
    """Get conversation starters based on the data"""
    starters = []
    
    # Basic starters
    starters.append("Tell me about this dataset")
    starters.append("What's the most interesting thing about this data?")
    starters.append("Give me a quick overview")
    
    # Data quality starters
    if analysis_results['missing_values']['total'] > 0:
        starters.append("How much missing data do we have?")
    
    # Numeric starters
    if analysis_results['numeric_summary']:
        starters.append("What are the main trends in the numeric data?")
        starters.append("Show me some interesting statistics")
    
    # Categorical starters
    if analysis_results['categorical_summary']:
        starters.append("What are the most common categories?")
    
    # Size-based starters
    if df.shape[0] > 1000:
        starters.append("This is a large dataset! What should I focus on?")
    elif df.shape[0] < 100:
        starters.append("This is a small dataset. What can we learn from it?")
    
    return starters[:6]

def create_smart_visualization(df, query, analysis_results):
    """Create smart visualizations based on query content"""
    try:
        query_lower = query.lower()
        
        # Correlation analysis
        if any(word in query_lower for word in ['correlation', 'relationship', 'related', 'connection']):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Correlation Heatmap')
                return fig
        
        # Distribution analysis
        elif any(word in query_lower for word in ['distribution', 'histogram', 'spread', 'range']):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(15, 5))
                if len(numeric_cols) == 1:
                    axes = [axes]
                for i, col in enumerate(numeric_cols[:3]):
                    df[col].hist(ax=axes[i], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                plt.tight_layout()
                return fig
        
        # Missing data
        elif any(word in query_lower for word in ['missing', 'null', 'empty']):
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_data[missing_data > 0].plot(kind='bar', ax=ax, color='lightcoral')
                ax.set_title('Missing Values by Column')
                ax.set_xlabel('Columns')
                ax.set_ylabel('Missing Count')
                plt.xticks(rotation=45)
                return fig
        
        # Categorical analysis
        elif any(word in query_lower for word in ['category', 'categorical', 'most common', 'top']):
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                col = categorical_cols[0]  # Use first categorical column
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts = df[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax, color='lightgreen')
                ax.set_title(f'Top 10 Values in {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                return fig
        
        return None
    except:
        return None

def main():
    # Main chat interface
    st.markdown("""
    <div class="main-container">
        <div class="chat-header">
            <h1>🤖 DataChat</h1>
            <p>Your AI companion for exploring data through conversation</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown("### 📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your data to start chatting"
        )
        
        if uploaded_file is not None:
            # File processing
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size > 1:
                st.error(f"File too large ({file_size:.2f}MB). Max 1MB.")
                return
            
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            uploaded_file.seek(0)
            
            # Encoding detection
            detected_encoding = detect_file_encoding(file_content)
            encodings = [detected_encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            encodings = list(dict.fromkeys(encodings))
            
            df = None
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except:
                    continue
            
            if df is None:
                st.error("❌ Could not read file")
                return
            
            st.session_state.uploaded_data = df
            
            # Analyze data
            with st.spinner("🔍 Analyzing your data..."):
                st.session_state.data_analysis = analyze_data_for_conversation(df)
            
            st.success(f"✅ Ready to chat! {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Data summary
            analysis = st.session_state.data_analysis
            st.markdown("### 📊 Quick Stats")
            st.write(f"**Quality:** {analysis['quality_score']:.1f}/100")
            st.write(f"**Missing:** {analysis['missing_values']['total']}")
            st.write(f"**Memory:** {analysis['basic_info']['memory_usage']:.2f} MB")
    
    # Main chat area
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        analysis = st.session_state.data_analysis
        
        # Chat messages area
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        
        # Welcome message if first time
        if not st.session_state.conversation_started:
            st.markdown("""
            <div class="message-bubble system-bubble">
                👋 Hi! I'm your data analysis assistant. I've loaded your dataset and I'm ready to help you explore it! 
                What would you like to know about your data?
            </div>
            """, unsafe_allow_html=True)
            st.session_state.conversation_started = True
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="message-bubble user-bubble">{message["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message-bubble assistant-bubble">{message["content"]}</div>', 
                          unsafe_allow_html=True)
                
                # Show visualization if available
                if 'visualization' in message and message['visualization'] is not None:
                    st.markdown('<div class="viz-container">', unsafe_allow_html=True)
                    st.pyplot(message['visualization'])
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Typing indicator
        if st.session_state.is_typing:
            st.markdown("""
            <div class="message-bubble assistant-bubble">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    Analyzing your data...
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick actions (conversation starters)
        if len(st.session_state.chat_history) == 0:
            st.markdown("### 💡 Conversation Starters")
            starters = get_conversation_starters(df, analysis)
            
            st.markdown('<div class="quick-actions">', unsafe_allow_html=True)
            cols = st.columns(3)
            for i, starter in enumerate(starters):
                with cols[i % 3]:
                    if st.button(starter, key=f"starter_{i}"):
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': starter
                        })
                        st.session_state.is_typing = True
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Input area
        st.markdown('<div class="input-area">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Ask me anything about your data...",
                placeholder="e.g., What's the most interesting pattern in this data?",
                key="chat_input"
            )
        
        with col2:
            if st.button("Send", key="send_button"):
                if user_input.strip():
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_input
                    })
                    st.session_state.is_typing = True
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process chat if typing
        if st.session_state.is_typing:
            ai_response = conversational_data_chat(df, user_input, analysis, st.session_state.chat_history)
            visualization = create_smart_visualization(df, user_input, analysis)
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': ai_response,
                'visualization': visualization
            })
            
            st.session_state.is_typing = False
            st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.conversation_started = False
            st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="welcome-container">
            <h2>Welcome to DataChat! 🤖</h2>
            <p style="font-size: 1.3rem; margin-bottom: 2rem;">
                Upload your CSV file and start a natural conversation with your data.
            </p>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <h4>💬 Natural Conversation</h4>
                    <p>Ask questions in plain English, just like chatting with a friend</p>
                </div>
                <div class="feature-card">
                    <h4>📊 Smart Insights</h4>
                    <p>Get automatic visualizations and data-driven answers</p>
                </div>
                <div class="feature-card">
                    <h4>🔍 Deep Analysis</h4>
                    <p>Explore patterns, correlations, and trends in your data</p>
                </div>
                <div class="feature-card">
                    <h4>🎯 Guided Discovery</h4>
                    <p>Get suggestions for what to explore next</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 