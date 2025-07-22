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
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'data_analysis' not in st.session_state:
    st.session_state.data_analysis = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def detect_file_encoding(file_content):
    """Detect the encoding of a file content"""
    import chardet
    
    try:
        result = chardet.detect(file_content)
        return result['encoding']
    except:
        return 'utf-8'

def analyze_data(df):
    """Perform comprehensive data analysis"""
    analysis = {
        'basic_info': {},
        'statistics': {},
        'missing_values': {},
        'data_types': {},
        'correlations': {},
        'insights': []
    }
    
    # Basic information
    analysis['basic_info'] = {
        'shape': df.shape,
        'columns': list(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    }
    
    # Data types
    analysis['data_types'] = df.dtypes.to_dict()
    
    # Missing values
    missing_data = df.isnull().sum()
    analysis['missing_values'] = {
        'total_missing': missing_data.sum(),
        'missing_percentage': (missing_data.sum() / len(df)) * 100,
        'missing_by_column': missing_data[missing_data > 0].to_dict()
    }
    
    # Statistical summary
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        analysis['statistics'] = {
            'numeric_columns': list(numeric_columns),
            'summary_stats': df[numeric_columns].describe().to_dict(),
            'skewness': df[numeric_columns].skew().to_dict(),
            'kurtosis': df[numeric_columns].kurtosis().to_dict()
        }
        
        # Correlations for numeric columns
        if len(numeric_columns) > 1:
            correlation_matrix = df[numeric_columns].corr()
            analysis['correlations'] = {
                'matrix': correlation_matrix.to_dict(),
                'high_correlations': []
            }
            
            # Find high correlations
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        analysis['correlations']['high_correlations'].append({
                            'col1': correlation_matrix.columns[i],
                            'col2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
    
    # Categorical analysis
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        analysis['categorical'] = {}
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            analysis['categorical'][col] = {
                'unique_values': len(value_counts),
                'top_values': value_counts.head(5).to_dict(),
                'null_count': df[col].isnull().sum()
            }
    
    # Generate insights using LLM
    insights_prompt = f"""
    Analyze this dataset and provide 3-5 key insights:
    
    Dataset Info:
    - Shape: {df.shape}
    - Columns: {list(df.columns)}
    - Data types: {df.dtypes.to_dict()}
    - Missing values: {missing_data.sum()} total
    
    Numeric columns: {list(numeric_columns) if len(numeric_columns) > 0 else 'None'}
    Categorical columns: {list(categorical_columns) if len(categorical_columns) > 0 else 'None'}
    
    Please provide concise, actionable insights about the data quality, patterns, and potential analysis opportunities.
    """
    
    try:
        response = model.invoke([HumanMessage(content=insights_prompt)])
        analysis['insights'] = response.content.split('\n')
    except Exception as e:
        analysis['insights'] = [f"Error generating insights: {str(e)}"]
    
    return analysis

def create_visualizations(df):
    """Create various visualizations for the dataset"""
    plots = {}
    
    # Get column types
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    # 1. Distribution plots for numeric columns
    if len(numeric_columns) > 0:
        fig_dist, axes = plt.subplots(2, min(3, len(numeric_columns)), figsize=(15, 10))
        if len(numeric_columns) == 1:
            axes = [axes]
        elif len(numeric_columns) <= 3:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_columns[:6]):  # Limit to 6 columns
            if i < len(axes):
                # Histogram
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plots['distributions'] = fig_dist
    
    # 2. Correlation heatmap
    if len(numeric_columns) > 1:
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Heatmap')
        plots['correlation'] = fig_corr
    
    # 3. Box plots for numeric columns
    if len(numeric_columns) > 0:
        fig_box, ax = plt.subplots(figsize=(12, 6))
        df[numeric_columns].boxplot(ax=ax)
        ax.set_title('Box Plots of Numeric Variables')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()
        plots['boxplots'] = fig_box
    
    # 4. Categorical plots
    if len(categorical_columns) > 0:
        for col in categorical_columns[:3]:  # Limit to 3 categorical columns
            fig_cat, ax = plt.subplots(figsize=(10, 6))
            value_counts = df[col].value_counts().head(10)
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title(f'Top 10 Values in {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plots[f'categorical_{col}'] = fig_cat
    
    return plots

def convert_to_serializable(obj):
    """Convert numpy/pandas objects to JSON serializable format"""
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.core.dtypes.dtypes._DTypeBase):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__class__') and 'dtype' in str(type(obj)):
        return str(obj)
    else:
        return obj

def query_data_with_llm(df, user_query, analysis_results):
    """Use LLM to answer questions about the data"""
    
    # Prepare context about the data
    data_context = f"""
    Dataset Information:
    - Shape: {df.shape}
    - Columns: {list(df.columns)}
    - Missing values: {df.isnull().sum().sum()} total
    
    Sample data (first 5 rows):
    {df.head().to_string()}
    
    Statistical summary:
    {df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else 'No numeric columns'}
    
    Key analysis insights:
    - Total rows: {df.shape[0]}
    - Total columns: {df.shape[1]}
    - Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}
    - Categorical columns: {len(df.select_dtypes(include=['object']).columns)}
    - Missing data percentage: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%
    
    Column information:
    {chr(10).join([f"- {col}: {df[col].dtype}" for col in df.columns])}
    
    Missing values by column:
    {chr(10).join([f"- {col}: {df[col].isnull().sum()}" for col in df.columns if df[col].isnull().sum() > 0])}
    """
    
    # Create the prompt
    prompt = f"""
    You are a data analyst assistant. You have access to a dataset with the following information:
    
    {data_context}
    
    User Question: {user_query}
    
    Instructions:
    1. Answer the question based ONLY on the provided data
    2. If the question cannot be answered with the available data, say so clearly
    3. Provide specific numbers and insights from the data
    4. Do not make assumptions or hallucinate information
    5. If you need to perform calculations, show your work
    6. Be concise but thorough
    
    Please provide your analysis:
    """
    
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        st.error(f"Error processing your query: {str(e)}")
        return f"I encountered an error while processing your query. Please try rephrasing your question or check if the data contains the information you're looking for."

def main():
    st.markdown('<h1 class="main-header">📊 Data Analyst Agent</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown('<h3 class="sub-header">📁 Upload Data</h3>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file (max 1MB)",
            type=['csv'],
            help="Upload a CSV file smaller than 1MB for analysis"
        )
        
        if uploaded_file is not None:
            # Check file size
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
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
                uploaded_file.seek(0)  # Reset file pointer
                
                # Try to detect encoding first
                detected_encoding = detect_file_encoding(file_content)
                
                # Try different encodings to handle various file formats
                encodings = [detected_encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                encodings = list(dict.fromkeys(encodings))  # Remove duplicates while preserving order
                
                df = None
                successful_encoding = None
                
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        successful_encoding = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        # If it's not an encoding error, try the next encoding
                        continue
                
                if df is None:
                    st.error("❌ Could not read file with any supported encoding. Please check your file format.")
                    return
                
                if successful_encoding and successful_encoding != 'utf-8':
                    st.info(f"📝 File read successfully using {successful_encoding} encoding")
                
                st.session_state.uploaded_data = df
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Perform analysis
                with st.spinner("Analyzing data..."):
                    st.session_state.data_analysis = analyze_data(df)
                    st.session_state.plots = create_visualizations(df)
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.info("💡 Try these solutions:")
                st.info("• Check if the file is a valid CSV")
                st.info("• Try opening and resaving the file in a text editor")
                st.info("• Ensure the file doesn't contain special characters")
                return
    
    # Main content area
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        analysis = st.session_state.data_analysis
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Visualizations", "🤖 AI Chat", "📋 Raw Data"])
        
        with tab1:
            st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
            
            # Basic metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", analysis['basic_info']['shape'][0])
            with col2:
                st.metric("Columns", analysis['basic_info']['shape'][1])
            with col3:
                st.metric("Memory Usage", f"{analysis['basic_info']['memory_usage']:.2f} MB")
            with col4:
                st.metric("Missing Values", analysis['missing_values']['total_missing'])
            
            # Data types
            st.subheader("Data Types")
            dtype_df = pd.DataFrame(list(analysis['data_types'].items()), columns=['Column', 'Data Type'])
            st.dataframe(dtype_df, use_container_width=True)
            
            # Missing values
            if analysis['missing_values']['total_missing'] > 0:
                st.subheader("Missing Values")
                missing_df = pd.DataFrame(list(analysis['missing_values']['missing_by_column'].items()), 
                                        columns=['Column', 'Missing Count'])
                st.dataframe(missing_df, use_container_width=True)
            
            # Statistical summary
            if analysis['statistics']:
                st.subheader("Statistical Summary")
                stats_df = pd.DataFrame(analysis['statistics']['summary_stats'])
                st.dataframe(stats_df, use_container_width=True)
            
            # Insights
            st.subheader("Key Insights")
            for insight in analysis['insights']:
                if insight.strip():
                    st.markdown(f"• {insight.strip()}")
        
        with tab2:
            st.markdown('<h2 class="sub-header">Data Visualizations</h2>', unsafe_allow_html=True)
            
            if hasattr(st.session_state, 'plots'):
                plots = st.session_state.plots
                
                # Display plots
                for plot_name, fig in plots.items():
                    st.subheader(plot_name.replace('_', ' ').title())
                    st.pyplot(fig)
                    plt.close(fig)  # Close to free memory
        
        with tab3:
            st.markdown('<h2 class="sub-header">AI Data Assistant</h2>', unsafe_allow_html=True)
            st.write("Ask questions about your data using natural language!")
            
            # Chat interface
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', 
                              unsafe_allow_html=True)
            
            # Input for new questions
            user_question = st.text_input("Ask a question about your data:", key="user_question")
            
            if st.button("Ask Question", key="ask_button"):
                if user_question.strip():
                    # Add user message to chat
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_question
                    })
                    
                    # Get AI response
                    with st.spinner("Analyzing your question..."):
                        ai_response = query_data_with_llm(df, user_question, analysis)
                    
                    # Add AI response to chat
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': ai_response
                    })
                    
                    # Clear input
                    st.rerun()
            
            # Clear chat button
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        with tab4:
            st.markdown('<h2 class="sub-header">Raw Data</h2>', unsafe_allow_html=True)
            
            # Convert dataframe to display-friendly format
            try:
                display_df = df.copy()
                # Convert any problematic data types
                for col in display_df.columns:
                    if display_df[col].dtype == 'object':
                        display_df[col] = display_df[col].astype(str)
                
                st.dataframe(display_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying data: {str(e)}")
                st.info("Try viewing the data in smaller chunks or check for problematic data types")
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download processed data as CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome message when no file is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 4rem;">
            <h2>Welcome to Data Analyst Agent! 🤖</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Upload a CSV file (max 1MB) to get started with AI-powered data analysis.
            </p>
            <div style="margin-top: 2rem;">
                <h3>Features:</h3>
                <ul style="text-align: left; display: inline-block;">
                    <li>📊 Comprehensive data analysis</li>
                    <li>📈 Interactive visualizations</li>
                    <li>🤖 Natural language queries</li>
                    <li>📋 Statistical summaries</li>
                    <li>🔍 Data quality insights</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 