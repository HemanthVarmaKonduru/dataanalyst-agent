# 📊 Data Analyst Agent

An intelligent data analysis tool powered by Streamlit and Google's Gemini AI that allows users to upload CSV files and interact with their data using natural language queries.

## 🚀 Features

- **📁 File Upload**: Upload CSV files up to 1MB for analysis
- **📊 Comprehensive Analysis**: Automatic data profiling, statistics, and insights
- **📈 Interactive Visualizations**: Multiple chart types including distributions, correlations, and categorical plots
- **🤖 AI-Powered Queries**: Ask questions about your data in natural language
- **📋 Data Quality Assessment**: Missing value analysis, data type detection, and quality insights
- **💬 Chat Interface**: Conversational AI assistant for data exploration
- **📥 Export Capabilities**: Download processed data and analysis results

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Google API key for Gemini AI

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd dataanalyst-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get Google API Key**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the API key

4. **Configure Environment**
   - Create a `.env` file in the project root
   - Add your Google API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   GEMINI_MODEL=gemini-1.5-pro
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

6. **Access the Application**
   - Open your browser and go to `http://localhost:8501`
   - The application will automatically open in your default browser

## 📖 Usage Guide

### 1. Upload Data
- Click "Browse files" in the sidebar
- Select a CSV file (max 1MB)
- The system will automatically analyze your data

### 2. Explore Overview
- **Dataset Overview**: Basic metrics, data types, and missing values
- **Statistical Summary**: Descriptive statistics for numeric columns
- **Key Insights**: AI-generated insights about your data

### 3. View Visualizations
- **Distributions**: Histograms for numeric variables
- **Correlations**: Heatmap showing relationships between variables
- **Box Plots**: Outlier detection and distribution spread
- **Categorical Plots**: Bar charts for categorical variables

### 4. Ask Questions
- Go to the "AI Chat" tab
- Type questions in natural language, such as:
  - "What is the average value of column X?"
  - "Show me the correlation between A and B"
  - "What are the top 5 values in column Y?"
  - "How many missing values are there?"
  - "What insights can you find in this data?"

### 5. Export Results
- Download processed data as CSV
- View raw data in the "Raw Data" tab

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Your Google API key | Required |
| `GEMINI_MODEL` | Gemini model to use | `gemini-1.5-pro` |

### Model Settings

The application uses Google's Gemini 1.5 Pro model with the following settings:
- **Temperature**: 0.1 (low creativity for accurate responses)
- **Max Tokens**: 4000 (sufficient for detailed analysis)
- **Model**: gemini-1.5-pro (latest and most capable)

## 📊 Supported Data Types

- **Numeric**: Integer, float, decimal
- **Categorical**: String, object, category
- **Date/Time**: Automatically detected and handled
- **Boolean**: True/False values

## 🎯 Example Queries

Here are some example questions you can ask the AI assistant:

### Basic Statistics
- "What is the mean of the price column?"
- "Show me the standard deviation of all numeric columns"
- "What is the range of values in column X?"

### Data Quality
- "How many missing values are there in each column?"
- "Which columns have the most missing data?"
- "Are there any outliers in the numeric columns?"

### Relationships
- "What is the correlation between price and mileage?"
- "Show me the relationship between category and value"
- "Are there any strong correlations in the data?"

### Insights
- "What patterns do you see in this data?"
- "What are the key insights from this dataset?"
- "What should I focus on when analyzing this data?"

## 🔒 Security & Privacy

- **Local Processing**: All data processing happens locally on your machine
- **API Calls**: Only metadata and analysis requests are sent to Google's API
- **No Data Storage**: Your data is not stored or transmitted beyond analysis requests
- **Secure API**: Uses Google's secure API endpoints with your private key

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your `.env` file exists and contains the correct API key
   - Verify the API key is valid and has proper permissions

2. **File Upload Issues**
   - Check file size (must be < 1MB)
   - Ensure file is in CSV format
   - Verify file is not corrupted

3. **Memory Issues**
   - Large datasets may cause memory problems
   - Consider sampling your data for initial analysis

4. **Installation Issues**
   - Update pip: `pip install --upgrade pip`
   - Install in a virtual environment
   - Check Python version compatibility

### Performance Tips

- Use smaller datasets for faster analysis
- Close other applications to free up memory
- Restart the application if it becomes slow

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Review the error messages in the application
3. Ensure all dependencies are properly installed

## 🔄 Updates

To update the application:
1. Pull the latest changes
2. Update dependencies: `pip install -r requirements.txt --upgrade`
3. Restart the application

---

**Happy Data Analyzing! 📊✨** 