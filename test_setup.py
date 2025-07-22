#!/usr/bin/env python3
"""
Test script to verify the Data Analyst Agent setup
"""

import sys
import os
from dotenv import load_dotenv

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas
        print(f"✅ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn
        print(f"✅ Seaborn {seaborn.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn import failed: {e}")
        return False
    
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import google.generativeai
        print("✅ Google Generative AI")
    except ImportError as e:
        print(f"❌ Google Generative AI import failed: {e}")
        return False
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✅ LangChain Google GenAI")
    except ImportError as e:
        print(f"❌ LangChain Google GenAI import failed: {e}")
        return False
    
    return True

def test_env_config():
    """Test environment configuration"""
    print("\n🔍 Testing environment configuration...")
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print("   Please create a .env file with your Google API key")
        return False
    
    if api_key == "your_google_api_key_here":
        print("❌ Please replace the placeholder API key with your actual Google API key")
        return False
    
    print("✅ Google API key found")
    return True

def test_google_api():
    """Test Google API connection"""
    print("\n🔍 Testing Google API connection...")
    
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            print("❌ No API key available for testing")
            return False
        
        genai.configure(api_key=api_key)
        
        # Test with a simple prompt
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content("Hello, this is a test message.")
        
        if response.text:
            print("✅ Google API connection successful")
            return True
        else:
            print("❌ Google API returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ Google API test failed: {e}")
        return False

def test_data_processing():
    """Test basic data processing capabilities"""
    print("\n🔍 Testing data processing capabilities...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = {
            'numeric': [1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10],
            'categorical': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'text': ['hello', 'world', 'test', 'data', 'analysis', 'python', 'streamlit', 'ai', 'ml', 'nlp']
        }
        
        df = pd.DataFrame(test_data)
        
        # Test basic operations
        assert len(df) == 10, "DataFrame should have 10 rows"
        assert len(df.columns) == 3, "DataFrame should have 3 columns"
        
        # Test missing value detection
        missing_count = df.isnull().sum().sum()
        assert missing_count == 1, f"Should have 1 missing value, found {missing_count}"
        
        # Test data type detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        assert 'numeric' in numeric_cols, "Numeric column not detected"
        assert 'categorical' in categorical_cols, "Categorical column not detected"
        assert 'text' in categorical_cols, "Text column not detected"
        
        print("✅ Data processing capabilities working")
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Data Analyst Agent - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Environment Configuration", test_env_config),
        ("Google API Connection", test_google_api),
        ("Data Processing", test_data_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nTo run the application:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Set up your Google API key in .env file")
        print("3. Check your internet connection for API tests")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 