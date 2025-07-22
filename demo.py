#!/usr/bin/env python3
"""
Demo script for Data Analyst Agent
This script demonstrates the capabilities of the agent using the included car dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import *
import os

def load_demo_data():
    """Load the demo car dataset"""
    try:
        df = pd.read_csv('data/Cars Datasets 2025.csv')
        print(f"✅ Loaded demo dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print("❌ Demo dataset not found. Please ensure 'data/Cars Datasets 2025.csv' exists.")
        return None
    except Exception as e:
        print(f"❌ Error loading demo dataset: {e}")
        return None

def clean_demo_data(df):
    """Clean and prepare the demo data for analysis"""
    print("\n🧹 Cleaning demo data...")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Clean price column - remove currency symbols and convert to numeric
    df_clean['Cars Prices'] = df_clean['Cars Prices'].str.replace('$', '').str.replace(',', '')
    df_clean['Cars Prices'] = df_clean['Cars Prices'].str.replace(' ', '')
    
    # Handle price ranges (take average)
    def extract_price(price_str):
        if '-' in str(price_str):
            parts = str(price_str).split('-')
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                return np.nan
        else:
            try:
                return float(price_str)
            except:
                return np.nan
    
    df_clean['Price_Numeric'] = df_clean['Cars Prices'].apply(extract_price)
    
    # Clean horsepower column
    df_clean['HorsePower_Clean'] = df_clean['HorsePower'].str.extract(r'(\d+)').astype(float)
    
    # Clean engine capacity
    df_clean['Engine_Capacity'] = df_clean['CC/Battery Capacity'].str.extract(r'(\d+)').astype(float)
    
    # Clean performance time
    df_clean['Performance_Clean'] = df_clean['Performance(0 - 100 )KM/H'].str.extract(r'(\d+\.?\d*)').astype(float)
    
    # Clean torque
    df_clean['Torque_Clean'] = df_clean['Torque'].str.extract(r'(\d+)').astype(float)
    
    print("✅ Data cleaning completed")
    return df_clean

def demonstrate_analysis(df):
    """Demonstrate various analysis capabilities"""
    print("\n📊 Demonstrating Analysis Capabilities")
    print("=" * 50)
    
    # Basic statistics
    print("\n1. Basic Dataset Information:")
    print(f"   - Total cars: {len(df)}")
    print(f"   - Columns: {list(df.columns)}")
    print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Missing values
    missing_data = df.isnull().sum()
    print(f"\n2. Missing Values:")
    print(f"   - Total missing: {missing_data.sum()}")
    print(f"   - Missing percentage: {(missing_data.sum() / len(df)) * 100:.2f}%")
    
    # Company analysis
    print(f"\n3. Car Companies Analysis:")
    company_counts = df['Company Names'].value_counts()
    print(f"   - Total companies: {len(company_counts)}")
    print(f"   - Top 5 companies: {list(company_counts.head().index)}")
    
    # Price analysis
    if 'Price_Numeric' in df.columns:
        price_stats = df['Price_Numeric'].describe()
        print(f"\n4. Price Analysis:")
        print(f"   - Average price: ${price_stats['mean']:,.0f}")
        print(f"   - Median price: ${price_stats['50%']:,.0f}")
        print(f"   - Price range: ${price_stats['min']:,.0f} - ${price_stats['max']:,.0f}")
    
    # Performance analysis
    if 'Performance_Clean' in df.columns:
        perf_stats = df['Performance_Clean'].describe()
        print(f"\n5. Performance Analysis (0-100 km/h):")
        print(f"   - Average time: {perf_stats['mean']:.1f} seconds")
        print(f"   - Fastest car: {perf_stats['min']:.1f} seconds")
        print(f"   - Slowest car: {perf_stats['max']:.1f} seconds")

def demonstrate_visualizations(df):
    """Demonstrate visualization capabilities"""
    print("\n📈 Demonstrating Visualization Capabilities")
    print("=" * 50)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Car Dataset Analysis - Demo Visualizations', fontsize=16, fontweight='bold')
    
    # 1. Company distribution
    if 'Company Names' in df.columns:
        company_counts = df['Company Names'].value_counts().head(10)
        axes[0, 0].bar(range(len(company_counts)), company_counts.values)
        axes[0, 0].set_title('Top 10 Car Companies')
        axes[0, 0].set_xlabel('Company')
        axes[0, 0].set_ylabel('Number of Cars')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_xticks(range(len(company_counts)))
        axes[0, 0].set_xticklabels(company_counts.index, rotation=45, ha='right')
    
    # 2. Price distribution
    if 'Price_Numeric' in df.columns:
        axes[0, 1].hist(df['Price_Numeric'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Car Price Distribution')
        axes[0, 1].set_xlabel('Price ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Performance vs Price scatter
    if 'Performance_Clean' in df.columns and 'Price_Numeric' in df.columns:
        axes[1, 0].scatter(df['Performance_Clean'], df['Price_Numeric'], alpha=0.6)
        axes[1, 0].set_title('Performance vs Price')
        axes[1, 0].set_xlabel('0-100 km/h Time (seconds)')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].invert_xaxis()  # Faster times on the right
    
    # 4. Fuel type distribution
    if 'Fuel Types' in df.columns:
        fuel_counts = df['Fuel Types'].value_counts()
        axes[1, 1].pie(fuel_counts.values, labels=fuel_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Fuel Type Distribution')
    
    plt.tight_layout()
    plt.savefig('demo_visualizations.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'demo_visualizations.png'")

def demonstrate_ai_queries(df):
    """Demonstrate AI query capabilities"""
    print("\n🤖 Demonstrating AI Query Capabilities")
    print("=" * 50)
    
    # Example queries that would work with the car dataset
    example_queries = [
        "What is the average price of cars in this dataset?",
        "Which car company has the most models?",
        "What is the correlation between horsepower and price?",
        "Which cars have the fastest 0-100 km/h time?",
        "How many electric cars are there?",
        "What is the price range of Ferrari cars?",
        "Which cars have the highest horsepower?",
        "What is the average performance time for luxury brands?",
        "How many seats do most cars have?",
        "What insights can you find about hybrid vs petrol cars?"
    ]
    
    print("Example queries you can ask in the AI Chat:")
    for i, query in enumerate(example_queries, 1):
        print(f"   {i}. {query}")
    
    print(f"\n💡 The AI can answer questions about:")
    print(f"   - Statistical analysis (mean, median, correlation)")
    print(f"   - Data filtering and grouping")
    print(f"   - Pattern recognition and insights")
    print(f"   - Data quality assessment")
    print(f"   - Comparative analysis between categories")

def main():
    """Main demo function"""
    print("🚗 Data Analyst Agent - Demo")
    print("=" * 60)
    print("This demo showcases the capabilities of the Data Analyst Agent")
    print("using the included car dataset.")
    print()
    
    # Load data
    df = load_demo_data()
    if df is None:
        return
    
    # Clean data
    df_clean = clean_demo_data(df)
    
    # Demonstrate capabilities
    demonstrate_analysis(df_clean)
    demonstrate_visualizations(df_clean)
    demonstrate_ai_queries(df_clean)
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("\nTo run the full application:")
    print("   1. Set up your Google API key in .env file")
    print("   2. Run: streamlit run app.py")
    print("   3. Upload the car dataset or any CSV file")
    print("   4. Start asking questions in the AI Chat!")
    
    print("\n📁 Files created:")
    print("   - demo_visualizations.png (sample charts)")
    print("   - README.md (complete documentation)")
    print("   - test_setup.py (setup verification)")
    print("   - quick_start.py (easy setup script)")

if __name__ == "__main__":
    main() 