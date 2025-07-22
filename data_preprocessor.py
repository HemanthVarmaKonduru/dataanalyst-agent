"""
Advanced Data Preprocessing Module for Data Analyst Agent
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Advanced data preprocessing class with multiple cleaning and transformation options"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.transformations = []
        self.quality_report = {}
        
    def detect_data_types(self):
        """Automatically detect and suggest optimal data types"""
        suggestions = {}
        
        for col in self.df.columns:
            col_data = self.df[col].dropna()
            
            if len(col_data) == 0:
                suggestions[col] = 'object'
                continue
                
            # Check if it's numeric
            try:
                pd.to_numeric(col_data)
                suggestions[col] = 'numeric'
            except:
                # Check if it's datetime
                try:
                    pd.to_datetime(col_data)
                    suggestions[col] = 'datetime'
                except:
                    # Check if it's boolean
                    if col_data.nunique() == 2:
                        suggestions[col] = 'boolean'
                    else:
                        suggestions[col] = 'categorical'
        
        return suggestions
    
    def clean_column_names(self):
        """Clean column names by removing special characters and standardizing format"""
        cleaned_names = {}
        
        for col in self.df.columns:
            # Remove special characters and spaces
            cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
            # Remove multiple underscores
            cleaned = re.sub(r'_+', '_', cleaned)
            # Remove leading/trailing underscores
            cleaned = cleaned.strip('_')
            # Convert to lowercase
            cleaned = cleaned.lower()
            
            if cleaned != col:
                cleaned_names[col] = cleaned
        
        # Rename columns
        if cleaned_names:
            self.df = self.df.rename(columns=cleaned_names)
            self.transformations.append(f"Cleaned column names: {list(cleaned_names.keys())}")
        
        return self.df
    
    def handle_missing_values(self, strategy='auto', columns=None):
        """Handle missing values with various strategies"""
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            missing_count = self.df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            col_type = self.df[col].dtype
            
            if strategy == 'auto':
                if pd.api.types.is_numeric_dtype(col_type):
                    # For numeric columns, use median
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    # For categorical columns, use mode
                    mode_value = self.df[col].mode()
                    if len(mode_value) > 0:
                        self.df[col] = self.df[col].fillna(mode_value[0])
                    else:
                        self.df[col] = self.df[col].fillna('Unknown')
            
            elif strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
            
            elif strategy == 'interpolate':
                if pd.api.types.is_numeric_dtype(col_type):
                    self.df[col] = self.df[col].interpolate()
            
            elif strategy == 'knn':
                if pd.api.types.is_numeric_dtype(col_type):
                    imputer = KNNImputer(n_neighbors=5)
                    self.df[col] = imputer.fit_transform(self.df[[col]])[:, 0]
        
        self.transformations.append(f"Handled missing values using {strategy} strategy")
        return self.df
    
    def remove_duplicates(self, subset=None, keep='first'):
        """Remove duplicate rows"""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_count - len(self.df)
        
        if removed_count > 0:
            self.transformations.append(f"Removed {removed_count} duplicate rows")
        
        return self.df
    
    def handle_outliers(self, method='iqr', columns=None, threshold=1.5):
        """Handle outliers using various methods"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        outliers_removed = 0
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outliers_removed += len(outliers)
                
                # Replace outliers with bounds
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = self.df[z_scores > threshold]
                outliers_removed += len(outliers)
                
                # Replace outliers with mean
                self.df.loc[z_scores > threshold, col] = self.df[col].mean()
        
        if outliers_removed > 0:
            self.transformations.append(f"Handled {outliers_removed} outliers using {method} method")
        
        return self.df
    
    def encode_categorical(self, method='label', columns=None):
        """Encode categorical variables"""
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns
        
        encoders = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'label':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                encoders[col] = le
            
            elif method == 'onehot':
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df = self.df.drop(col, axis=1)
        
        if encoders:
            self.transformations.append(f"Encoded {len(encoders)} categorical columns using {method} encoding")
        
        return self.df, encoders
    
    def scale_numeric(self, method='standard', columns=None):
        """Scale numeric variables"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        scalers = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'standard':
                scaler = StandardScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                scalers[col] = scaler
            
            elif method == 'minmax':
                scaler = MinMaxScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                scalers[col] = scaler
        
        if scalers:
            self.transformations.append(f"Scaled {len(scalers)} numeric columns using {method} scaling")
        
        return self.df, scalers
    
    def extract_datetime_features(self, columns=None):
        """Extract features from datetime columns"""
        if columns is None:
            # Try to detect datetime columns
            columns = []
            for col in self.df.columns:
                try:
                    pd.to_datetime(self.df[col])
                    columns.append(col)
                except:
                    continue
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            try:
                datetime_col = pd.to_datetime(self.df[col])
                
                # Extract various datetime features
                self.df[f'{col}_year'] = datetime_col.dt.year
                self.df[f'{col}_month'] = datetime_col.dt.month
                self.df[f'{col}_day'] = datetime_col.dt.day
                self.df[f'{col}_dayofweek'] = datetime_col.dt.dayofweek
                self.df[f'{col}_quarter'] = datetime_col.dt.quarter
                self.df[f'{col}_is_weekend'] = datetime_col.dt.dayofweek.isin([5, 6]).astype(int)
                
                # Remove original column
                self.df = self.df.drop(col, axis=1)
                
            except Exception as e:
                print(f"Error processing datetime column {col}: {e}")
        
        if columns:
            self.transformations.append(f"Extracted datetime features from {len(columns)} columns")
        
        return self.df
    
    def generate_quality_report(self):
        """Generate a comprehensive data quality report"""
        report = {
            'original_shape': self.original_df.shape,
            'current_shape': self.df.shape,
            'transformations_applied': self.transformations,
            'missing_values': {},
            'data_types': {},
            'unique_values': {},
            'outliers': {},
            'quality_score': 0
        }
        
        # Missing values analysis
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_percentage = (missing_count / len(self.df)) * 100
            report['missing_values'][col] = {
                'count': missing_count,
                'percentage': missing_percentage
            }
        
        # Data types
        for col in self.df.columns:
            report['data_types'][col] = str(self.df[col].dtype)
        
        # Unique values
        for col in self.df.columns:
            report['unique_values'][col] = self.df[col].nunique()
        
        # Outliers analysis for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)]
            report['outliers'][col] = len(outliers)
        
        # Calculate quality score
        score = 100
        
        # Deduct for missing values
        total_missing_percentage = sum([report['missing_values'][col]['percentage'] for col in self.df.columns])
        score -= total_missing_percentage * 0.5
        
        # Deduct for outliers
        total_outliers = sum(report['outliers'].values())
        outlier_percentage = (total_outliers / (len(self.df) * len(numeric_cols))) * 100
        score -= outlier_percentage * 0.3
        
        # Deduct for duplicates
        duplicates = len(self.df) - len(self.df.drop_duplicates())
        duplicate_percentage = (duplicates / len(self.df)) * 100
        score -= duplicate_percentage * 0.2
        
        report['quality_score'] = max(0, min(100, score))
        
        self.quality_report = report
        return report
    
    def auto_preprocess(self, steps=None):
        """Automatically apply common preprocessing steps"""
        if steps is None:
            steps = [
                'clean_names',
                'handle_missing',
                'remove_duplicates',
                'handle_outliers',
                'encode_categorical'
            ]
        
        for step in steps:
            if step == 'clean_names':
                self.clean_column_names()
            elif step == 'handle_missing':
                self.handle_missing_values()
            elif step == 'remove_duplicates':
                self.remove_duplicates()
            elif step == 'handle_outliers':
                self.handle_outliers()
            elif step == 'encode_categorical':
                self.encode_categorical()
            elif step == 'extract_datetime':
                self.extract_datetime_features()
        
        # Generate quality report
        self.generate_quality_report()
        
        return self.df
    
    def get_transformation_summary(self):
        """Get a summary of all transformations applied"""
        return {
            'transformations': self.transformations,
            'quality_report': self.quality_report,
            'shape_change': {
                'original': self.original_df.shape,
                'current': self.df.shape
            }
        }
    
    def reset_to_original(self):
        """Reset the dataframe to its original state"""
        self.df = self.original_df.copy()
        self.transformations = []
        self.quality_report = {}
        return self.df

def create_preprocessing_pipeline(df, steps=None):
    """Convenience function to create and run a preprocessing pipeline"""
    preprocessor = DataPreprocessor(df)
    return preprocessor.auto_preprocess(steps)

def get_data_insights(df):
    """Get quick insights about the data"""
    insights = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'duplicate_percentage': ((len(df) - len(df.drop_duplicates())) / len(df)) * 100,
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'datetime_columns': len([col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])])
    }
    
    return insights 