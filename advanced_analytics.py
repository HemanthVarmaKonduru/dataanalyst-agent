"""
Advanced Analytics Module for Data Analyst Agent
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    """Advanced analytics class with statistical analysis and insights"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.analysis_results = {}
        
    def statistical_analysis(self, columns=None):
        """Perform comprehensive statistical analysis"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        stats_analysis = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Basic statistics
            stats_analysis[col] = {
                'count': len(col_data),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis(),
                'iqr': col_data.quantile(0.75) - col_data.quantile(0.25),
                'cv': col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
            }
            
            # Normality test
            try:
                shapiro_stat, shapiro_p = stats.shapiro(col_data)
                stats_analysis[col]['normality_test'] = {
                    'shapiro_statistic': shapiro_stat,
                    'shapiro_p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            except:
                stats_analysis[col]['normality_test'] = {
                    'shapiro_statistic': None,
                    'shapiro_p_value': None,
                    'is_normal': None
                }
        
        self.analysis_results['statistical_analysis'] = stats_analysis
        return stats_analysis
    
    def detect_anomalies(self, columns=None, method='isolation_forest', contamination=0.1):
        """Detect anomalies in the data"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        anomalies = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue
            
            if method == 'isolation_forest':
                # Use Isolation Forest
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                anomaly_scores = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                anomalies[col] = {
                    'method': 'isolation_forest',
                    'anomaly_indices': np.where(anomaly_scores == -1)[0],
                    'anomaly_count': np.sum(anomaly_scores == -1),
                    'anomaly_percentage': (np.sum(anomaly_scores == -1) / len(col_data)) * 100
                }
            
            elif method == 'iqr':
                # Use IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomaly_mask = (col_data < lower_bound) | (col_data > upper_bound)
                anomalies[col] = {
                    'method': 'iqr',
                    'anomaly_indices': col_data[anomaly_mask].index.tolist(),
                    'anomaly_count': anomaly_mask.sum(),
                    'anomaly_percentage': (anomaly_mask.sum() / len(col_data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            
            elif method == 'zscore':
                # Use Z-score method
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                anomaly_mask = z_scores > 3
                anomalies[col] = {
                    'method': 'zscore',
                    'anomaly_indices': col_data[anomaly_mask].index.tolist(),
                    'anomaly_count': anomaly_mask.sum(),
                    'anomaly_percentage': (anomaly_mask.sum() / len(col_data)) * 100
                }
        
        self.analysis_results['anomalies'] = anomalies
        return anomalies
    
    def correlation_analysis(self, method='pearson', threshold=0.7):
        """Perform detailed correlation analysis"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr(method=method)
        
        # Find high correlations
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) >= 0.8 else 'moderate'
                    })
        
        # Sort by absolute correlation value
        high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        correlation_analysis = {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations,
            'method': method,
            'threshold': threshold
        }
        
        self.analysis_results['correlation_analysis'] = correlation_analysis
        return correlation_analysis
    
    def clustering_analysis(self, columns=None, n_clusters=3, method='kmeans'):
        """Perform clustering analysis"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        if len(columns) < 2:
            return {}
        
        # Prepare data
        data = self.df[columns].dropna()
        if len(data) == 0:
            return {}
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        if method == 'kmeans':
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels to dataframe
            data_with_clusters = data.copy()
            data_with_clusters['cluster'] = cluster_labels
            
            # Analyze clusters
            cluster_analysis = {}
            for cluster_id in range(n_clusters):
                cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'percentage': (len(cluster_data) / len(data_with_clusters)) * 100,
                    'centroid': cluster_data[columns].mean().to_dict(),
                    'std': cluster_data[columns].std().to_dict()
                }
            
            clustering_results = {
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_analysis': cluster_analysis,
                'inertia': kmeans.inertia_
            }
        
        self.analysis_results['clustering'] = clustering_results
        return clustering_results
    
    def time_series_analysis(self, date_column, value_column, freq='D'):
        """Perform time series analysis"""
        if date_column not in self.df.columns or value_column not in self.df.columns:
            return {}
        
        try:
            # Convert to datetime and set as index
            ts_data = self.df[[date_column, value_column]].copy()
            ts_data[date_column] = pd.to_datetime(ts_data[date_column])
            ts_data = ts_data.set_index(date_column)
            ts_data = ts_data.sort_index()
            
            # Resample to specified frequency
            ts_resampled = ts_data.resample(freq).mean()
            
            # Calculate time series statistics
            ts_stats = {
                'total_periods': len(ts_resampled),
                'start_date': ts_resampled.index.min(),
                'end_date': ts_resampled.index.max(),
                'mean': ts_resampled[value_column].mean(),
                'std': ts_resampled[value_column].std(),
                'trend': 'increasing' if ts_resampled[value_column].iloc[-1] > ts_resampled[value_column].iloc[0] else 'decreasing',
                'seasonality': self._detect_seasonality(ts_resampled[value_column])
            }
            
            # Calculate moving averages
            ts_resampled['ma_7'] = ts_resampled[value_column].rolling(window=7).mean()
            ts_resampled['ma_30'] = ts_resampled[value_column].rolling(window=30).mean()
            
            time_series_analysis = {
                'time_series_data': ts_resampled.to_dict(),
                'statistics': ts_stats,
                'frequency': freq
            }
            
            self.analysis_results['time_series'] = time_series_analysis
            return time_series_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_seasonality(self, series):
        """Detect seasonality in time series data"""
        if len(series) < 12:
            return 'insufficient_data'
        
        # Simple seasonality detection using autocorrelation
        autocorr = series.autocorr(lag=1)
        if abs(autocorr) > 0.7:
            return 'strong'
        elif abs(autocorr) > 0.3:
            return 'moderate'
        else:
            return 'weak'
    
    def generate_insights(self):
        """Generate comprehensive insights from all analyses"""
        insights = []
        
        # Statistical insights
        if 'statistical_analysis' in self.analysis_results:
            stats = self.analysis_results['statistical_analysis']
            for col, col_stats in stats.items():
                if col_stats['skewness'] > 1:
                    insights.append(f"Column '{col}' is highly right-skewed (skewness: {col_stats['skewness']:.2f})")
                elif col_stats['skewness'] < -1:
                    insights.append(f"Column '{col}' is highly left-skewed (skewness: {col_stats['skewness']:.2f})")
                
                if col_stats['cv'] > 1:
                    insights.append(f"Column '{col}' has high variability (CV: {col_stats['cv']:.2f})")
        
        # Anomaly insights
        if 'anomalies' in self.analysis_results:
            anomalies = self.analysis_results['anomalies']
            for col, anomaly_info in anomalies.items():
                if anomaly_info['anomaly_percentage'] > 5:
                    insights.append(f"Column '{col}' has {anomaly_info['anomaly_percentage']:.1f}% anomalies")
        
        # Correlation insights
        if 'correlation_analysis' in self.analysis_results:
            corr_analysis = self.analysis_results['correlation_analysis']
            high_corrs = corr_analysis['high_correlations']
            if len(high_corrs) > 0:
                insights.append(f"Found {len(high_corrs)} strong correlations between variables")
                for corr in high_corrs[:3]:  # Top 3 correlations
                    insights.append(f"Strong correlation ({corr['correlation']:.2f}) between '{corr['variable1']}' and '{corr['variable2']}'")
        
        # Clustering insights
        if 'clustering' in self.analysis_results:
            clustering = self.analysis_results['clustering']
            cluster_analysis = clustering['cluster_analysis']
            insights.append(f"Data can be grouped into {clustering['n_clusters']} distinct clusters")
            
            # Find largest cluster
            largest_cluster = max(cluster_analysis.items(), key=lambda x: x[1]['size'])
            insights.append(f"Largest cluster contains {largest_cluster[1]['percentage']:.1f}% of the data")
        
        return insights
    
    def create_visualizations(self):
        """Create advanced visualizations"""
        plots = {}
        
        # Correlation heatmap
        if 'correlation_analysis' in self.analysis_results:
            corr_matrix = pd.DataFrame(self.analysis_results['correlation_analysis']['correlation_matrix'])
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, square=True)
            ax.set_title('Correlation Heatmap', fontweight='bold')
            plots['correlation_heatmap'] = fig
        
        # Anomaly plots
        if 'anomalies' in self.analysis_results:
            anomalies = self.analysis_results['anomalies']
            for col, anomaly_info in anomalies.items():
                if col in self.df.columns and anomaly_info['anomaly_count'] > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    col_data = self.df[col].dropna()
                    
                    # Plot normal data
                    normal_mask = ~col_data.index.isin(anomaly_info['anomaly_indices'])
                    ax.scatter(col_data[normal_mask].index, col_data[normal_mask], 
                             alpha=0.6, label='Normal', color='blue')
                    
                    # Plot anomalies
                    anomaly_mask = col_data.index.isin(anomaly_info['anomaly_indices'])
                    ax.scatter(col_data[anomaly_mask].index, col_data[anomaly_mask], 
                             alpha=0.8, label='Anomalies', color='red', s=50)
                    
                    ax.set_title(f'Anomaly Detection - {col}', fontweight='bold')
                    ax.set_xlabel('Index')
                    ax.set_ylabel(col)
                    ax.legend()
                    plots[f'anomaly_{col}'] = fig
        
        # Clustering plots
        if 'clustering' in self.analysis_results:
            clustering = self.analysis_results['clustering']
            if 'cluster_labels' in clustering:
                # PCA for dimensionality reduction if needed
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    data = self.df[numeric_cols].dropna()
                    if len(data) > 0:
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(data)
                        
                        if len(numeric_cols) > 2:
                            pca = PCA(n_components=2)
                            reduced_data = pca.fit_transform(scaled_data)
                        else:
                            reduced_data = scaled_data
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                                           c=clustering['cluster_labels'], cmap='viridis')
                        ax.set_title('Clustering Results', fontweight='bold')
                        ax.set_xlabel('Component 1')
                        ax.set_ylabel('Component 2')
                        plt.colorbar(scatter)
                        plots['clustering'] = fig
        
        return plots
    
    def get_analysis_summary(self):
        """Get a summary of all analyses performed"""
        summary = {
            'analyses_performed': list(self.analysis_results.keys()),
            'total_insights': len(self.generate_insights()),
            'data_shape': self.df.shape,
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object']).columns)
        }
        
        return summary

def perform_comprehensive_analysis(df, analyses=None):
    """Perform comprehensive analysis on the dataset"""
    if analyses is None:
        analyses = ['statistical', 'anomalies', 'correlation', 'clustering']
    
    analytics = AdvancedAnalytics(df)
    results = {}
    
    if 'statistical' in analyses:
        results['statistical'] = analytics.statistical_analysis()
    
    if 'anomalies' in analyses:
        results['anomalies'] = analytics.detect_anomalies()
    
    if 'correlation' in analyses:
        results['correlation'] = analytics.correlation_analysis()
    
    if 'clustering' in analyses:
        results['clustering'] = analytics.clustering_analysis()
    
    results['insights'] = analytics.generate_insights()
    results['summary'] = analytics.get_analysis_summary()
    
    return results, analytics 