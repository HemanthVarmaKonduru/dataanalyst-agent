"""
Report Generator Module for Data Analyst Agent
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import jinja2
import os

class ReportGenerator:
    """Generate comprehensive analysis reports in PDF and HTML formats"""
    
    def __init__(self, df, analysis_results, data_quality_score=0):
        self.df = df
        self.analysis_results = analysis_results
        self.data_quality_score = data_quality_score
        self.report_data = {}
        
    def generate_pdf_report(self, filename="data_analysis_report.pdf"):
        """Generate a comprehensive PDF report"""
        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Data Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Report metadata
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns", styles['Normal']))
        story.append(Paragraph(f"Data Quality Score: {self.data_quality_score:.1f}/100", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        summary_text = f"""
        This report provides a comprehensive analysis of the dataset containing {self.df.shape[0]} records 
        and {self.df.shape[1]} variables. The data quality assessment indicates a score of {self.data_quality_score:.1f}/100, 
        suggesting {'excellent' if self.data_quality_score >= 80 else 'good' if self.data_quality_score >= 60 else 'poor'} data quality.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Data Overview
        story.append(Paragraph("Data Overview", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Basic statistics table
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_data = [['Column', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']]
            for col in numeric_cols[:5]:  # Limit to first 5 columns
                stats = self.df[col].describe()
                stats_data.append([
                    col,
                    f"{stats['mean']:.2f}",
                    f"{stats['50%']:.2f}",
                    f"{stats['std']:.2f}",
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}"
                ])
            
            stats_table = Table(stats_data)
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(stats_table)
            story.append(Spacer(1, 20))
        
        # Missing Values Analysis
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            story.append(Paragraph("Missing Values Analysis", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            missing_text = f"Total missing values: {missing_data.sum()} ({missing_data.sum() / (self.df.shape[0] * self.df.shape[1]) * 100:.2f}%)"
            story.append(Paragraph(missing_text, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Missing values table
            missing_table_data = [['Column', 'Missing Count', 'Missing %']]
            for col, count in missing_data[missing_data > 0].items():
                missing_table_data.append([
                    col,
                    str(count),
                    f"{(count / len(self.df)) * 100:.2f}%"
                ])
            
            missing_table = Table(missing_table_data)
            missing_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(missing_table)
            story.append(Spacer(1, 20))
        
        # Key Insights
        if 'insights' in self.analysis_results:
            story.append(Paragraph("Key Insights", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for insight in self.analysis_results['insights'][:5]:  # Limit to 5 insights
                story.append(Paragraph(f"• {insight}", styles['Normal']))
                story.append(Spacer(1, 6))
            story.append(Spacer(1, 20))
        
        # Recommendations
        if 'recommendations' in self.analysis_results:
            story.append(Paragraph("Recommendations", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for rec in self.analysis_results['recommendations']:
                story.append(Paragraph(f"💡 {rec}", styles['Normal']))
                story.append(Spacer(1, 6))
            story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        return filename
    
    def generate_html_report(self, filename="data_analysis_report.html"):
        """Generate a comprehensive HTML report"""
        
        # Create visualizations for the report
        plots = self._create_report_plots()
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Data Analysis Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 3px solid #1f77b4;
                }
                .header h1 {
                    color: #1f77b4;
                    margin: 0;
                    font-size: 2.5em;
                }
                .metadata {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }
                .metadata-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                }
                .metric-card {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                }
                .metric-card h3 {
                    margin: 0;
                    font-size: 1.2em;
                }
                .metric-card p {
                    margin: 5px 0 0 0;
                    font-size: 1.5em;
                    font-weight: bold;
                }
                .section {
                    margin-bottom: 30px;
                }
                .section h2 {
                    color: #2c3e50;
                    border-left: 4px solid #1f77b4;
                    padding-left: 15px;
                    margin-bottom: 20px;
                }
                .insights-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                }
                .insight-card {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #28a745;
                }
                .recommendation-card {
                    background-color: #fff3cd;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #ffc107;
                }
                .plot-container {
                    text-align: center;
                    margin: 20px 0;
                }
                .plot-container img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #1f77b4;
                    color: white;
                }
                tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                .quality-score {
                    font-size: 2em;
                    font-weight: bold;
                    text-align: center;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }
                .quality-excellent {
                    background-color: #d4edda;
                    color: #155724;
                }
                .quality-good {
                    background-color: #fff3cd;
                    color: #856404;
                }
                .quality-poor {
                    background-color: #f8d7da;
                    color: #721c24;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Data Analysis Report</h1>
                    <p>Generated on {{ generation_time }}</p>
                </div>
                
                <div class="metadata">
                    <div class="metadata-grid">
                        <div class="metric-card">
                            <h3>Dataset Size</h3>
                            <p>{{ shape[0] }} × {{ shape[1] }}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Data Quality Score</h3>
                            <p>{{ quality_score }}/100</p>
                        </div>
                        <div class="metric-card">
                            <h3>Numeric Columns</h3>
                            <p>{{ numeric_cols }}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Categorical Columns</h3>
                            <p>{{ categorical_cols }}</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📈 Data Quality Assessment</h2>
                    <div class="quality-score quality-{{ quality_class }}">
                        Data Quality Score: {{ quality_score }}/100
                    </div>
                    <p><strong>Interpretation:</strong> {{ quality_interpretation }}</p>
                </div>
                
                {% if missing_analysis %}
                <div class="section">
                    <h2>🔍 Missing Values Analysis</h2>
                    <p>Total missing values: {{ missing_total }} ({{ missing_percentage }}%)</p>
                    {% if missing_table %}
                    <table>
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Missing Count</th>
                                <th>Missing %</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in missing_table %}
                            <tr>
                                <td>{{ row[0] }}</td>
                                <td>{{ row[1] }}</td>
                                <td>{{ row[2] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                    {% endif %}
                </div>
                {% endif %}
                
                {% if insights %}
                <div class="section">
                    <h2>💡 Key Insights</h2>
                    <div class="insights-grid">
                        {% for insight in insights %}
                        <div class="insight-card">
                            <p>{{ insight }}</p>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
                
                {% if recommendations %}
                <div class="section">
                    <h2>🎯 Recommendations</h2>
                    <div class="insights-grid">
                        {% for rec in recommendations %}
                        <div class="recommendation-card">
                            <p>{{ rec }}</p>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
                
                {% if plots %}
                <div class="section">
                    <h2>📊 Visualizations</h2>
                    {% for plot_name, plot_data in plots.items() %}
                    <div class="plot-container">
                        <h3>{{ plot_name.replace('_', ' ').title() }}</h3>
                        <img src="data:image/png;base64,{{ plot_data }}" alt="{{ plot_name }}">
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                <div class="section">
                    <h2>📋 Data Summary</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Data Type</th>
                                <th>Unique Values</th>
                                <th>Missing Values</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for col in columns %}
                            <tr>
                                <td>{{ col }}</td>
                                <td>{{ dtypes[col] }}</td>
                                <td>{{ unique_counts[col] }}</td>
                                <td>{{ missing_counts[col] }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Prepare data for template
        missing_data = self.df.isnull().sum()
        missing_total = missing_data.sum()
        missing_percentage = (missing_total / (self.df.shape[0] * self.df.shape[1])) * 100
        
        quality_class = 'excellent' if self.data_quality_score >= 80 else 'good' if self.data_quality_score >= 60 else 'poor'
        quality_interpretation = {
            'excellent': 'The dataset has excellent quality with minimal issues.',
            'good': 'The dataset has good quality with some areas for improvement.',
            'poor': 'The dataset has quality issues that should be addressed before analysis.'
        }[quality_class]
        
        template_data = {
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'shape': self.df.shape,
            'quality_score': f"{self.data_quality_score:.1f}",
            'quality_class': quality_class,
            'quality_interpretation': quality_interpretation,
            'numeric_cols': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': len(self.df.select_dtypes(include=['object']).columns),
            'missing_total': missing_total,
            'missing_percentage': f"{missing_percentage:.2f}",
            'missing_analysis': missing_total > 0,
            'missing_table': [[col, count, f"{(count / len(self.df)) * 100:.2f}%"] 
                             for col, count in missing_data[missing_data > 0].items()],
            'insights': self.analysis_results.get('insights', [])[:5],
            'recommendations': self.analysis_results.get('recommendations', [])[:3],
            'plots': plots,
            'columns': self.df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'unique_counts': {col: self.df[col].nunique() for col in self.df.columns},
            'missing_counts': {col: missing_data[col] for col in self.df.columns}
        }
        
        # Render template
        template = jinja2.Template(html_template)
        html_content = template.render(**template_data)
        
        # Save HTML file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename
    
    def _create_report_plots(self):
        """Create plots for the report"""
        plots = {}
        
        # Distribution plots for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, min(3, len(numeric_cols)), figsize=(15, 10))
            if len(numeric_cols) == 1:
                axes = [axes]
            elif len(numeric_cols) <= 3:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:6]):
                if i < len(axes):
                    self.df[col].hist(ax=axes[i], bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plots['distributions'] = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = self.df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Heatmap')
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plots['correlation'] = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
        
        return plots

def generate_analysis_report(df, analysis_results, data_quality_score=0, output_format='html'):
    """Convenience function to generate analysis reports"""
    generator = ReportGenerator(df, analysis_results, data_quality_score)
    
    if output_format.lower() == 'pdf':
        return generator.generate_pdf_report()
    else:
        return generator.generate_html_report() 