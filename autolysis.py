#!/usr/bin/env python3
"""
Autolysis: Automated Data Analysis and Narrative Generation Script

This script provides comprehensive data analysis, visualization, and 
narrative generation capabilities for various datasets.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json

class DataAnalyzer:
    """
    A comprehensive data analysis class that provides multiple 
    analytical and visualization capabilities.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize the DataAnalyzer with configurable logging.
        
        Args:
            log_level (int): Logging level, defaults to INFO
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('autolysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, filepath: str, encoding: str = 'ISO-8859-1') -> pd.DataFrame:
        """
        Load data from a CSV file with robust error handling.
        
        Args:
            filepath (str): Path to the CSV file
            encoding (str): File encoding
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            self.logger.info(f"Successfully loaded dataset from {filepath}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

    def analyze_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Perform comprehensive data analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            Tuple of summary statistics, missing values, and correlation matrix
        """
        try:
            # Summary statistics
            summary_stats = df.describe()
            
            # Missing values
            missing_values = df.isnull().sum()
            
            # Correlation matrix (numeric columns only)
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
            
            self.logger.info("Data analysis completed successfully")
            return summary_stats, missing_values, corr_matrix
        
        except Exception as e:
            self.logger.error(f"Data analysis failed: {e}")
            raise

    def detect_outliers(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.Series: Outlier counts per column
        """
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | 
                        (numeric_df > (Q3 + 1.5 * IQR))).sum()
            
            self.logger.info("Outlier detection completed")
            return outliers
        
        except Exception as e:
            self.logger.error(f"Outlier detection failed: {e}")
            raise

    def visualize_data(self, df: pd.DataFrame, corr_matrix: pd.DataFrame, 
                       outliers: pd.Series, output_dir: str = '.') -> Dict[str, Optional[str]]:
        """
        Generate comprehensive data visualizations.
        
        Args:
            df (pd.DataFrame): Input dataframe
            corr_matrix (pd.DataFrame): Correlation matrix
            outliers (pd.Series): Outlier information
            output_dir (str): Directory to save visualizations
        
        Returns:
            Dict of visualization file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        visualizations = {}

        try:
            # Correlation Heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                        linewidths=0.5, fmt=".2f", square=True)
            plt.title('Advanced Correlation Matrix')
            heatmap_path = os.path.join(output_dir, 'correlation_matrix.png')
            plt.tight_layout()
            plt.savefig(heatmap_path)
            plt.close()
            visualizations['correlation'] = heatmap_path

            # Outliers Visualization
            if not outliers.empty and outliers.sum() > 0:
                plt.figure(figsize=(12, 6))
                outliers.plot(kind='bar', color='salmon')
                plt.title('Outlier Distribution Across Features')
                plt.xlabel('Features')
                plt.ylabel('Outlier Count')
                plt.tight_layout()
                outliers_path = os.path.join(output_dir, 'outliers_distribution.png')
                plt.savefig(outliers_path)
                plt.close()
                visualizations['outliers'] = outliers_path

            # Distribution Plot for Numeric Columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                plt.figure(figsize=(15, 5))
                for i, col in enumerate(numeric_cols[:3], 1):  # Plot first 3 numeric columns
                    plt.subplot(1, 3, i)
                    sns.histplot(df[col], kde=True)
                    plt.title(f'Distribution of {col}')
                dist_path = os.path.join(output_dir, 'numeric_distributions.png')
                plt.tight_layout()
                plt.savefig(dist_path)
                plt.close()
                visualizations['distributions'] = dist_path

            self.logger.info("Visualizations generated successfully")
            return visualizations
        
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            return {}

    def generate_narrative(self, analysis_context: Dict[str, Any]) -> str:
        """
        Generate a narrative using an AI proxy.
        
        Args:
            analysis_context (Dict): Context of data analysis
        
        Returns:
            str: Generated narrative
        """
        try:
            token = os.environ.get("AIPROXY_TOKEN")
            if not token:
                raise ValueError("AIPROXY_TOKEN not set")

            api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
            
            prompt = f"""
            Generate a creative and engaging data story based on the following analysis:
            
            Context:
            {json.dumps(analysis_context, indent=2)}
            
            Requirements:
            - Create a compelling narrative
            - Highlight key insights from the data
            - Use a storytelling approach
            - Provide meaningful interpretations
            """

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }

            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a data storyteller."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }

            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()

            story = response.json()['choices'][0]['message']['content'].strip()
            self.logger.info("Narrative generated successfully")
            return story
        
        except Exception as e:
            self.logger.error(f"Narrative generation failed: {e}")
            return "Unable to generate narrative."

    def create_report(self, df: pd.DataFrame, output_dir: str = '.') -> str:
        """
        Create a comprehensive markdown report.
        
        Args:
            df (pd.DataFrame): Input dataframe
            output_dir (str): Directory to save report
        
        Returns:
            str: Path to generated report
        """
        try:
            # Perform analysis steps
            summary_stats, missing_values, corr_matrix = self.analyze_data(df)
            outliers = self.detect_outliers(df)
            visualizations = self.visualize_data(df, corr_matrix, outliers, output_dir)
            
            # Generate narrative
            analysis_context = {
                "summary_stats": summary_stats.to_dict(),
                "missing_values": missing_values.to_dict(),
                "outliers": outliers.to_dict(),
                "visualizations": list(visualizations.keys())
            }
            narrative = self.generate_narrative(analysis_context)

            # Create report
            report_path = os.path.join(output_dir, 'ANALYSIS_REPORT.md')
            with open(report_path, 'w') as f:
                f.write("# Comprehensive Data Analysis Report\n\n")
                
                # Sections
                sections = [
                    ("Summary Statistics", summary_stats),
                    ("Missing Values", missing_values),
                    ("Outliers", outliers)
                ]

                for title, data in sections:
                    f.write(f"## {title}\n")
                    f.write(data.to_markdown() + "\n\n")

                # Visualizations
                f.write("## Visualizations\n")
                for viz_type, path in visualizations.items():
                    f.write(f"### {viz_type.capitalize()} Visualization\n")
                    f.write(f"![{viz_type.capitalize()}]({path})\n\n")

                # Narrative
                f.write("## Data Story\n")
                f.write(narrative)

            self.logger.info(f"Report generated: {report_path}")
            return report_path
        
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return ""

def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Autolysis: Advanced Data Analysis Tool")
    parser.add_argument('dataset', help='Path to the input CSV file')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Set logging level')
    
    args = parser.parse_args()

    # Set logging level
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }

    try:
        analyzer = DataAnalyzer(log_level=log_levels[args.log_level])
        df = analyzer.load_data(args.dataset)
        analyzer.create_report(df)
        print("Analysis complete. Check the generated report and visualizations.")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
