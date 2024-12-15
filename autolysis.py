import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai

# Function to analyze the data (basic summary stats, missing values, correlation matrix)
def analyze_data(df):
    print("Analyzing the data...")
    summary_stats = df.describe()
    missing_values = df.isnull().sum()
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
    print("Data analysis complete.")
    return summary_stats, missing_values, corr_matrix

# Function to detect outliers using the IQR method
def detect_outliers(df):
    print("Detecting outliers...")
    df_numeric = df.select_dtypes(include=[np.number])
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
    print("Outliers detection complete.")
    return outliers

# Function to generate visualizations
def visualize_data(corr_matrix, outliers, df, output_dir):
    print("Generating visualizations...")
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    # Outliers plot
    if outliers.sum() > 0:
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        plt.savefig(os.path.join(output_dir, 'outliers.png'))
        plt.close()

    # Distribution plot for the first numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        first_numeric_column = numeric_columns[0]
        plt.figure(figsize=(10, 6))
        sns.histplot(df[first_numeric_column], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {first_numeric_column}')
        plt.savefig(os.path.join(output_dir, 'distribution.png'))
        plt.close()

    print("Visualizations generated.")

# Function to create the README.md with a narrative and visualizations
def create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir):
    print("Creating README file...")
    
    readme_file = os.path.join(output_dir, 'README.md')
    
    with open(readme_file, 'w') as f:
        f.write("# Automated Data Analysis Report\n\n")
        
        # Introduction Section
        f.write("## Introduction\n")
        f.write("This is an automated analysis of the dataset.\n\n")
        
        # Summary Statistics Section
        f.write("## Summary Statistics\n")
        f.write(summary_stats.to_markdown() + "\n")  # Using pandas to_markdown for better formatting
        
        # Missing Values Section
        f.write("## Missing Values\n")
        f.write(missing_values.to_markdown() + "\n")
        
        # Outliers Detection Section
        f.write("## Outliers Detection\n")
        f.write(outliers.to_markdown() + "\n")
        
        # Correlation Matrix Section
        f.write("## Correlation Matrix\n")
        f.write("![Correlation Matrix](correlation_matrix.png)\n\n")

        # Outliers Visualization Section
        if outliers.sum() > 0:
            f.write("![Outliers](outliers.png)\n\n")

        # Distribution Plot Section
        f.write("![Distribution](distribution.png)\n\n")

        # Conclusion Section
        f.write("## Conclusion\n")
        f.write("The analysis has provided insights into the dataset.\n")

    print(f"README file created: {readme_file}")
    
# Main function that integrates all the steps
def main(csv_file):
    print("Starting the analysis...")
    
    try:
        df = pd.read_csv(csv_file)
        print("Dataset loaded successfully!")
        
        summary_stats, missing_values, corr_matrix = analyze_data(df)
        
        outliers = detect_outliers(df)
        
        output_dir = "."
        os.makedirs(output_dir, exist_ok=True)

        visualize_data(corr_matrix, outliers, df, output_dir)

        readme_file = create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
        
    main(sys.argv[1])
