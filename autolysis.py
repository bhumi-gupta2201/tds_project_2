# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "ipykernel",  # Added ipykernel
# ]
# ///

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json

def analyze_data(df):
    """
    Analyze the dataset to compute summary statistics, missing values, and correlations.
    
    Args:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        tuple: summary_stats, missing_values, corr_matrix
    """
    print("Analyzing the dataset...")
    summary_stats = df.describe()
    missing_values = df.isnull().sum()
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
    print("Analysis complete.")
    return summary_stats, missing_values, corr_matrix

def detect_outliers(df):
    """
    Detect outliers using the IQR method for numerical columns.
    
    Args:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        pd.Series: Number of outliers per column.
    """
    print("Detecting outliers...")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric columns for outlier detection.")
        return pd.Series()
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
    print("Outlier detection complete.")
    return outliers

def visualize_data(corr_matrix, outliers, df, output_dir):
    """
    Generate visualizations for correlation matrix, outliers, and data distribution.
    
    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.
        outliers (pd.Series): Detected outliers.
        df (pd.DataFrame): The input dataframe.
        output_dir (str): Directory to save the visualizations.
    
    Returns:
        tuple: Paths to generated plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    heatmap_file, outliers_file, dist_plot_file = None, None, None
    
    # Heatmap for Correlation Matrix
    if not corr_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        heatmap_file = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(heatmap_file)
        plt.close()

    # Outliers Visualization
    if not outliers.empty and outliers.sum() > 0:
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        outliers_file = os.path.join(output_dir, 'outliers.png')
        plt.savefig(outliers_file)
        plt.close()
    
    # Distribution Plot
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        first_column = numeric_columns[0]
        plt.figure(figsize=(10, 6))
        sns.histplot(df[first_column], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {first_column}')
        dist_plot_file = os.path.join(output_dir, f'distribution_{first_column}.png')
        plt.savefig(dist_plot_file)
        plt.close()
    
    print("Visualizations generated.")
    return heatmap_file, outliers_file, dist_plot_file

def create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir):
    """
    Create a README.md file summarizing the analysis results and visualizations.
    
    Args:
        summary_stats (pd.DataFrame): Summary statistics.
        missing_values (pd.Series): Missing values count.
        corr_matrix (pd.DataFrame): Correlation matrix.
        outliers (pd.Series): Detected outliers.
        output_dir (str): Directory to save the README file.
    
    Returns:
        str: Path to README file.
    """
    print("Creating README file...")
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Summary Statistics\n")
        f.write(summary_stats.to_markdown() + "\n\n")
        f.write("## Missing Values\n")
        f.write(missing_values.to_markdown() + "\n\n")
        f.write("## Correlation Matrix\n")
        f.write("![Correlation Matrix](correlation_matrix.png)\n\n")
        f.write("## Outliers Detection\n")
        f.write("![Outliers](outliers.png)\n\n")
        f.write("## Data Distribution\n")
        f.write("![Distribution](distribution_*.png)\n\n")
    print(f"README created at: {readme_path}")
    return readme_path

def generate_story(context):
    """
    Generate a narrative story using the AI proxy API.
    
    Args:
        context (str): Context about data analysis.
    
    Returns:
        str: Generated story or fallback message.
    """
    print("Generating story...")
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": context}],
            "max_tokens": 500,
        }
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Story generation failed: {e}")
        return "Story generation failed."

def main(csv_path):
    print("Starting analysis pipeline...")
    try:
        df = pd.read_csv(csv_path)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    summary_stats, missing_values, corr_matrix = analyze_data(df)
    outliers = detect_outliers(df)

    output_dir = "./output"
    heatmap, outliers_plot, dist_plot = visualize_data(corr_matrix, outliers, df, output_dir)
    readme_file = create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir)

    story_context = f"Summary: {summary_stats}\nMissing: {missing_values}\nOutliers: {outliers}\nCorrelations: {corr_matrix}"
    story = generate_story(story_context)

    with open(readme_file, "a") as f:
        f.write("\n## Generated Story\n")
        f.write(story)

    print("Pipeline complete! Check output folder for results.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])
