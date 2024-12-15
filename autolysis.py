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
#   "ipykernel",
# ]
# ///

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai

def analyze_data(df):
    """
    Analyze the dataset to compute summary statistics, missing values, and correlations.
    
    Args:
        df (pd.DataFrame): The input dataset.

    Returns:
        tuple: summary_stats (DataFrame), missing_values (Series), corr_matrix (DataFrame).
    """
    print("Analyzing the data...")
    summary_stats = df.describe()
    missing_values = df.isnull().sum()
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
    print("Data analysis complete.")
    return summary_stats, missing_values, corr_matrix

def detect_outliers(df):
    """
    Detect outliers in the dataset using the IQR method.

    Args:
        df (pd.DataFrame): The input dataset.

    Returns:
        pd.Series: A series containing the count of outliers per column.
    """
    print("Detecting outliers...")
    df_numeric = df.select_dtypes(include=[np.number])
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
    print("Outliers detection complete.")
    return outliers

def visualize_data(corr_matrix, outliers, df, output_dir):
    """
    Generate visualizations: correlation heatmap, outlier bar plot, and data distribution plot.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.
        outliers (pd.Series): Outlier counts.
        df (pd.DataFrame): The input dataset.
        output_dir (str): Directory to save visualizations.

    Returns:
        tuple: Paths to the generated visualization files.
    """
    print("Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)

    # Correlation Heatmap
    heatmap_file = os.path.join(output_dir, 'correlation_matrix.png')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(heatmap_file)
    plt.close()

    # Outliers Bar Plot
    outliers_file = None
    if not outliers.empty and outliers.sum() > 0:
        outliers_file = os.path.join(output_dir, 'outliers.png')
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        plt.savefig(outliers_file)
        plt.close()

    # Distribution Plot
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    dist_plot_file = None
    if len(numeric_columns) > 0:
        first_numeric_column = numeric_columns[0]
        dist_plot_file = os.path.join(output_dir, f'distribution.png')
        plt.figure(figsize=(10, 6))
        sns.histplot(df[first_numeric_column], kde=True, bins=30)
        plt.title(f'Distribution of {first_numeric_column}')
        plt.savefig(dist_plot_file)
        plt.close()

    print("Visualizations generated.")
    return heatmap_file, outliers_file, dist_plot_file

def create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir):
    """
    Create a README.md file summarizing the analysis and visualizations.

    Args:
        summary_stats (pd.DataFrame): Summary statistics.
        missing_values (pd.Series): Missing values.
        corr_matrix (pd.DataFrame): Correlation matrix.
        outliers (pd.Series): Outlier counts.
        output_dir (str): Output directory.

    Returns:
        str: Path to the README file.
    """
    print("Creating README file...")
    readme_file = os.path.join(output_dir, 'README.md')
    with open(readme_file, 'w') as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Summary Statistics\n")
        for column in summary_stats.columns:
            f.write(f"**{column}**:\n")
            f.write(f"- Mean: {summary_stats.loc['mean', column]:.2f}\n")
            f.write(f"- Std Dev: {summary_stats.loc['std', column]:.2f}\n")

        f.write("\n## Missing Values\n")
        f.write(missing_values.to_string())

        f.write("\n\n## Correlation Matrix\n")
        f.write("![Correlation Matrix](correlation_matrix.png)\n")

        if outliers.sum() > 0:
            f.write("\n## Outliers\n")
            f.write("![Outliers](outliers.png)\n")

        f.write("\n## Data Distribution\n")
        f.write("![Distribution](distribution.png)\n")

    print("README file created.")
    return readme_file

def question_llm(prompt, context):
    """
    Generate a creative story using OpenAI's proxy API.

    Args:
        prompt (str): Prompt for the LLM.
        context (str): Additional context to send to the LLM.

    Returns:
        str: Generated story.
    """
    print("Generating story using LLM...")
    try:
        token = os.environ["AIPROXY_TOKEN"]
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}\nContext: {context}"}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return "Failed to generate story."
    except Exception as e:
        return f"Error: {e}"

def main(csv_file):
    """
    Main function to orchestrate the entire process.

    Args:
        csv_file (str): Path to the CSV file.
    """
    print("Starting analysis...")
    try:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    summary_stats, missing_values, corr_matrix = analyze_data(df)
    outliers = detect_outliers(df)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    heatmap_file, outliers_file, dist_plot_file = visualize_data(corr_matrix, outliers, df, output_dir)
    readme_file = create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir)

    # Append story to README
    story = question_llm("Generate a story about the dataset analysis", "Dataset summary here.")
    with open(readme_file, 'a') as f:
        f.write("\n## Story\n")
        f.write(f"{story}\n")
    print(f"Analysis complete. Results saved in '{output_dir}'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])
