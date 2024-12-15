# =============================
# AUTOLYSIS: Automated Data Analysis and Story Generation
# Version: 1.0
# Author: Yashi Gupta
# =============================

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

# =============================
# Function 1: Analyze Data
# - Generate summary statistics
# - Identify missing values
# - Compute correlation matrix
# =============================

def analyze_data(df):
    print("\n=== Step 1: Analyzing Data ===")

    # Generate summary statistics for numeric columns
    print("Generating summary statistics...")
    summary_stats = df.describe()

    # Identify missing values
    print("Checking for missing values...")
    missing_values = df.isnull().sum()

    # Extract correlation matrix for numeric columns
    print("Calculating correlation matrix for numeric columns...")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()

    print("Data analysis complete!\n")
    return summary_stats, missing_values, corr_matrix


# =============================
# Function 2: Detect Outliers
# - Use Interquartile Range (IQR) to identify outliers
# =============================

def detect_outliers(df):
    print("\n=== Step 2: Detecting Outliers ===")
    df_numeric = df.select_dtypes(include=[np.number])

    # Apply IQR for outlier detection
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1

    print("Applying IQR method...")
    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()

    print("Outlier detection complete!\n")
    return outliers


# =============================
# Function 3: Visualize Data
# - Generate correlation heatmap, outlier plot, and distribution plot
# =============================

def visualize_data(corr_matrix, outliers, df, output_dir):
    print("\n=== Step 3: Generating Visualizations ===")
    os.makedirs(output_dir, exist_ok=True)

    # Correlation heatmap
    print("Generating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    heatmap_file = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(heatmap_file)
    plt.close()

    # Outliers bar plot
    if not outliers.empty and outliers.sum() > 0:
        print("Generating outliers visualization...")
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        outliers_file = os.path.join(output_dir, 'outliers.png')
        plt.savefig(outliers_file)
        plt.close()
    else:
        print("No significant outliers detected.")
        outliers_file = None

    # Distribution plot for first numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if numeric_columns.any():
        print("Generating distribution plot...")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[numeric_columns[0]], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {numeric_columns[0]}')
        dist_plot_file = os.path.join(output_dir, 'distribution.png')
        plt.savefig(dist_plot_file)
        plt.close()
    else:
        print("No numeric columns available for distribution plot.")
        dist_plot_file = None

    print("Visualizations saved successfully!\n")
    return heatmap_file, outliers_file, dist_plot_file


# =============================
# Function 4: Generate Story using OpenAI
# =============================

def question_llm(prompt, context):
    print("\n=== Step 4: Generating Data Story ===")
    try:
        # Use proxy API URL
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}"}

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}\n{context}"}
            ],
            "max_tokens": 1000
        }
        response = requests.post(api_url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            print("Story successfully generated!\n")
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return "Failed to generate story."
    except Exception as e:
        print(f"Error: {e}")
        return "Failed to generate story."


# =============================
# Function 5: Create README Report
# - Summarize analysis and attach visuals and story
# =============================

def create_readme(summary_stats, missing_values, outliers, story, output_dir):
    print("\n=== Step 5: Creating Report (README) ===")
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Summary Statistics\n")
        f.write(summary_stats.to_markdown() + "\n\n")

        f.write("## Missing Values\n")
        f.write(missing_values.to_markdown() + "\n\n")

        f.write("## Outliers\n")
        f.write(outliers.to_markdown() + "\n\n")

        f.write("## Story\n")
        f.write(story + "\n")

    print(f"Report saved as {readme_path}\n")


# =============================
# MAIN FUNCTION
# =============================

def main(csv_file):
    print("Starting Automated Analysis...\n")

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(csv_file, encoding='ISO-8859-1')
    print(f"Dataset '{csv_file}' loaded successfully!")

    # Step 1: Data Analysis
    summary_stats, missing_values, corr_matrix = analyze_data(df)

    # Step 2: Outliers Detection
    outliers = detect_outliers(df)

    # Step 3: Visualize Data
    output_dir = "output"
    heatmap_file, outliers_file, dist_plot_file = visualize_data(corr_matrix, outliers, df, output_dir)

    # Step 4: Generate Story
    story_context = f"Summary: {summary_stats}, Missing: {missing_values}, Outliers: {outliers}"
    story = question_llm("Generate an engaging story from the dataset analysis", story_context)

    # Step 5: Create README Report
    create_readme(summary_stats, missing_values, outliers, story, output_dir)

    print("\nAutomated Analysis Completed Successfully!\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset.csv>")
    else:
        main(sys.argv[1])
