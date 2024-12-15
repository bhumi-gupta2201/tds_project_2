import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import openai

# Function to analyze the data (basic summary stats, missing values, correlation matrix)
def analyze_data(df):
    """Analyze the dataset and return summary statistics, missing values, and correlation matrix."""
    print("Analyzing the data...")
    summary_stats = df.describe()
    missing_values = df.isnull().sum()
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
    print("Data analysis complete.")
    return summary_stats, missing_values, corr_matrix

# Function to detect outliers using the IQR method
def detect_outliers(df):
    """Detect outliers in the dataset using the IQR method."""
    print("Detecting outliers...")
    df_numeric = df.select_dtypes(include=[np.number])
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
    print("Outliers detection complete.")
    return outliers

# Function to generate visualizations (correlation heatmap, outliers plot, and distribution plot)
def visualize_data(corr_matrix, outliers, df, output_dir):
    """Generate visualizations for the correlation matrix, outliers, and distribution."""
    print("Generating visualizations...")
    # Generate a heatmap for the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    # Check if there are outliers to plot
    if not outliers.empty and outliers.sum() > 0:
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        plt.savefig(os.path.join(output_dir, 'outliers.png'))
        plt.close()
    else:
        print("No outliers detected to visualize.")

    # Generate a distribution plot for the first numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[numeric_columns[0]], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {numeric_columns[0]}')
        plt.savefig(os.path.join(output_dir, 'distribution.png'))
        plt.close()

    print("Visualizations generated.")

# Function to create the README.md with a narrative and visualizations
def create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir):
    """Create a README file with analysis results and visualizations."""
    print("Creating README file...")
    readme_file = os.path.join(output_dir, 'README.md')
    with open(readme_file, 'w') as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Introduction\n")
        f.write("This is an automated analysis of the dataset, providing summary statistics, visualizations, and insights from the data.\n\n")

        # Summary Statistics Section
        f.write("## Summary Statistics\n")
        f.write("The summary statistics of the dataset are as follows:\n")
        f.write("\n| Statistic    | Value |\n")
        f.write("|--------------|-------|\n")
        for column in summary_stats.columns:
            f.write(f"| {column} - Mean | {summary_stats.loc['mean', column]:.2f} |\n")
            f.write(f"| {column} - Std Dev | {summary_stats.loc['std', column]:.2f} |\n")
            f.write(f"| {column} - Min | {summary_stats.loc['min', column]:.2f} |\n")
            f.write(f"| {column} - 25th Percentile | {summary_stats.loc['25%', column]:.2f} |\n")
            f.write(f"| {column} - 50th Percentile (Median) | {summary_stats.loc['50%', column]:.2f} |\n")
            f.write(f"| {column} - 75th Percentile | {summary_stats.loc['75%', column]:.2f} |\n")
            f.write(f"| {column} - Max | {summary_stats.loc['max', column]:.2f} |\n")
            f.write("|--------------|-------|\n")
        
        f.write("\n")

        # Missing Values Section
        f.write("## Missing Values\n")
        f.write("The following columns contain missing values, with their respective counts:\n")
        f.write("\n| Column       | Missing Values Count |\n")
        f.write("|--------------|----------------------|\n")
        for column, count in missing_values.items():
            f.write(f"| {column} | {count} |\n")
        f.write("\n")

        # Outliers Detection Section
        f.write("## Outliers Detection\n")
        f.write("The following columns contain outliers detected using the IQR method (values beyond the typical range):\n")
        f.write("\n| Column       | Outlier Count |\n")
        f.write("|--------------|---------------|\n")
        for column, count in outliers.items():
            f.write(f"| {column} | {count} |\n")
        f.write("\n")

        # Correlation Matrix Section
        f.write("## Correlation Matrix\n")
        f.write("Below is the correlation matrix of numerical features, indicating relationships between different variables:\n\n")
        f.write("![Correlation Matrix](correlation_matrix.png)\n\n")

        # Outliers Visualization Section
        f.write("## Outliers Visualization\n")
        f.write("This chart visualizes the number of outliers detected in each column:\n\n")
        f.write("![Outliers](outliers.png)\n\n")

        # Distribution Plot Section
        f.write("## Distribution of Data\n")
        f.write("Below is the distribution plot of the first numerical column in the dataset:\n\n")
        f.write("![Distribution](distribution.png)\n\n")

        # Conclusion Section
        f.write("## Conclusion\n")
        f.write("The analysis has provided insights into the dataset, including summary statistics, outlier detection, and correlations between key variables.\n")
        f.write("The generated visualizations and statistical insights can help in understanding the patterns and relationships in the data.\n\n")

        print(f"README file created: {readme_file}")
        return readme_file

# Function to generate a detailed story using the OpenAI API
def question_llm(prompt, context):
    """Generate a story using the OpenAI API based on the analysis."""
    print("Generating story using LLM...")
    try:
        token = os.environ["AIPROXY_TOKEN"]
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        full_prompt = f"""
        Based on the following data analysis, please generate a creative and engaging story. The story should include multiple paragraphs, a clear structure with an introduction, body, and conclusion, and should feel like a well-rounded narrative.

        Context:
        {context}

        Data Analysis Prompt:
        {prompt}

        The story should be elaborate and cover the following:
        - An introduction to set the context.
        - A detailed body that expands on the data points and explores their significance.
        - A conclusion that wraps up the analysis and presents any potential outcomes or lessons.
        - Use transitions to connect ideas and keep the narrative flowing smoothly.
        - Format the story with clear paragraphs and structure.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }

        response = requests.post(api_url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            story = response.json()['choices'][0]['message']['content'].strip()
            print("Story generated.")
            return story
        else:
            print(f"Error with request: {response.status_code} - {response.text}")
            return "Failed to generate story."

    except Exception as e:
        print(f"Error: {e}")
        return "Failed to generate story."

# Main function that integrates all the steps
def main(csv_file):
    """Main function to execute the data analysis pipeline."""
    print("Starting the analysis...")

    try:        
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        print("Dataset loaded successfully!")  # Debugging line
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Analyze the data
    summary_stats, missing_values, corr_matrix = analyze_data(df)

    # Debugging print
    print("Summary Stats:")
    print(summary_stats)

    # Detect outliers
    outliers = detect_outliers(df)

    # Debugging print
    print("Outliers detected:")
    print(outliers)

    # Create output directory
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    # Visualize the data
    visualize_data(corr_matrix, outliers, df, output_dir)

    # Generate the story using the LLM
    story = question_llm(
        "Generate a nice and creative story from the analysis", 
        context=f"Dataset Analysis:\nSummary Statistics:\n{summary_stats}\n\nMissing Values:\n{missing_values}\n\nCorrelation Matrix:\n{corr_matrix}\n\nOutliers:\n{outliers}"
    )

    # Create the README file with the analysis and the story
    readme_file = create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir)
    if readme_file:
        try:
            # Append the story to the README.md file
            with open(readme_file, 'a') as f:
                f.write("## Story\n")
                f.write(f"{story}\n")

            print(f"Analysis complete! Results saved in '{output_dir}' directory.")
            print(f"README file: {readme_file}")
        except Exception as e:
            print(f"Error appending story to README.md: {e}")
    else:
        print("Error generating the README.md file.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])
