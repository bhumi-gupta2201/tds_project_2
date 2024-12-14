import sys
import os
import httpx
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import re
import pandas as pd
import seaborn as sns
import chardet
import matplotlib.pyplot as plt
from dateutil import parser
import subprocess
import json


# Environment variable for AI Proxy token
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN is not set. Please set it before running the script.")

# Function definitions
def detect_encoding(file_path):
    """
    Detect the encoding of a CSV file.
    """
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        return result['encoding']

def parse_date_with_regex(date_str):
    """
    Parse a date string using regex patterns to identify different date formats.
    """
    if not isinstance(date_str, str):  # Skip non-string values (e.g., NaN, float)
        return date_str  # Return the value as-is

    # Check if the string contains digits, as we expect date-like strings to contain numbers
    if not re.search(r'\d', date_str):
        return np.nan  # If no digits are found, it's not likely a date

    # Define regex patterns for common date formats
    patterns = [
        (r"\d{2}-[A-Za-z]{3}-\d{4}", "%d-%b-%Y"),   # e.g., 15-Nov-2024
        (r"\d{2}-[A-Za-z]{3}-\d{2}", "%d-%b-%y"),   # e.g., 15-Nov-24
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),         # e.g., 2024-11-15
        (r"\d{2}/\d{2}/\d{4}", "%m/%d/%Y"),         # e.g., 11/15/2024
        (r"\d{2}/\d{2}/\d{4}", "%d/%m/%Y"),         # e.g., 15/11/2024
    ]

    # Check which regex pattern matches the date string
    for pattern, date_format in patterns:
        if re.match(pattern, date_str):
            try:
                return pd.to_datetime(date_str, format=date_format, errors='coerce')
            except Exception as e:
                print(f"Error parsing date: {date_str} with format {date_format}. Error: {e}")
                return np.nan

    # If no regex pattern matched, try dateutil parser as a fallback
    try:
        return parser.parse(date_str, fuzzy=True, dayfirst=False)
    except Exception as e:
        print(f"Error parsing date with dateutil: {date_str}. Error: {e}")
        return np.nan

def is_date_column(column):
    """
    Determines whether a column likely contains dates based on column name or content.
    Checks if the column contains date-like strings and returns True if it's likely a date column.
    """
    # Check if the column name contains date-related terms
    if isinstance(column, str):
        if any(keyword in column.lower() for keyword in ['date', 'time', 'timestamp']):
            return True

    # Check the column's content for date-like patterns (e.g., strings with numbers)
    sample_values = column.dropna().head(10)  # Check the first 10 non-NaN values
    date_patterns = [r"\d{2}-[A-Za-z]{3}-\d{2}", r"\d{2}-[A-Za-z]{3}-\d{4}", r"\d{4}-\d{2}-\d{2}", r"\d{2}/\d{2}/\d{4}"]

    for value in sample_values:
        if isinstance(value, str):
            for pattern in date_patterns:
                if re.match(pattern, value):
                    return True
    return False

def read_csv(file_path):
    """
    Read a CSV file with automatic encoding detection and flexible date parsing using regex.
    """
    try:
        print("Detecting file encoding...")
        encoding = detect_encoding(file_path)
        print(f"Detected encoding: {encoding}")

        # Load the CSV file with the detected encoding
        df = pd.read_csv(file_path, encoding=encoding, encoding_errors='replace')

        # Attempt to parse date columns using regex
        for column in df.columns:
            if df[column].dtype == object and is_date_column(df[column]):
                # Only apply date parsing to columns likely containing dates
                print(f"Parsing dates in column: {column}")
                df[column] = df[column].apply(parse_date_with_regex)

        return df

    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)


def perform_advanced_analysis(df):
    analysis = {
        "shape": df.shape,
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe(include="all").to_dict(),
    }
    for column in df.select_dtypes(include=[np.datetime64]).columns:
        df[column] = df[column].dt.strftime('%Y-%m-%d %H:%M:%S')
    outliers = detect_outliers(df)
    if outliers is not None:
        analysis["outliers"] = outliers.value_counts().to_dict()
    return analysis

def detect_outliers(df):
    """Detect outliers using Isolation Forest."""
    numeric_data = df.select_dtypes(include=[np.number])
    if numeric_data.empty:
        return None
    iso = IsolationForest(contamination=0.05, random_state=42)
    numeric_data["outliers"] = iso.fit_predict(numeric_data)
    return numeric_data["outliers"]

def regression_analysis(df):
    """Perform regression analysis on numeric columns."""
    numeric_data = df.select_dtypes(include=[np.number])
    if numeric_data.shape[1] < 2:
        return None
    x = numeric_data.iloc[:, :-1]  # Independent variables
    y = numeric_data.iloc[:, -1]  # Dependent variable
    model = LinearRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    metrics = {
        "MSE": mean_squared_error(y, predictions),
        "R2": r2_score(y, predictions),
        "Coefficients": dict(zip(x.columns, model.coef_)),
    }
    return metrics

def clustering_analysis(df):
    """Perform clustering analysis on numeric columns."""
    numeric_data = df.select_dtypes(include=[np.number]).dropna()
    datetime_columns = df.select_dtypes(include=[np.datetime64])
    for col in datetime_columns:
        numeric_data[col] = (df[col] - df[col].min()).dt.days
    if numeric_data.empty:
        return None, None
    try:
        kmeans = KMeans(n_clusters=3, random_state=42)
        numeric_data['Cluster'] = kmeans.fit_predict(numeric_data)
        return numeric_data['Cluster'], numeric_data.index
    except Exception as e:
        print(f"Error while Clustering: {e}")
        return None, None

def summarize_correlation(df):
    """Summarize key insights from the correlation matrix."""
    numeric_data = df.select_dtypes(include=[np.number])

    if numeric_data.empty:
        return "No numeric data available to compute correlations."

    corr_matrix = numeric_data.corr()

    # Get the highest correlation pairs
    correlations = corr_matrix.unstack().sort_values(ascending=False)

    # Filter out self-correlation (corr with itself)
    correlations = correlations[correlations < 1]

    # Get the top 5 most correlated variable pairs
    top_correlations = correlations.head(5)

    summary = "Top 5 most correlated variables:\n"
    for idx, corr_value in top_correlations.items():
        summary += f"{idx[0]} & {idx[1]}: {corr_value:.2f}\n"

    return summary

def summarize_pairplot(df):
    """Summarize the relationships between numeric variables."""
    numeric_data = df.select_dtypes(include=[np.number])

    if numeric_data.empty:
        return "No numeric data available to analyze in pairplot."

    # Count the number of variables
    num_vars = len(numeric_data.columns)

    summary = f"A pairplot has been created with {num_vars} numeric variables.\n"

    # Describe pairwise relationships (this can be extended to specifics based on domain knowledge)
    if num_vars > 1:
        summary += "The pairplot shows the pairwise relationships between the variables, helping to identify trends, correlations, and possible outliers.\n"
    else:
        summary += "Only one numeric variable is present, so no pairwise relationships could be visualized.\n"

    return summary

def summarize_clustering(df, clusters):
    """Summarize the results of clustering analysis."""
    if clusters is None or len(clusters) == 0:
        return "No clustering results available."

    # Add the cluster labels to the dataframe for analysis
    df['Cluster'] = clusters

    # Count the number of samples in each cluster
    cluster_counts = df['Cluster'].value_counts().sort_values(ascending=False)

    summary = "Clustering results summary:\n"
    for cluster, count in cluster_counts.items():
        summary += f"Cluster {cluster}: {count} samples\n"

    return summary

def generate_summary(df, clusters):
    """Generate a full summary based on the analysis and visualizations."""
    correlation_summary = summarize_correlation(df)
    pairplot_summary = summarize_pairplot(df)
    clustering_summary = summarize_clustering(df, clusters)

    # Combine the summaries into a single narrative
    full_summary = (
        "### Data Analysis Summary\n\n"
        f"#### Correlation Insights:\n{correlation_summary}\n\n"
        f"#### Pairplot Insights:\n{pairplot_summary}\n\n"
        f"#### Clustering Insights:\n{clustering_summary}\n"
    )

    return full_summary


def visualize_advanced(df, output_folder):
    visualizations = []

    # Correlation Heatmap
    numeric_data = df.select_dtypes(include=[np.number]).dropna()
    if not numeric_data.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        heatmap_path = os.path.join(output_folder, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        visualizations.append(heatmap_path)
        plt.close()

    # Pairplot
    pairplot_path = os.path.join(output_folder, "pairplot.png")
    sns.pairplot(numeric_data)
    plt.savefig(pairplot_path)
    visualizations.append(pairplot_path)
    plt.close()

    # Clustering visualization
    if 'Cluster' in df.columns:
        sns.scatterplot(data=df, x=numeric_data.columns[0], y=numeric_data.columns[1], hue='Cluster', palette='viridis')
        clustering_plot_path = os.path.join(output_folder, "clustering_plot.png")
        plt.savefig(clustering_plot_path)
        visualizations.append(clustering_plot_path)
        plt.close()

    # Summary of generated visualizations
    summary = f"Generated {len(visualizations)} visualizations:\n" + "\n".join(visualizations)
    return visualizations, summary


def create_story(analysis, summary):
    """
    Create a story-like narrative that summarizes the findings.
    """
    story = f"### Data Analysis Report\n\n"
    story += f"#### Shape of the Data: {analysis['shape']}\n"
    story += f"#### Summary Statistics: {json.dumps(analysis['summary_statistics'], indent=2)}\n"
    story += f"#### Missing Values: {json.dumps(analysis['missing_values'], indent=2)}\n"
    story += f"#### Outliers detected: {analysis.get('outliers', 'None')}\n\n"
    story += f"#### Data Correlation Summary:\n{summary['correlation']}\n\n"
    story += f"#### Pairplot Insights:\n{summary['pairplot']}\n\n"
    story += f"#### Clustering Summary:\n{summary['clustering']}\n"
    return story


def save_results(analysis, visualizations, summary, output_folder):
    """Save analysis results, visualizations, and the generated story to files."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save analysis results
    with open(os.path.join(output_folder, 'analysis_results.json'), 'w') as f:
        json.dump(analysis, f)

    # Save visualizations paths
    with open(os.path.join(output_folder, 'visualizations.json'), 'w') as f:
        json.dump(visualizations, f)

    # Save the narrative story
    with open(os.path.join(output_folder, 'analysis_story.txt'), 'w') as f:
        f.write(summary)


def main():
    """
    Main function that orchestrates the script execution.
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"Processing file: {file_path}")

    # Read the CSV file
    df = read_csv(file_path)

    # Perform advanced analysis
    analysis = perform_advanced_analysis(df)

    # Visualize the data
    output_folder = "analysis_results"
    os.makedirs(output_folder, exist_ok=True)
    visualizations, summary = visualize_advanced(df, output_folder)

    # Generate narrative
    story = create_story(analysis, summary)

    # Save the results
    save_results(analysis, visualizations, story, output_folder)

    print("Analysis complete. Results saved to 'analysis_results' folder.")

if __name__ == "__main__":
    main()
