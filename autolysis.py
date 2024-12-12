import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from chardet import detect

def detect_encoding(filename):
    try:
        encoding = detect(open(filename, 'rb').read())['encoding']
        if encoding in ['ascii', 'utf-8']:
            return encoding
        return 'latin1'  # fallback to latin1 for mixed encodings
    except Exception as e:
        print(f"Error detecting encoding: {e}")
        return 'utf-8'  # default to utf-8 in case of error


def analyze_data(filename):
    """
    Loads and analyzes the given CSV dataset.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing summary statistics, missing values, correlation matrix, outliers, and trends.
    """
    try:
        # Detect encoding
        encoding = detect_encoding(filename)
        # Read file with detected encoding
        df = pd.read_csv(filename, encoding=encoding)
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}. Failed to decode {filename}.")
        return None, None, None, None, None
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None, None, None, None

    # Select numeric columns only
    numeric_df = df.select_dtypes(include='number')
    
    # Summary statistics
    summary_stats = numeric_df.describe().to_string() if not numeric_df.empty else "No numeric data found."
    
    # Missing values
    missing_values = df.isnull().sum().to_string()

    # Correlation matrix
    correlation_matrix = numeric_df.corr() if not numeric_df.empty else None

    # Detecting outliers using IQR (Interquartile Range)
    outliers = detect_outliers(numeric_df)

    # Analyzing trends using linear regression
    trends = analyze_trends(df)

    return summary_stats, missing_values, correlation_matrix, outliers, trends

def detect_outliers(df):
    """
    Detects outliers in the numeric columns of a DataFrame using the IQR method.

    Args:
        df (DataFrame): DataFrame with numeric data.

    Returns:
        str: Outliers summary.
    """
    if df.empty:
        return "No numeric data for outlier detection."
    outlier_summary = ""
    for column in df.columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_summary += f"Column '{column}': {len(outliers)} outliers detected.\n"
    return outlier_summary

def analyze_trends(df):
    """
    Placeholder for analyzing trends. Custom logic can be added as required.

    Args:
        df (DataFrame): The DataFrame to analyze.

    Returns:
        str: Trends summary.
    """
    # Placeholder for actual trend analysis
    return "Trend analysis not implemented."

def save_visualizations(df, dataset_name):
    """
    Creates and saves visualizations for a dataset.

    Args:
        df (DataFrame): DataFrame of the dataset.
        dataset_name (str): Name of the dataset.
    """
    if df.empty:
        print(f"No data to visualize for {dataset_name}.")
        return

    # Select numeric columns only for plotting
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        print(f"No numeric data to visualize in {dataset_name}.")
        return

    # Generate pairplot for correlation
    try:
        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        plt.matshow(corr, cmap='coolwarm', fignum=1)
        plt.colorbar()
        plt.title(f"Correlation Matrix for {dataset_name}", pad=15)
        plt.savefig(f"{dataset_name}_correlation_matrix.png")
        plt.close()
        print(f"Correlation matrix visualization saved for {dataset_name}.")
    except Exception as e:
        print(f"Error generating correlation matrix visualization for {dataset_name}: {e}")

def main(dataset_files):
    """
    Main function to analyze and visualize multiple datasets.

    Args:
        dataset_files (list): List of dataset file paths.
    """
    for dataset_filename in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(dataset_filename))[0]
        print(f"Analyzing {dataset_filename}...")

        try:
            summary_stats, missing_values, correlation_matrix, outliers, trends = analyze_data(dataset_filename)
            if summary_stats:
                print(f"Summary Statistics for {dataset_name}:\n{summary_stats}")
            if missing_values:
                print(f"Missing Values for {dataset_name}:\n{missing_values}")
            if correlation_matrix is not None:
                print(f"Correlation Matrix for {dataset_name}:\n{correlation_matrix}")
            if outliers:
                print(f"Outliers for {dataset_name}:\n{outliers}")
            if trends:
                print(f"Trends for {dataset_name}:\n{trends}")

            # Save visualizations
            df = pd.read_csv(dataset_filename, encoding=detect_encoding(dataset_filename))
            save_visualizations(df, dataset_name)

        except Exception as e:
            print(f"An error occurred while analyzing {dataset_filename}: {e}")

if __name__ == "__main__":
    dataset_files = ["goodreads.csv", "happiness.csv", "media.csv"]  # Update with actual file paths
    main(dataset_files)
