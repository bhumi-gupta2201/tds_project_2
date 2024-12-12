import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import requests
from sklearn.linear_model import LinearRegression


# Set up the OpenAI API token (replace with your actual token)
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if AIPROXY_TOKEN is None:
    raise ValueError("AIPROXY_TOKEN environment variable not set.")

# Set the proxy URL for OpenAI API
openai.api_base = "https://aiproxy.sanand.workers.dev/openai"

# Set the API key
openai.api_key = AIPROXY_TOKEN


def analyze_data(filename):
    """
    Loads and analyzes the given CSV dataset.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing summary statistics, missing values, correlation matrix, outliers, and trends.
    """

    try:
        df = pd.read_csv(filename, encoding='latin1')  # Using latin1 as fallback encoding
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None, None, None, None

    numeric_df = df.select_dtypes(include='number')
    
    # Summary statistics
    summary_stats = numeric_df.describe().to_string()
    
    # Missing values
    missing_values = df.isnull().sum().to_string()

    # Correlation matrix
    correlation_matrix = numeric_df.corr() if not numeric_df.empty else None

    # Detecting outliers using IQR (Interquartile Range)
    outliers = detect_outliers(numeric_df)

    # Analyzing trends using linear regression
    trends = analyze_trends(df)

    return summary_stats, missing_values, correlation_matrix, outliers, trends


def detect_outliers(numeric_df):
    """
    Detects outliers in the numeric dataframe using IQR (Interquartile Range).

    Args:
        numeric_df (pd.DataFrame): DataFrame containing numeric columns.

    Returns:
        pd.Series: Outliers in each column.
    """
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
    return outliers


def analyze_trends(df):
    """
    Analyzes trends in numeric data using linear regression.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        dict: A dictionary with column names as keys and regression coefficients as values.
    """
    numeric_df = df.select_dtypes(include='number')
    trend_results = {}
    
    # Assuming 'Time' is a column that holds a time-related value for trend analysis
    if 'Time' in numeric_df.columns:
        X = numeric_df[['Time']]
        for column in numeric_df.columns:
            if column != 'Time':
                y = numeric_df[column]
                model = LinearRegression()
                model.fit(X, y)
                trend_results[column] = model.coef_[0]  # Coefficient of the regression line
    return trend_results


def create_story(summary_stats, missing_values, correlation_matrix, outliers, trends, dataset_description):
    """
    Uses LLM to create a narrative about the analysis.

    Args:
        summary_stats (str): Summary statistics string.
        missing_values (str): Missing values string.
        correlation_matrix (pd.DataFrame): The correlation matrix.
        outliers (pd.Series): Outliers in the data.
        trends (dict): The trends in the data.
        dataset_description (str): Brief description of the dataset.

    Returns:
        str: The narrative story created by LLM.
    """

    if correlation_matrix is not None:
        correlation_matrix_markdown = correlation_matrix.to_markdown()
    else:
        correlation_matrix_markdown = "No correlation matrix available."

    prompt = f"""
        Dataset Description:
        {dataset_description}

        **Summary Statistics:**
        {summary_stats}

        **Missing Values:**
        {missing_values}

        **Correlation Matrix:**
        {correlation_matrix_markdown}

        **Outliers:**
        {outliers}

        **Trends (Regression Coefficients):**
        {trends}

        Based on this data, create a summary with the following structure:
        1. A brief description of the dataset (e.g., "This dataset contains ratings of 10,000 books.")
        2. Explanation of the analysis and key insights.
        3. Any surprising or important findings.
        4. Suggestions for real-world actions or implications (e.g., "Focus on improving genres with low ratings.")
    """

    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",  # Model used by the proxy
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        # Make a POST request to the proxy endpoint
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers=headers,
            json=data
        )

        response.raise_for_status()  # Raise error for bad responses
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with OpenAI API: {e}")
        return "Error: Failed to create story using LLM."


def create_folder(dataset_name):
    """
    Creates a folder for each dataset to store the analysis files.

    Args:
        dataset_name (str): Name of the dataset for folder creation.
    """
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)


def main(dataset_filenames):
    """
    Main function to run the analysis and create the narrative for multiple datasets.

    Args:
        dataset_filenames (list): List of paths to CSV files.
    """

    try:
        for dataset_filename in dataset_filenames:
            dataset_name = dataset_filename.split('.')[0]  # Use the base name of the file for the folder

            print(f"Analyzing {dataset_filename}...")

            # Create folder for each dataset
            create_folder(dataset_name)

            summary_stats, missing_values, correlation_matrix, outliers, trends = analyze_data(dataset_filename)

            # Brief description of the dataset (can be customized based on the dataset)
            dataset_description = f"This dataset contains data about {dataset_name}."  # Modify as needed

            # Generate the story
            story = create_story(summary_stats, missing_values, correlation_matrix, outliers, trends, dataset_description)

            # Save the story to README.md
            with open(f'{dataset_name}/README.md', 'w') as f:
                f.write("# Automated Data Analysis\n")
                f.write(f"## Analysis of {dataset_filename}\n")
                f.write(f"### Summary Statistics\n{summary_stats}\n")
                f.write(f"### Missing Values\n{missing_values}\n")
                f.write(f"### Correlation Matrix\n")
                f.write(f"![Correlation Matrix](correlation_matrix.png)\n")
                f.write(f"### Outliers\n")
                f.write(f"![Outliers](outliers.png)\n")
                f.write(f"### Trend Analysis\n")
                f.write(f"![Trends](trends.png)\n")
                f.write(f"### Analysis Story\n{story}\n")

            print(f"Analysis for {dataset_filename} complete.\n")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # List of datasets to process (you can modify this to include your desired CSV files)
    dataset_files = ['goodreads.csv', 'happiness.csv', 'media.csv']

    main(dataset_files)
