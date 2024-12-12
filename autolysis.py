
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt
import httpx
import chardet

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIxZjMwMDAzMjhAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.sdErABQZRIrLR5TaqR1lBDMgCsP2myC7MtqsanZbvQk"

def load_data(file_path):
    """Load CSV data with encoding detection."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    return pd.read_csv(file_path, encoding=encoding)

def analyze_data(df):
    """Perform basic data analysis."""
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict()  # Compute correlation only on numeric columns
    }
    return analysis

def visualize_data(df, file_name):
    """Generate and save visualizations."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        plt.savefig(f'{file_name}_{column}_distribution.png')
        plt.close()

def generate_narrative(analysis):
    """Generate narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = f"Provide a detailed analysis based on the following data summary: {analysis}"
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return "Narrative generation failed due to an error."

def process_file(file_path):
    """Process a single CSV file."""
    df = load_data(file_path)
    analysis = analyze_data(df)
    visualize_data(df, os.path.splitext(os.path.basename(file_path))[0])  # Save with the base filename
    narrative = generate_narrative(analysis)
    report_filename = f'{os.path.splitext(os.path.basename(file_path))[0]}_README.md'
    with open(report_filename, 'w') as f:
        f.write(narrative)

def main():
    # List of CSV files to process
    csv_files = ["goodreads.csv", "happiness.csv", "media.csv"]
    
    # Process each file
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        if os.path.exists(csv_file):
            process_file(csv_file)
            print(f"Finished processing {csv_file}. Results saved.")
        else:
            print(f"File {csv_file} not found. Skipping.")

if __name__ == "__main__":
    main()
