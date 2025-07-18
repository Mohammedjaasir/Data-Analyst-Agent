# tools.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from typing import Dict, Any

def query_data(df: pd.DataFrame, expression: str) -> Dict[str, Any]:
    """
    Returns dataframe rows that match a pandas query string.
    Use this for filtering data based on conditions.
    Input is a pandas query expression (e.g., 'revenue > 1000 and region == \"North\"').
    """
    try:
        result_df = df.query(expression)
        
        # Convert to list of dicts for JSON serialization
        data_preview = result_df.head(5).to_dict(orient='records') if not result_df.empty else []
        
        response_dict = {
            "answer": f"Successfully filtered data with '{expression}'. Found {len(result_df)} matching rows.",
            "query_sql": expression,
            "data_preview": data_preview
        }
        return response_dict
    except Exception as e:
        return {
            "answer": f"Error querying data with expression '{expression}': {e}",
            "query_sql": expression
        }

def quick_stats(df: pd.DataFrame, column: str, metric: str) -> Dict[str, Any]:
    """
    Compute sum or average of a numeric column. Metric must be 'sum' or 'avg'.
    Use this for single column aggregations.
    """
    if column not in df.columns:
        return {"answer": f"Error: Column '{column}' not found in the data."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"answer": f"Error: Column '{column}' is not numeric. Cannot compute {metric}."}

    try:
        if metric == 'sum':
            result = df[column].sum()
            answer = f"The total sum of '{column}' is: {result}"
        elif metric == 'avg':
            result = df[column].mean()
            answer = f"The average of '{column}' is: {result}"
        else:
            return {"answer": f"Error: Invalid metric '{metric}'. Choose 'sum' or 'avg'."}
        
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"Error computing {metric} for '{column}': {e}"}

def plot_timeseries(df: pd.DataFrame, date_column: str, value_column: str) -> Dict[str, Any]:
    """
    Generate a time series line plot for a numeric 'value_column' over a 'date_column'.
    Requires names of 'date_column' and 'value_column' as inputs.
    Use this when the user asks for trends or plots over time.
    """
    if date_column not in df.columns:
        return {"answer": f"Error: Date column '{date_column}' not found in the data."}
    if value_column not in df.columns:
        return {"answer": f"Error: Value column '{value_column}' not found in the data."}
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return {"answer": f"Error: Value column '{value_column}' is not numeric."}

    try:
        df[date_column] = pd.to_datetime(df[date_column])
        df_sorted = df.sort_values(by=date_column)

        plt.figure(figsize=(10, 6))
        plt.plot(df_sorted[date_column], df_sorted[value_column])
        plt.xlabel(date_column)
        plt.ylabel(value_column)
        plt.title(f'{value_column} Over Time ({date_column})')
        plt.grid(True)
        plt.tight_layout()

        chart_filename = f"charts/timeseries_{value_column}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        plt.savefig(chart_filename)
        plt.close() # Close the plot to free memory

        return {
            "answer": f"Generated time series plot for '{value_column}' over '{date_column}'.",
            "chart_path": chart_filename
        }
    except Exception as e:
        return {"answer": f"Error generating time series plot: {e}"}

def save_log(chat_history_str: str) -> Dict[str, Any]:
    """
    Save the entire current conversation transcript to a text file.
    The input for this tool should be the full chat history string, formatted clearly.
    """
    try:
        log_filename = f"logs/chat_log_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        with open(log_filename, "w") as f:
            f.write("Conversation Log:\n\n")
            f.write(chat_history_str)
        return {"answer": f"Conversation log saved to {log_filename}"}
    except Exception as e:
        return {"answer": f"Error saving log: {e}"}