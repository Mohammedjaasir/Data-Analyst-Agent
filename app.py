# app.py
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- Define Schema (Equivalent to schema.py content) ---
class AnalysisOutput(BaseModel):
    """Output schema for the AI Data Analyst's response."""
    answer: str = Field(description="A concise answer or insight based on the user's query.")
    query_sql: Optional[str] = Field(None, description="The pandas query string used if data was queried or filtered.")
    data_preview: Optional[List[Dict[str, Any]]] = Field(None, description="A preview of the data, if a query was performed and results are concise (e.g., first 5 rows).")
    chart_path: Optional[str] = Field(None, description="The file path to a generated chart image, if a plot was requested.")

# --- Define Tools (Equivalent to tools.py content) ---
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

# Ensure charts and logs directories exist for tools
os.makedirs("charts", exist_ok=True)
os.makedirs("logs", exist_ok=True)

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

def get_row_count(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns the total number of rows in the dataset.
    Use this when the user asks for the total count of records or rows.
    """
    try:
        row_count = len(df)
        return {"answer": f"The dataset contains {row_count} rows."}
    except Exception as e:
        return {"answer": f"Error getting row count: {e}"}

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


# Load environment variables (for GROQ_API_KEY)
load_dotenv()

# --- Configuration ---
# Max number of conversational turns to keep in memory (user + AI message = 1 turn)
MAX_HISTORY_LENGTH = 8

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI Data Analyst (Groq)", layout="wide")
st.title("📊 AI Data Analyst (Powered by Groq)")
st.markdown("Upload your CSV, ask questions, and get insights!")

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "data_columns" not in st.session_state:
    st.session_state.data_columns = []

# --- File Uploader Section ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if st.session_state.df is None or not df.equals(st.session_state.df):
            st.session_state.df = df
            st.session_state.data_columns = df.columns.tolist()
            st.sidebar.success("CSV file uploaded successfully!")
            st.sidebar.subheader("Data Preview:")
            st.sidebar.dataframe(df.head())
            st.sidebar.write(f"Columns: {', '.join(st.session_state.data_columns)}")
            st.session_state.chat_history = []
            st.session_state.agent_executor = None
            st.rerun() # Rerun to re-initialize agent with new df
        else:
            st.sidebar.info("Same CSV file already loaded.")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        st.session_state.df = None
        st.session_state.data_columns = []
        st.session_state.chat_history = []
        st.session_state.agent_executor = None


# --- Agent Setup (Conditional on DataFrame being present) ---
if st.session_state.df is not None and st.session_state.agent_executor is None:
    st.info("Initializing AI Data Analyst...")
    try:
        llm = ChatGroq(model="llama3-8b-8192", temperature=0.2)
        
        # Modified SYSTEM_PROMPT: Removed explicit JSON example to avoid prompt variable conflict
        SYSTEM_PROMPT = f"""
        You are an AI Data Analyst. Your goal is to help users understand their CSV data.
        The current dataset has the following columns: {', '.join(st.session_state.data_columns)}.

        **CRITICAL INSTRUCTION: ALWAYS, WITHOUT EXCEPTION, RETURN YOUR FINAL RESPONSE AS A SINGLE, VALID JSON OBJECT AND NOTHING ELSE.**
        DO NOT include any conversational text, introductions, or conclusions outside the JSON. The JSON should be the ONLY content you return.

        **Your JSON output should aim to include the following keys, if applicable:**
        - "answer": A concise answer or insight based on the user's query.
        - "query_sql": The pandas query string used if data was queried or filtered.
        - "data_preview": A preview of the data (e.g., first 5 rows as a list of dictionaries), if a query was performed and results are concise.
        - "chart_path": The file path to a generated chart image, if a plot was requested.

        ---
        **TOOL USAGE INSTRUCTIONS:**
        You have access to powerful data analysis tools. Always consider using them to answer the user's questions before attempting to generate a direct answer.
        When using a tool, ensure that all arguments match the tool's description precisely.

        **IMPORTANT:** When providing column names to any tool, they MUST exactly match one of the available columns: {', '.join(st.session_state.data_columns)}.
        If a column name is not explicitly mentioned by the user but is clearly implied (e.g., "sales over time" implies a 'date' column and a 'sales' column), you may infer it if unambiguous. If ambiguous, ask for clarification.

        **Workflow:**
        1. Carefully read and understand the user's query.
        2. **Determine if a tool is needed.** If the query asks for calculations (sum, average, row count), filtering data, plotting trends, or saving the conversation, a tool is required.
        3. If a tool is selected, call it with the precise arguments, ensuring column names are correct.
        4. If a tool is executed, process its output and formulate your final `answer` within the JSON.
        5. If the query does not require a tool, provide the `answer` directly in the JSON.
        6. If you cannot answer using the tools or available data, clearly state so in the `answer` field of your JSON output.
        ---
        """

        tools = [
            StructuredTool.from_function(
                func=lambda expression: query_data(st.session_state.df, expression),
                name="query_data",
                description="Returns dataframe rows that match a pandas query string. Use this for filtering data based on conditions. Input is a pandas query expression (e.g., 'revenue > 1000 and region == \"North\"')."
            ),
            StructuredTool.from_function(
                func=lambda column, metric: quick_stats(st.session_state.df, column, metric),
                name="quick_stats",
                description="Compute sum or average of a numeric column. Metric must be 'sum' or 'avg'. Use this for single column aggregations."
            ),
            StructuredTool.from_function(
                func=lambda date_column, value_column: plot_timeseries(st.session_state.df, date_column, value_column),
                name="plot_timeseries",
                description="Generate a time series line plot for a numeric 'value_column' over a 'date_column'. Requires names of 'date_column' and 'value_column' as inputs. Use this when the user asks for trends or plots over time."
            ),
            StructuredTool.from_function(
                func=lambda: get_row_count(st.session_state.df), # No arguments needed for get_row_count
                name="get_row_count",
                description="Returns the total number of rows in the dataset. Use this when the user asks for the total count of records or rows."
            ),
            StructuredTool.from_function(
                func=save_log,
                name="save_log",
                description="Save the entire current conversation transcript to a text file. The input for this tool should be the full chat history string, formatted clearly."
            )
        ]

        # Prompt creation
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{query}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        parser = PydanticOutputParser(pydantic_object=AnalysisOutput)

        agent = create_tool_calling_agent(llm, tools, prompt)
        # verbose=True will print agent's thought process to console, crucial for debugging tool calls
        st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        st.success("AI Data Analyst ready! Ask a question about your data.")

    except Exception as e:
        st.error(f"Error initializing agent: {e}. Please ensure your Groq API key is set in `.env`.")
        st.session_state.agent_executor = None

# --- Chat Interface ---
if st.session_state.df is None:
    st.warning("Please upload a CSV file to begin.")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                # Try to parse the content if it's a JSON string from the agent
                try:
                    # Use the parser to validate and load the content
                    parsed_output = AnalysisOutput.model_validate_json(message.content)
                    st.markdown(f"**Insight:** {parsed_output.answer}")
                    if parsed_output.chart_path:
                        st.image(parsed_output.chart_path, caption="Generated Chart")
                    if parsed_output.data_preview:
                        st.subheader("Data Preview:")
                        st.dataframe(pd.DataFrame(parsed_output.data_preview))
                    if parsed_output.query_sql:
                        st.code(f"Query Used: {parsed_output.query_sql}", language="python")
                except Exception as display_e:
                    st.error(f"Display Error: AI did not return expected JSON format. {display_e}")
                    st.markdown(f"**Raw AI Output:**\n```\n{message.content}\n```")

        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)


    # Chat input
    user_query = st.chat_input("Ask a question about your data...")

    if user_query and st.session_state.agent_executor:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    response = st.session_state.agent_executor.invoke(
                        {
                            "query": user_query,
                            "chat_history": st.session_state.chat_history[-(MAX_HISTORY_LENGTH * 2):],
                        }
                    )
                    agent_output_content = response["output"]

                    # The parser is used here to validate the incoming string from the LLM
                    try:
                        parsed_output = AnalysisOutput.model_validate_json(agent_output_content)
                        st.markdown(f"**Insight:** {parsed_output.answer}")
                        if parsed_output.chart_path:
                            st.image(parsed_output.chart_path, caption="Generated Chart")
                        if parsed_output.data_preview:
                            st.subheader("Data Preview:")
                            st.dataframe(pd.DataFrame(parsed_output.data_preview))
                        if parsed_output.query_sql:
                            st.code(f"Query Used: {parsed_output.query_sql}", language="python")
                        
                        # Store the validated JSON string back into chat history
                        st.session_state.chat_history.append(AIMessage(content=parsed_output.model_dump_json()))

                    except Exception as parse_e:
                        st.error(f"Error parsing agent output: {parse_e}")
                        st.markdown(f"**Raw AI Output:**\n```\n{agent_output_content}\n```")
                        st.session_state.chat_history.append(AIMessage(content=agent_output_content))


                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.session_state.chat_history.append(AIMessage(content=f"Error: {e}"))

    # Clear chat history button
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat History and Reset Data"):
        st.session_state.chat_history = []
        st.session_state.agent_executor = None
        st.session_state.df = None
        st.session_state.data_columns = []
        st.rerun()