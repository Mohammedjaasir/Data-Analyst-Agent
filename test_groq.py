# test_groq.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# Load environment variables (make sure your .env file is in the same directory)
load_dotenv()

# Define a very simple tool
@tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a given location.
    Input must be the name of a city, e.g., 'London' or 'Salem'.
    """
    if "london" in location.lower():
        return "It's rainy and 15 degrees Celsius in London."
    elif "salem" in location.lower():
        return "It's hot and 35 degrees Celsius in Salem."
    else:
        return "Weather data not available for this location."

# Initialize Groq LLM
# Use temperature=0 for consistent behavior during testing
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

print("--- Testing basic chat completion ---")
try:
    response = llm.invoke("Hello, how are you today?")
    print(f"Basic chat response: {response.content}")
except Exception as e:
    print(f"Error with basic chat completion: {e}")
    print("ACTION REQUIRED: Please check your GROQ_API_KEY in your .env file.")

print("\n--- Testing tool calling capability ---")
try:
    # Bind the tool to the LLM
    llm_with_tools = llm.bind_tools([get_current_weather])
    
    # Try to invoke with a query that should trigger the tool
    ai_message = llm_with_tools.invoke("What's the weather like in London?")
    
    print("\nLLM's response (should contain a tool call or answer):")
    print(ai_message) # This will print the AIMessage object
    
    # Check if a tool call was actually made
    if ai_message.tool_calls:
        print("\nSUCCESS: LLM attempted a tool call!")
        for tc in ai_message.tool_calls:
            print(f"  Tool Name: {tc['name']}")
            print(f"  Tool Arguments: {tc['args']}")
        
        # Optionally execute the tool call
        tool_output = get_current_weather(**ai_message.tool_calls[0]['args'])
        print(f"  Tool Output: {tool_output}")
    else:
        print("\nWARNING: LLM did NOT make a tool call. Raw content:")
        print(ai_message.content)
        print("This might indicate an issue with the model's tool calling capability or API configuration.")

except Exception as e:
    print(f"CRITICAL ERROR with tool calling: {e}")
    print("This means the model cannot perform tool calls in your environment.")