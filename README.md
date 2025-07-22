# AI Data Analyst App

## What is this?

This is a web application where you can **upload your CSV data** and then **ask questions about it using natural language**. An AI assistant, powered by Groq, will analyze your data and give you answers, summaries, or even charts.

Think of it as your personal data assistant!

## Sample Image

<img width="960" height="504" alt="Screenshot 2025-07-18 233253" src="https://github.com/user-attachments/assets/6bbdeb39-fdef-4647-8dbd-55913f9b7f5a" />


## How to Set It Up (Installation)

Follow these steps to get the app running on your computer:

1.  **Get the Code:**
    Download or clone this project to your computer. If you're using Git:
    ```bash
    git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git) # Replace with your actual repo URL
    cd your-repo-name
    ```

2.  **Install Required Software:**
    You need Python installed. Then, create a `requirements.txt` file (if you don't have one) listing all the libraries needed:
    ```bash
    # Run this once in your project folder to create requirements.txt
    pip install streamlit pandas matplotlib python-dotenv langchain-core langchain-groq pydantic
    pip freeze > requirements.txt
    ```
    Then, you can install them using:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Get a Groq API Key:**
    * Go to [Groq Console](https://console.groq.com/keys) and create an account to get your free API Key.
    * In your project folder, create a new file called `.env`
    * Inside `.env`, add this line, replacing `your_actual_groq_api_key_here` with your key:
        ```
        GROQ_API_KEY="your_actual_groq_api_key_here"
        ```
    * **Important:** Do not share your `.env` file publicly!

## How to Use It

1.  **Start the App:**
    Open your terminal or command prompt, go into your project folder, and run:
    ```bash
    streamlit run app.py
    ```

2.  **Open in Browser:**
    Your web browser should automatically open to the app (usually at `http://localhost:8501`).

3.  **Upload Your Data:**
    Use the "Choose a CSV file" button on the left sidebar to upload your CSV.

4.  **Ask Questions!**
    Once your data is uploaded, type your questions in the chat box.
    * "How many rows are in this data?"
    * "What's the average 'Sales'?"
    * "Show me the 'Revenue' over 'Date' trend."
    * "Filter where 'City' is 'New York'."

---
