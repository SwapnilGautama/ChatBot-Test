import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
import ast

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the dataset
df = pd.read_csv("software_company_data.csv")

# App UI
st.set_page_config(page_title="Cloud Insights Chatbot")
st.title("\U0001F4AC Cloud Insights Chatbot")
st.markdown("""
Ask a question like:
> **Show revenue and cost breakdown for BMW**
""")

# Sidebar
clients = df['Client'].unique().tolist()
st.sidebar.header("\U0001F4CB Clients in Dataset")
for client in sorted(clients):
    st.sidebar.write("-", client)

# Input
user_query = st.text_input("", placeholder="Type your question here...")

# Function to generate code
def generate_code(user_input):
    user_input_lower = user_input.lower()

    # Predefined logic for overall totals
    if any(kw in user_input_lower for kw in ["overall", "total", "company-wide", "aggregate"]):
        return """
# Calculate overall revenue and cost
total_revenue = df['Revenue'].sum()
total_cost = df['Cost'].sum()

# Store in dictionary
result = {
    "Total Revenue ($M)": round(total_revenue, 2),
    "Total Cost ($M)": round(total_cost, 2)
}
        """

    # Otherwise, fall back to GPT
    prompt = f"""
You are a Python data analyst assistant. Generate Python Pandas code to answer the user's question based on the dataframe `df`.
Only use columns: Client, Type, Revenue, Cost, Location_Onshore, Location_Offshore, Resources.

User question: "{user_input}"

Output only the Python code that:
1. Filters by client or type if mentioned.
2. Calculates and summarizes revenue and cost.
3. Groups if needed.
4. Ends with a dictionary or table for Streamlit output.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful Python code generator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

# Process query
if user_query:
    with st.spinner("Generating insights..."):
        try:
            code = generate_code(user_query)
            st.expander("\u25BE Show GPT-Generated Code").code(code, language='python')

            # Prepare execution environment
            local_vars = {'df': df.copy()}
            exec(code, {}, local_vars)

            # Output
            for val in local_vars.values():
                if isinstance(val, (pd.DataFrame, pd.Series, dict)):
                    st.dataframe(val)
                elif isinstance(val, (int, float, str)):
                    st.write(val)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
