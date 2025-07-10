import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

st.set_page_config(page_title="Cloud Insights Chatbot", layout="wide")

# Hide Streamlit sidebar
hide_sidebar_style = """
    <style>
    [data-testid="stSidebar"] {display: none;}
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# Chatbot UI
st.markdown("## 💬 Cloud Insights Chatbot")
user_input = st.text_input("Hi There:", placeholder="Ask a question like: Show revenue and cost breakdown for BMW")

def show_intro():
    st.markdown("👋 Hello! I'm your **Cloud Insights** chatbot.")
    st.markdown("I can help analyze your software company’s data around revenue, cost, and resources.")
    st.markdown("**Try asking:**")
    st.markdown("- `Show revenue and cost breakdown for BMW`")
    st.markdown("- `Client report`")
    st.markdown("👉 After each question, I’ll suggest what to explore next!")

def summarize_breakdown(client):
    client_df = df[df['Client'].str.lower() == client.lower()]
    if client_df.empty:
        return f"No data available for {client}.", None

    summary = client_df.groupby("Type")[["Revenue ($M)", "Cost ($M)", "Resources"]].sum().reset_index()
    summary.rename(columns={"Resources": "Total Resources"}, inplace=True)

    total_revenue = summary["Revenue ($M)"].sum()
    total_cost = summary["Cost ($M)"].sum()

    text = f"### 🔍 Revenue and Cost Breakdown for {client}\n"
    text += f"**Total Revenue:** ${total_revenue:.2f}M  \n"
    text += f"**Total Cost:** ${total_cost:.2f}M  \n\n"
    text += "Here's the breakdown by type:\n"

    fig, ax1 = plt.subplots(figsize=(6, 4))
    summary.plot(kind='bar', x='Type', y=['Revenue ($M)', 'Cost ($M)'], ax=ax1)
    ax1.set_ylabel("Amount ($M)")
    ax1.set_title(f"{client} - Revenue and Cost by Type")

    return text, (summary, fig)

def show_client_report():
    grouped = df.groupby("Client")[["Revenue ($M)", "Cost ($M)", "Resources"]].sum().reset_index()
    grouped.rename(columns={"Resources": "Total Resources"}, inplace=True)

    summary_text = "### 📊 Client Report\n"
    summary_text += "Here's a summary across all clients with total revenue, cost, and resources used.\n"

    return summary_text, grouped

def get_follow_up_suggestions():
    return [
        "Try asking: `Show revenue and cost breakdown for Porsche`",
        "Or: `Client report`",
    ]

def interpret_query(query):
    query = query.lower()

    # Revenue breakdown for a specific client
    match = re.search(r"revenue.*cost.*(?:for|of)?\s*(\w+)", query)
    if match:
        return "breakdown", match.group(1)

    if "client report" in query:
        return "client_report", None

    if query.strip() in ["hi", "hello", "hey"]:
        return "intro", None

    return "unknown", None

# Process user input
if user_input:
    action, value = interpret_query(user_input)

    if action == "intro":
        show_intro()

    elif action == "breakdown":
        msg, viz = summarize_breakdown(value)
        st.markdown(msg)
        if viz:
            st.dataframe(viz[0])
            st.pyplot(viz[1])
        for f in get_follow_up_suggestions():
            st.markdown(f"➡️ {f}")

    elif action == "client_report":
        msg, table = show_client_report()
        st.markdown(msg)
        st.dataframe(table)
        for f in get_follow_up_suggestions():
            st.markdown(f"➡️ {f}")

    else:
        st.warning("⚠️ Sorry, I’m not yet trained to answer that question. Please try one of the suggested formats.")
else:
    show_intro()
