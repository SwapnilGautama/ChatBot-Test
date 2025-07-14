import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt

# 🔑 Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 📄 GitHub raw CSV URL
CSV_URL = "https://raw.githubusercontent.com/SwapnilGautama/CloudInsights/main/SoftwareCompany_2025_Data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    df['Month'] = pd.to_datetime(df['Month'])
    return df

df = load_data()

# Greeting message
st.set_page_config(page_title="Cloud Insights Chatbot", layout="wide")
st.title("💬 Cloud Insights Chatbot")

# 🧠 Show welcome message if no input yet
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.markdown("""
    👋 **Hello! I’m your Cloud Insights Assistant.**

    I analyze software company data including:
    - Revenue and cost by client
    - Fixed vs project-based engagement analysis
    - Onshore vs offshore cost distribution
    - Monthly trends and summaries

    ✅ **Clients available**: BMW, TCS, WIPRO, Capgemini, Accenture, HCL

    ### You can try asking questions like:
    - Revenue and cost breakdown for BMW
    - Compare revenue across clients
    - Monthly trend for Wipro
    - Project vs Fixed comparison

    _Just type your question below to get started!_
    """)
    st.session_state.started = True

user_query = st.text_input("Ask a question:")

# GPT-based interpretation
def ask_gpt(user_query, df_sample):
    prompt = f"""
You are a data analyst. Given a dataset with these columns:
{', '.join(df_sample.columns)}

The user asked: "{user_query}"

Generate a Python pandas code snippet that:
- Handles questions related to:
  - revenue and cost breakdown by client (case-insensitive)
  - compare revenue, cost, and resources across clients
  - show monthly trends for a specific client
  - compare revenue, cost, and resources by engagement type (Fixed_Position vs Project)
- If question is unclear or unsupported, raise ValueError with an informative message.

Return result as:
- result: filtered dataframe
- summary1: grouped revenue summary (optional)
- summary2: grouped cost summary (optional)

Only return valid executable Python code.
Assume df is the dataframe.
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

if user_query:
    try:
        code = ask_gpt(user_query, df.head(3))

        local_vars = {"df": df.copy()}
        clean_code = code.strip().strip("`").replace("python", "").strip()
        exec(clean_code, {}, local_vars)

        result_df = local_vars["result"]

        if result_df.empty:
            st.warning("No matching data found.")
        else:
            if "Type" in result_df.columns:
                agg = result_df.groupby("Type").agg({
                    "Revenue": "sum",
                    "Cost": "sum",
                    "Resources_Total": "sum"
                }).reset_index()

                agg["Revenue ($M)"] = (agg["Revenue"] / 1_000_000).round(2)
                agg["Cost ($M)"] = (agg["Cost"] / 1_000_000).round(2)
                agg.rename(columns={"Resources_Total": "Total Resources"}, inplace=True)

                st.subheader("📌 Key Insights Summary")
                for _, row in agg.iterrows():
                    st.markdown(f"- **The total revenue is ${row['Revenue ($M)']}M and total cost is ${row['Cost ($M)']}M for `{row['Type']}` engagements.**")

                st.subheader("📊 Summary by Type")
                st.dataframe(agg[["Type", "Revenue ($M)", "Cost ($M)", "Total Resources"]], use_container_width=True)

                # 📈 Monthly Trend by Type
                st.subheader("📈 Monthly Revenue & Cost by Type")
                monthly = result_df.groupby(["Month", "Type"]).agg({
                    "Revenue": "sum",
                    "Cost": "sum"
                }).reset_index()

                for metric in ["Revenue", "Cost"]:
                    pivot = monthly.pivot(index="Month", columns="Type", values=metric)
                    st.line_chart(pivot, use_container_width=True, height=300)

            # 📊 Compare by Client
            if "Client" in result_df.columns:
                st.subheader("📊 Compare Revenue & Cost by Client")
                client_summary = result_df.groupby("Client").agg({
                    "Revenue": "sum",
                    "Cost": "sum",
                    "Resources_Total": "sum"
                }).reset_index()

                client_summary["Revenue ($M)"] = (client_summary["Revenue"] / 1_000_000).round(2)
                client_summary["Cost ($M)"] = (client_summary["Cost"] / 1_000_000).round(2)

                st.dataframe(client_summary[["Client", "Revenue ($M)", "Cost ($M)", "Resources_Total"]], use_container_width=True)

                # Charts
                st.bar_chart(client_summary.set_index("Client")[["Revenue ($M)", "Cost ($M)"]])

            # 📋 Raw Results
            st.subheader("📋 Detailed Project Records")
            st.dataframe(result_df, use_container_width=True, height=300)

            # 💬 Follow-up suggestions
            st.markdown("You could also ask:")
            st.markdown("- Compare revenue by type (Fixed vs Project)")
            st.markdown("- Monthly trend for Wipro")
            st.markdown("- Breakdown by location (onshore/offshore)")

    except Exception as e:
        st.error("I couldn't process that request. Please ask about revenue, cost, or client analysis.")
