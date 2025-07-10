import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import io
import requests
from fpdf import FPDF
import base64
import re

openai.api_key = st.secrets["OPENAI_API_KEY"]

CSV_URL = "https://raw.githubusercontent.com/SwapnilGautama/CloudInsights/main/SoftwareCompany_2025_Data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    df['Month'] = pd.to_datetime(df['Month'])
    return df

def ask_gpt(user_query, df_sample):
    lowered = user_query.lower()
    if any(kw in lowered for kw in ["total", "overall", "aggregate", "company-wide"]):
        code = """
result = df.copy()
summary1 = result.groupby("Type")["Revenue"].sum().reset_index()
summary2 = pd.DataFrame({
    "Location": ["Onshore", "Offshore"],
    "Cost": [
        result["Location_Onshore"].sum(),
        result["Location_Offshore"].sum()
    ]
})
"""
    else:
        prompt = f"""
You are a data analyst. Given a dataset with these columns:
{', '.join(df_sample.columns)}

The user asked: "{user_query.lower()}"

Generate a Python pandas code snippet that filters and analyzes the dataset to provide:
1. If a client is mentioned, filter by that client (case-insensitive).
2. Provide:
    - Total revenue and cost
    - Revenue by 'Type' (Fixed_Position vs Project)
    - Cost split by Onshore vs Offshore

Assume the dataframe is called df.
- Use `.str.lower()` for string comparisons
- Return these variables:
    - result → filtered df
    - summary1 → revenue by Type
    - summary2 → cost by Onshore/Offshore

Only return the code.
"""
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        code = response.choices[0].message.content
    return code

def plot_bar(data, title, ylabel):
    fig, ax = plt.subplots(figsize=(6, 4))
    data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Client-wise Summary Report", ln=True, align='C')
    pdf.ln(10)
    col_names = list(df.columns)
    col_width = 190 / len(col_names)
    for col in col_names:
        pdf.cell(col_width, 10, txt=str(col), border=1)
    pdf.ln()
    for _, row in df.iterrows():
        for col in col_names:
            pdf.cell(col_width, 10, txt=str(row[col]), border=1)
        pdf.ln()
    return pdf.output(dest='S').encode('latin1')

def generate_summary(df):
    prompt = f"""
You are a senior business analyst. Given this client-level summary:

{df.to_markdown(index=False)}

Write a concise executive summary (3-4 bullets max) highlighting:
- Top clients by revenue, cost, and resources
- Any major outliers or trends

Avoid redundancy. Be sharp.
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# 🚀 Main App
st.set_page_config(page_title="Cloud Insights Chatbot", page_icon="💬", layout="wide")
st.title("💬 Cloud Insights Chatbot")

df = load_data()

with st.sidebar:
    st.markdown("### 🗞 Clients in Dataset")
    for client in sorted(df["Client"].unique()):
        st.markdown(f"- {client}")

user_query = st.text_input("Ask a question like:", "")

if user_query:
    try:
        greeting = user_query.lower().strip()
        if greeting in ["hello", "hi", "hey", "hi there", "hello there"]:
            st.markdown("👋 Hello! I'm your **Cloud Insights** chatbot.")
            st.markdown("""
I can help you explore and analyze revenue, cost, and resource data from your software projects.

Here’s what I can help with:
- 📊 Revenue and cost by client, project, or type
- 📈 Monthly trends
- 🧾 Full client summary reports with charts
- 🧠 Executive summaries

Try:
- `Show revenue and cost breakdown for BMW`
- `What are the overall totals for cost and revenue?`
- `Client report`
""")

        elif "client report" in greeting:
            st.subheader("📊 Client-wise Summary Table")
            summary = df.groupby("Client").agg({
                "Revenue": "sum",
                "Cost": "sum",
                "Resources_Total": "sum"
            }).reset_index()

            summary["Revenue ($M)"] = (summary["Revenue"] / 1_000_000).round(2)
            summary["Cost ($M)"] = (summary["Cost"] / 1_000_000).round(2)
            summary["Revenue/Resource ($K)"] = (summary["Revenue"] / summary["Resources_Total"] / 1_000).round(2)
            summary["Cost/Resource ($K)"] = (summary["Cost"] / summary["Resources_Total"] / 1_000).round(2)

            total_row = pd.DataFrame({
                "Client": ["Total"],
                "Revenue": [summary["Revenue"].sum()],
                "Cost": [summary["Cost"].sum()],
                "Resources_Total": [summary["Resources_Total"].sum()],
                "Revenue ($M)": [summary["Revenue ($M)"].sum().round(2)],
                "Cost ($M)": [summary["Cost ($M)"].sum().round(2)],
                "Revenue/Resource ($K)": [((summary["Revenue"].sum() / summary["Resources_Total"].sum()) / 1_000).round(2)],
                "Cost/Resource ($K)": [((summary["Cost"].sum() / summary["Resources_Total"].sum()) / 1_000).round(2)]
            })

            final = pd.concat([summary, total_row], ignore_index=True)

            with st.expander("🧠 AI Summary", expanded=True):
                st.markdown(generate_summary(final[["Client", "Revenue ($M)", "Cost ($M)", "Resources_Total"]]))

            st.dataframe(final[["Client", "Revenue ($M)", "Cost ($M)", "Resources_Total", "Revenue/Resource ($K)", "Cost/Resource ($K)"]], use_container_width=True)

            pie_cols = ["Revenue", "Cost", "Resources_Total"]
            labels = final["Client"][:-1]
            for metric in pie_cols:
                fig, ax = plt.subplots()
                ax.pie(summary[metric], labels=labels, autopct='%1.1f%%')
                ax.set_title(f"{metric} by Client")
                st.pyplot(fig)

        else:
            st.markdown("Generating insights...")
            code = ask_gpt(user_query, df.head(3))
            local_vars = {'df': df.copy()}
            clean_code = re.sub(r"```(?:python)?", "", code).strip("`").strip()
            exec(clean_code, {}, local_vars)

            if 'result' in local_vars:
                agg = local_vars['result'].groupby("Type").agg({
                    "Revenue": "sum",
                    "Cost": "sum",
                    "Resources_Total": "sum"
                }).reset_index()

                agg["Revenue ($M)"] = (agg["Revenue"] / 1_000_000).round(2)
                agg["Cost ($M)"] = (agg["Cost"] / 1_000_000).round(2)
                agg.rename(columns={"Resources_Total": "Total Resources"}, inplace=True)

                st.subheader("📌 Summary")
                for _, row in agg.iterrows():
                    st.markdown(f"- **${row['Revenue ($M)']}M Revenue & ${row['Cost ($M)']}M Cost for `{row['Type']}`**")

                st.subheader("📊 Summary by Type")
                st.dataframe(agg[["Type", "Revenue ($M)", "Cost ($M)", "Total Resources"]], use_container_width=True)

                fig, ax1 = plt.subplots(figsize=(6, 4))
                ax2 = ax1.twinx()
                ax1.bar(agg["Type"], agg["Revenue ($M)"], label="Revenue", color="skyblue")
                ax2.plot(agg["Type"], agg["Cost ($M)"], label="Cost", color="red", marker="o")
                ax1.set_ylabel("Revenue ($M)")
                ax2.set_ylabel("Cost ($M)")
                st.pyplot(fig)

                st.subheader("📈 Monthly Revenue vs Cost Trend")
                monthly = local_vars['result'].groupby("Month").agg({"Revenue": "sum", "Cost": "sum"}).sort_index()
                fig, ax1 = plt.subplots(figsize=(8, 4))
                ax2 = ax1.twinx()
                ax1.bar(monthly.index.strftime("%b %Y"), monthly["Revenue"] / 1_000_000, color="lightgreen")
                ax2.plot(monthly.index.strftime("%b %Y"), monthly["Cost"] / 1_000_000, color="orange", marker="o")
                ax1.set_ylabel("Revenue ($M)")
                ax2.set_ylabel("Cost ($M)")
                st.pyplot(fig)

                st.subheader("📋 Raw Data")
                st.dataframe(local_vars['result'], use_container_width=True)

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
