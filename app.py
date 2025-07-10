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
    prompt = f"""
You are a data analyst. Given a dataset with these columns:
{', '.join(df_sample.columns)}

The user asked: "{user_query.lower()}"

Generate a Python pandas code snippet that filters and analyzes the dataset to provide:
1. If the user asks for 'total', 'overall', 'aggregate', or 'company-wide', show revenue and cost across the entire dataset.
2. If a client is mentioned, filter by that client (case-insensitive).
3. Provide:
    - Total revenue and cost
    - Revenue by 'Type' (Fixed_Position vs Project)
    - Cost split by Onshore vs Offshore

Assume the dataframe is called df.
Return the following variables:
- result
- summary1
- summary2

Only return clean Python code.
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def generate_summary(df):
    prompt = f"""
You are a senior business analyst. Given this client-level summary:

{df.to_markdown(index=False)}

Write a concise executive summary (3-4 bullet points max) highlighting:
- Top clients by revenue, cost, and resources
- Notable trends or deviations
Avoid verbose or redundant phrases.
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

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

# App UI
st.set_page_config(page_title="Cloud Insights Chatbot", page_icon="💬", layout="wide")
st.title("💬 Cloud Insights Chatbot")
df = load_data()

with st.sidebar:
    st.markdown("### 🗞 Clients in Dataset")
    for client in sorted(df["Client"].unique()):
        st.markdown(f"- {client}")

user_query = st.text_input("Ask a question like:", "")

if user_query:
    greeting = user_query.lower().strip()
    
    if greeting in ["hello", "hi", "hey", "hi there", "hello there"]:
        st.markdown("👋 Hello! I'm your **Cloud Insights** chatbot.")
        st.markdown("""
I can help you analyze revenue, cost, and resource data across your clients and projects.

### Here’s what you can ask:
- `Show revenue and cost breakdown for BMW`
- `Give me overall totals for cost and revenue`
- `Client report` (full report with summary + visuals)
- `Monthly trend for Mercedes`
- `Compare revenue across clients`

👉 Once you try something, I’ll suggest smart follow-up questions too!
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

        with st.expander("🧠 AI-Generated Business Summary", expanded=True):
            st.markdown(generate_summary(final[["Client", "Revenue ($M)", "Cost ($M)", "Resources_Total"]]))

        st.dataframe(final[["Client", "Revenue ($M)", "Cost ($M)", "Resources_Total", "Revenue/Resource ($K)", "Cost/Resource ($K)"]], use_container_width=True)

        pie_cols = ["Revenue", "Cost", "Resources_Total"]
        labels = final["Client"][:-1]
        figs = []
        for metric in pie_cols:
            fig, ax = plt.subplots()
            ax.pie(summary[metric], labels=labels, autopct='%1.1f%%')
            ax.set_title(f"{metric} by Client")
            figs.append(fig)

        st.subheader("🔹 Distribution by Client")
        col1, col2, col3 = st.columns(3)
        col1.pyplot(figs[0])
        col2.pyplot(figs[1])
        col3.pyplot(figs[2])

        st.markdown("### 📈 Monthly Revenue Trend by Client")
        df["Month_Parsed"] = pd.to_datetime(df["Month"])
        monthly_group = df.groupby(["Client", "Month_Parsed"])["Revenue"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        for client in monthly_group["Client"].unique():
            cdata = monthly_group[monthly_group["Client"] == client]
            ax.plot(cdata["Month_Parsed"], cdata["Revenue"], label=client)
        ax.set_title("Monthly Revenue by Client")
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue")
        ax.legend()
        st.pyplot(fig)

        pdf_bytes = generate_pdf(final[["Client", "Revenue ($M)", "Cost ($M)", "Resources_Total", "Revenue/Resource ($K)", "Cost/Resource ($K)"]])
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="Client_Report.pdf">📄 Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)

    else:
        try:
            st.markdown("Generating insights...")
            code = ask_gpt(user_query, df.head(3))
            local_vars = {'df': df.copy()}
            exec(re.sub(r"```(?:python)?", "", code).strip("`").strip(), {}, local_vars)

            if 'result' not in local_vars:
                raise Exception("No usable result returned from GPT.")

            agg = local_vars['result'].groupby("Type").agg({
                "Revenue": "sum",
                "Cost": "sum",
                "Resources_Total": "sum"
            }).reset_index()

            agg["Revenue ($M)"] = (agg["Revenue"] / 1_000_000).round(2)
            agg["Cost ($M)"] = (agg["Cost"] / 1_000_000).round(2)
            agg.rename(columns={"Resources_Total": "Total Resources"}, inplace=True)

            st.subheader("📌 Key Insights Summary")
            for _, row in agg.iterrows():
                st.markdown(f"- **${row['Revenue ($M)']}M revenue & ${row['Cost ($M)']}M cost for `{row['Type']}` engagements.**")

            st.subheader("📊 Summary by Type (Aggregated)")
            st.dataframe(agg[["Type", "Revenue ($M)", "Cost ($M)", "Total Resources"]], use_container_width=True)

            fig, ax1 = plt.subplots(figsize=(6, 4))
