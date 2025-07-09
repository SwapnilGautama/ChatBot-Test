# --- Full Working app.py with Overall Totals Fix & Intro Prompt ---
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

Generate a Python pandas code snippet that:
1. If the user asks for 'total', 'overall', 'aggregate', or 'company-wide', show revenue and cost across the **entire dataset**.
2. If a client is mentioned, filter by that client (case-insensitive).
3. Provide:
    - Total revenue and cost
    - Revenue by 'Type' (Fixed_Position vs Project)
    - Cost split by Onshore vs Offshore (Location_Onshore and Location_Offshore)

Assume the dataframe is called df.
Use `.str.lower()` for string comparisons.

Return these 3 variables:
- result → filtered df
- summary1 → revenue by Type
- summary2 → cost by Location split

Return only Python code.
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    code = response.choices[0].message.content
    code = re.sub(r'\breturn\s+.*', '', code)  # 🛠 Fix: avoid 'return' statement crash
    return code

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

# ---------------- MAIN APP ------------------
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
        lower_query = user_query.lower().strip()

        if lower_query in ["hello", "hi", "hey", "hi there", "hello there"]:
            st.markdown("👋 Hello! I'm your Cloud Insights chatbot.")
            st.markdown("""
Here’s what I can help you with:
- 📊 Show revenue and cost breakdowns by client, project, or time
- 🔎 Compare clients by revenue, cost, or resource usage
- 📈 Show trends over time (monthly revenue/cost)
- 🧾 Generate a full client report by typing **client report**

Try asking something like:
- `Show revenue and cost breakdown for BMW`
- `Give me the overall totals`
- `Client report`
            """)

        elif "client report" in lower_query:
            st.subheader("📊 Client-wise Summary Table")
            summary = df.groupby("Client").agg({
                "Revenue": "sum", "Cost": "sum", "Resources_Total": "sum"
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
                "Revenue ($M)": [summary["Revenue ($M)"].sum()],
                "Cost ($M)": [summary["Cost ($M)"].sum()],
                "Revenue/Resource ($K)": [((summary["Revenue"].sum() / summary["Resources_Total"].sum()) / 1_000).round(2)],
                "Cost/Resource ($K)": [((summary["Cost"].sum() / summary["Resources_Total"].sum()) / 1_000).round(2)]
            })

            final = pd.concat([summary, total_row], ignore_index=True)

            with st.expander("🧠 AI-Generated Business Summary", expanded=True):
                st.markdown(generate_summary(final[["Client", "Revenue ($M)", "Cost ($M)", "Resources_Total"]]))

            st.dataframe(final[["Client", "Revenue ($M)", "Cost ($M)", "Resources_Total", "Revenue/Resource ($K)", "Cost/Resource ($K)"]], use_container_width=True)

            st.subheader("🔹 Distribution by Client")
            pie_cols = ["Revenue", "Cost", "Resources_Total"]
            labels = summary["Client"]
            col1, col2, col3 = st.columns(3)
            for i, col in enumerate(pie_cols):
                fig, ax = plt.subplots()
                ax.pie(summary[col], labels=labels, autopct='%1.1f%%')
                ax.set_title(f"{col} by Client")
                [col1, col2, col3][i].pyplot(fig)

            st.markdown("### 📊 Monthly Revenue Trend by Client")
            df["Month_Parsed"] = pd.to_datetime(df["Month"])
            monthly_group = df.groupby(["Client", "Month_Parsed"])["Revenue"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            for client in monthly_group["Client"].unique():
                client_data = monthly_group[monthly_group["Client"] == client]
                ax.plot(client_data["Month_Parsed"], client_data["Revenue"], label=client, marker="o")
            ax.set_title("Revenue by Client (Monthly)")
            ax.legend()
            st.pyplot(fig)

            pdf_bytes = generate_pdf(final[["Client", "Revenue ($M)", "Cost ($M)", "Resources_Total", "Revenue/Resource ($K)", "Cost/Resource ($K)"]])
            b64_pdf = base64.b64encode(pdf_bytes).decode()
            st.markdown(f'<a href="data:application/pdf;base64,{b64_pdf}" download="Client_Report.pdf">📄 Download PDF Report</a>', unsafe_allow_html=True)

        else:
            st.markdown("Generating insights...")
            code = ask_gpt(user_query, df.head(3))
            local_vars = {"df": df.copy()}
            exec(code.strip(), {}, local_vars)

            if 'result' in local_vars:
                result = local_vars['result']
                summary1 = local_vars.get('summary1')
                summary2 = local_vars.get('summary2')

                st.subheader("📌 Key Insights Summary")
                for _, row in summary1.reset_index().iterrows():
                    st.markdown(f"- **Total revenue: ${row['Revenue'] / 1_000_000:.2f}M** for `{row['Type']}`")

                st.subheader("📊 Summary by Type (Aggregated)")
                summary_df = result.groupby("Type").agg({"Revenue": "sum", "Cost": "sum", "Resources_Total": "sum"}).reset_index()
                summary_df["Revenue ($M)"] = (summary_df["Revenue"] / 1_000_000).round(2)
                summary_df["Cost ($M)"] = (summary_df["Cost"] / 1_000_000).round(2)
                summary_df.rename(columns={"Resources_Total": "Total Resources"}, inplace=True)
                st.dataframe(summary_df[["Type", "Revenue ($M)", "Cost ($M)", "Total Resources"]], use_container_width=True)

                st.subheader("📋 Project-wise Data")
                st.dataframe(result, use_container_width=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
