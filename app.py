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
1. If a client is mentioned, filter by that client (case-insensitive).
2. Provide:
    - Total revenue and cost
    - Revenue by 'Type' (Fixed_Position vs Project)
    - Cost split by Onshore vs Offshore (Location_Onshore and Location_Offshore)

Assume the dataframe is called df.
- Use `.str.lower()` for string comparisons
- Return the following variables:
    - result → filtered df
    - summary1 → revenue by Type
    - summary2 → cost by Onshore/Offshore

Just return executable Python code, no explanation.
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

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

Write a **concise executive summary** (3-4 bullet points max) highlighting:
- Top clients by revenue, cost, and resources
- Notable trends or deviations
Avoid verbose or redundant phrases. Be sharp and analytical.
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# 🚀 Main App UI
st.set_page_config(page_title="Cloud Insights Chatbot", page_icon="💬", layout="wide")
st.title("💬 Cloud Insights Chatbot")

df = load_data()
user_query = st.text_input("Say hello to get started...", "")

if user_query:
    try:
        greeting = user_query.lower().strip()
        if greeting in ["hello", "hi", "hey", "hi there", "hello there"]:
            st.markdown("👋 Hello! I'm your **Cloud Insights** chatbot.")
            st.markdown("""
I help analyze cost, revenue, and resourcing data from your software delivery projects.

I work with data across multiple clients including:
- """ + ", ".join(sorted(df["Client"].unique())) + """

### ✅ You can ask me:
- `Show revenue and cost breakdown for BMW`  
- `Client report`  
- `Compare revenue and cost by type (Fixed vs Project)`  
 
👉 Ask a question to get started, and I’ll also guide you with follow-up questions!
""")

        elif "compare" in greeting and "type" in greeting:
            st.subheader("🔍 Comparison by Type: Fixed Position vs Project")

            by_type = df.groupby(["Month", "Client", "Type"]).agg({
                "Revenue": "sum",
                "Cost": "sum",
                "Resources_Total": "sum"
            }).reset_index()

            st.dataframe(by_type.head(100), use_container_width=True)

            st.markdown("### 📈 Monthly Trend by Type")
            by_month = df.groupby(["Month", "Type"]).agg({
                "Revenue": "sum",
                "Cost": "sum",
                "Resources_Total": "sum"
            }).reset_index()

            fig, ax = plt.subplots(figsize=(10, 5))
            for t in by_month["Type"].unique():
                t_data = by_month[by_month["Type"] == t]
                ax.plot(t_data["Month"], t_data["Revenue"], label=f"{t} Revenue", marker="o")
            ax.set_title("Revenue Trend by Type")
            ax.set_xlabel("Month")
            ax.set_ylabel("Revenue")
            ax.legend()
            st.pyplot(fig)

        elif "client report" in greeting:
            # (same as earlier - no changes)
            ...
        
        else:
            st.markdown("Generating insights...")
            if "total" in greeting or "overall" in greeting or "aggregate" in greeting:
                st.warning("I'm not yet configured to handle overall totals. Please ask a client-specific question like `Show revenue and cost for Infosys`.")
            else:
                code = ask_gpt(user_query, df.head(3))
                local_vars = {'df': df.copy()}
                clean_code = re.sub(r"```(?:python)?", "", code).strip("`").strip()
                exec(clean_code, {}, local_vars)

                if 'result' in local_vars and isinstance(local_vars['result'], pd.DataFrame) and not local_vars['result'].empty:
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
                        st.markdown(f"- **The total revenue is ${row['Revenue ($M)']}M and total cost is ${row['Cost ($M)']}M for `{row['Type']}` engagements.**")

                    st.subheader("📊 Summary by Type (Aggregated)")
                    col1, col2 = st.columns([1.1, 1])
                    with col1:
                        st.dataframe(agg[["Type", "Revenue ($M)", "Cost ($M)", "Total Resources"]], use_container_width=True, height=350)
                    with col2:
                        fig, ax1 = plt.subplots(figsize=(6, 4))
                        ax2 = ax1.twinx()
                        ax1.bar(agg["Type"], agg["Revenue ($M)"], label="Revenue ($M)", color="skyblue")
                        ax2.plot(agg["Type"], agg["Cost ($M)"], label="Cost ($M)", color="red", marker="o")
                        ax1.set_ylabel("Revenue ($M)")
                        ax2.set_ylabel("Cost ($M)")
                        ax1.set_title("Revenue and Cost by Type")
                        ax1.legend(loc="upper left")
                        ax2.legend(loc="upper right")
                        st.pyplot(fig)

                    st.subheader("📈 Monthly Revenue vs Cost Trend")
                    monthly = local_vars['result'].groupby("Month").agg({"Revenue": "sum", "Cost": "sum"}).sort_index()
                    fig, ax1 = plt.subplots(figsize=(8, 4))
                    ax2 = ax1.twinx()
                    ax1.bar(monthly.index.strftime("%b %Y"), monthly["Revenue"] / 1_000_000, label="Revenue ($M)", color="lightgreen")
                    ax2.plot(monthly.index.strftime("%b %Y"), monthly["Cost"] / 1_000_000, label="Cost ($M)", color="orange", marker="o")
                    ax1.set_ylabel("Revenue ($M)")
                    ax2.set_ylabel("Cost ($M)")
                    ax1.set_title("Monthly Revenue vs Cost")
                    ax1.set_xticklabels(monthly.index.strftime("%b %Y"), rotation=45)
                    ax1.legend(loc="upper left")
                    ax2.legend(loc="upper right")
                    st.pyplot(fig)

                    st.subheader("📋 Project-wise and Fixed Position Data")
                    st.dataframe(local_vars['result'], use_container_width=True, height=400)

                    st.markdown("💡 _Try also asking:_")
                    st.markdown("- `Compare revenue and cost by type (Fixed vs Project)`")
                    st.markdown("- `Client report`")

                else:
                    st.warning("🤖 I couldn’t understand that question. Please ask something related to revenue, cost, or client reports.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
