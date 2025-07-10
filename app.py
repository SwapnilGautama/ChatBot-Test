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

# Start blank and guide the user
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
- `Monthly trend for Infosys`  
- `Compare revenue across clients`  

👉 Ask a question to get started, and I’ll also guide you with follow-up questions!
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

            st.markdown("### 📊 Monthly Revenue Trend by Client")
            df["Month_Parsed"] = pd.to_datetime(df["Month"])
            monthly_group = df.groupby(["Client", "Month_Parsed"])["Revenue"].sum().reset_index()

            fig, ax = plt.subplots(figsize=(10, 5))
            clients = monthly_group["Client"].unique()
            colors = plt.cm.tab10.colors

            for idx, client in enumerate(clients):
                client_data = monthly_group[monthly_group["Client"] == client]
                ax.plot(
                    client_data["Month_Parsed"],
                    client_data["Revenue"],
                    label=client,
                    linewidth=2.5,
                    marker="o",
                    markersize=5,
                    color=colors[idx % len(colors)],
                )

            ax.set_title("Revenue by Client (Monthly)", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Revenue ($)", fontsize=12)
            ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
            ax.legend(title="Client", fontsize=9, title_fontsize=10, loc="upper right")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            pdf_bytes = generate_pdf(final[["Client", "Revenue ($M)", "Cost ($M)", "Resources_Total", "Revenue/Resource ($K)", "Cost/Resource ($K)"]])
            b64_pdf = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="Client_Report.pdf">📄 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

        else:
            st.markdown("Generating insights...")
            if "total" in greeting or "overall" in greeting or "aggregate" in greeting:
                st.warning("I'm not yet configured to handle overall totals. Please ask a client-specific question like `Show revenue and cost for Infosys`.")
            else:
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
                    st.markdown("- `Monthly trend for Infosys`")
                    st.markdown("- `Compare revenue across clients`")
                    st.markdown("- `Client report`")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
