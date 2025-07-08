# ✅ Updated Streamlit Code for Multi-Report Question Answering


import streamlit as st
import os
import json
import pandas as pd
import re
from PIL import Image
from medical_ocr import MedicalReportOCR
import tempfile
import requests
from pandasai import SmartDataframe
from ollama_llm import OllamaLLM


# Load multiple report JSONs from a folder
def load_all_reports(json_folder="output/json"):
    all_reports = []
    for filename in os.listdir(json_folder):
        if filename.endswith(".json") and not filename.endswith("_error.json"):
            with open(os.path.join(json_folder, filename), encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    all_reports.append(data)
                except json.JSONDecodeError:
                    continue
    return all_reports

# Convert test results into a DataFrame
def build_test_dataframe(reports):
    records = []
    for report in reports:
        patient_name = report.get("patient_info", {}).get("name", "Unknown")
        for test in report.get("test_results", []):
            test_name = test.get("test_name", "").lower()
            raw_value = str(test.get("result_value", ""))
            try:
                numeric_value = float(''.join(filter(lambda c: c.isdigit() or c=='.', raw_value)))
            except:
                numeric_value = None
            records.append({
                "patient": patient_name,
                "test": test_name,
                "value": numeric_value,
                "unit": test.get("unit", "")
            })
    return pd.DataFrame(records)

# Set Streamlit config
st.set_page_config(page_title="Medical Report Assistant (Multi-Report)", layout="wide")
st.title("🧠 Multi-Report Medical Assistant")
st.markdown("Ask questions based on *all processed medical reports*." )

# Load and cache reports
data_folder = "output/json"
all_reports = load_all_reports(data_folder)

df = build_test_dataframe(all_reports)

# Set your OpenAI API key (or use another LLM provider supported by pandasai)
llm = Ollama(model="gemma:2b", base_url="http://localhost:11434")  # Or use Ollama if supported

sdf = SmartDataframe(df, config={"llm": llm})

user_query = st.text_input("💬 Ask a question (across all reports):")

def count_patients_with_test_above(df, test_keyword, threshold):
    mask = (df['test'].str.contains(test_keyword, case=False, na=False)) & (df['value'] > threshold)
    return df[mask]['patient'].nunique()

def count_patients_with_test_below(df, test_keyword, threshold):
    mask = (df['test'].str.contains(test_keyword, case=False, na=False)) & (df['value'] < threshold)
    return df[mask]['patient'].nunique()

if user_query:
    # Example: "How many patients have an RBC count greater than 1000?"
    m = re.search(r'(?:how many|number of) patients? have an? ([\w\s]+?) count (greater|more|above|less|lower|below) than (\d+(?:\.\d+)?)', user_query.lower())
    if m:
        test_keyword = m.group(1).strip()
        direction = m.group(2)
        threshold = float(m.group(3))
        if direction in ["greater", "more", "above"]:
            count = count_patients_with_test_above(df, test_keyword, threshold)
            st.success(f"🧠 Assistant: {count} patient(s) have a {test_keyword.upper()} count greater than {threshold}.")
        else:
            count = count_patients_with_test_below(df, test_keyword, threshold)
            st.success(f"🧠 Assistant: {count} patient(s) have a {test_keyword.upper()} count less than {threshold}.")
    elif re.search(r"hospital names?", user_query.lower()):
        # Extract unique hospital names from all_reports
        hospital_names = set()
        for report in all_reports:
            hosp = report.get("hospital_info", {}).get("name")
            if hosp:
                hospital_names.add(hosp.strip())
        if hospital_names:
            st.success("🧠 Assistant: The hospital names used in these reports are:")
            st.markdown("\n".join(f"- {name}" for name in sorted(hospital_names)))
        else:
            st.info("No hospital names found in the reports.")
    else:
        with st.spinner("🤖 Thinking..."):
            try:
                answer = sdf.chat(user_query)
                st.success("✅ Answer generated:")
                st.markdown(f"**🧠 Assistant:** {answer}")
            except Exception as e:
                st.error(f"Error: {e}")

# Optional: show test statistics
if st.checkbox("📊 Show Report Statistics"):
    st.subheader("Test Summary")
    st.dataframe(df.groupby("test")["value"].describe())
