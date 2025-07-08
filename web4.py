import streamlit as st
import os
import json
import pandas as pd
import re
import requests

# --- Streamlit Config ---
st.set_page_config(page_title="Medical Report Assistant (Multi-Report)", layout="wide")
st.title("🧠 Multi-Report Medical Assistant")
st.markdown("Ask questions based on *all processed medical reports*. Powered by **Ollama**.")

# --- Load multiple JSON reports ---
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

# --- Load reports ---
data_folder = "output/json"
all_reports = load_all_reports(data_folder)

if not all_reports:
    st.warning("No JSON reports found in output/json")
    st.stop()

# --- Build test summary DataFrame ---
def build_test_dataframe(reports):
    records = []
    for report in reports:
        patient_name = report.get("patient_info", {}).get("name", "Unknown")
        for test in report.get("test_results", []):
            test_name = test.get("test_name", "").lower()
            raw_value = str(test.get("result_value", ""))
            try:
                numeric_value = float(''.join(filter(lambda c: c.isdigit() or c == '.', raw_value)))
            except:
                numeric_value = None
            records.append({
                "patient": patient_name,
                "test": test_name,
                "value": numeric_value,
                "unit": test.get("unit", "")
            })
    return pd.DataFrame(records)

df = build_test_dataframe(all_reports)

# --- Chat UI ---
st.subheader("💬 Ask a Question")
user_query = st.text_input("Type your question across all reports:")

if user_query:
    with st.spinner("🤖 Generating response..."):

        # Handle RBC count queries directly using pandas
        match = re.search(r"RBC count (?:greater|more) than (\d+)", user_query, re.IGNORECASE)
        if match:
            threshold = float(match.group(1))
            filtered = df[(df["test"].str.contains("rbc")) & (df["value"] > threshold)]
            count = filtered["patient"].nunique()
            st.markdown(f"**🧠 Assistant:** {count} patient(s) have an RBC count greater than {threshold}.")
        else:
            try:
                # Convert all reports to JSON string
                json_context = json.dumps(all_reports, indent=2)

                # Build system prompt
                chat_prompt = f"""
You are a helpful assistant analyzing medical reports.
Below is a list of structured JSON documents, each representing a medical report.

JSON REPORT DATA:
{json_context}

USER QUESTION:
{user_query}

You must answer only using the above data. If the answer is not available, say: "This information is not available in the reports."
"""

                # Send to Ollama
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "gemma:2b",  # You can switch to "llama3:8b" or "mistral:7b-instruct"
                        "prompt": chat_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "max_tokens": 1024
                        }
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "").strip()
                    st.markdown(f"**🧠 Assistant:** {answer}")
                else:
                    st.error(f"❌ Ollama returned status code {response.status_code}")

            except Exception as e:
                st.error(f"❌ Error communicating with Ollama: {e}")

# --- Optional: Show analytics and raw JSON ---
if st.checkbox("📊 Show Report Statistics"):
    st.subheader("Test Summary")
    st.dataframe(df.groupby("test")["value"].describe())

if st.checkbox("🧾 Show Raw Reports JSON"):
    st.json(all_reports)
