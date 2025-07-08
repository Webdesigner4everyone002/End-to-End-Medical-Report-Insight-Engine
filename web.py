import streamlit as st
import tempfile
import os
import json
from PIL import Image
import requests
from medical_ocr import MedicalReportOCR  # Your OCR logic from earlier

# Config
st.set_page_config(page_title="Medical Report Chatbot", layout="wide")
ocr = MedicalReportOCR()

st.title("🧠 Medical Report AI Assistant (OCR + Ollama)")
st.markdown("Upload a scanned *medical report image*, then ask anything about it!")

# Image uploader
uploaded_file = st.file_uploader("📤 Upload Image (JPG, PNG, TIFF)", type=["jpg", "jpeg", "png", "tiff", "tif"])

# Initialize session state
if "json_data" not in st.session_state:
    st.session_state.json_data = None

if uploaded_file:
    with st.spinner("🔍 Extracting data from image..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            image_bytes = uploaded_file.read()
            tmp_file.write(image_bytes)
            temp_image_path = tmp_file.name

        # Show image preview
        st.image(Image.open(temp_image_path), caption="Uploaded Report", use_column_width=True)

        # Process image using OCR + Ollama
        result = ocr.process_image(temp_image_path)
        os.remove(temp_image_path)

        if result['success']:
            st.success("✅ Successfully processed the image!")
            st.session_state.json_data = result['structured_json']

            # 💾 Save JSON to local file (for dev/debug)
            with open("last_result.json", "w", encoding="utf-8") as f:
                json.dump(st.session_state.json_data, f, indent=2, ensure_ascii=False)

            # 🧾 Optional: show raw JSON inside expandable developer section
            with st.expander("🧾 Show Extracted JSON (Developer Only)"):
                st.json(st.session_state.json_data)

            # ⬇ Optional: download button
            json_str = json.dumps(st.session_state.json_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="⬇ Download Extracted JSON",
                data=json_str,
                file_name="medical_report.json",
                mime="application/json"
            )

        else:
            st.error(f"❌ Failed to process image: {result.get('error', 'Unknown error')}")
            if 'ollama_raw_response' in result:
                with st.expander("🧪 Ollama Raw Response"):
                    st.text(result['ollama_raw_response'])
            st.stop()

# 💬 Chatbot section
if st.session_state.json_data:
    st.subheader("💬 Chat with the Medical Report")

    user_query = st.text_input("Ask a question about the report...")

    if user_query:
        with st.spinner("🤖 Generating response..."):
            # Build the prompt
            json_context = json.dumps(st.session_state.json_data, indent=2)
            chat_prompt = f"""
You are a helpful assistant for analyzing medical reports.
You are given the extracted structured information from a medical report in JSON format.

JSON DATA:
{json_context}

USER QUESTION:
{user_query}

Answer clearly and accurately based only on the provided JSON data.
If the answer is not present in the data, say "This information is not available in the report."
"""

            # Call Ollama
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": ocr.model_name,
                        "prompt": chat_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "max_tokens": 512
                        }
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "").strip()
                    st.markdown(f"🧠 Assistant:** {answer}")
                else:
                    st.error("❌ Failed to get response from Ollama")

            except Exception as e:
                st.error(f"❌ Error communicating with Ollama: {e}")