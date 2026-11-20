
import streamlit as st
import requests
import uuid

# Backend URL from Streamlit Cloud Secrets
API_URL = st.secrets["API_URL"].rstrip("/")

st.set_page_config(
    page_title="Financial RAG",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Financial RAG Assistant")
st.caption("Upload financial reports and ask questions about them.")

# Session management
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# PDF Upload Section
st.subheader("📄 Upload Financial Report")

uploaded_file = st.file_uploader(
    "Choose your annual report",
    type=["pdf"]
)

if uploaded_file:
    st.write(f"Selected file: {uploaded_file.name}")

    if st.button("Upload and Process PDF"):

        with st.spinner("Processing financial report..."):

            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    },
                    timeout=600
                )

                response.raise_for_status()
                result = response.json()

                st.success(
                    result.get(
                        "message",
                        "PDF uploaded and indexed successfully!"
                    )
                )

                # Reset conversation after new document upload
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())

            except requests.HTTPError:
                st.error(f"Upload failed: {response.status_code}")
                st.code(response.text)

            except requests.RequestException as e:
                st.error(f"Connection Error: {e}")

            except ValueError:
                st.error("Backend returned an invalid JSON response.")

st.divider()

# Chat Section
st.subheader("💬 Ask Financial Questions")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a financial question...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing financial report..."):

            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question,
                        "thread_id": st.session_state.thread_id
                    },
                    timeout=180
                )

                response.raise_for_status()

                result = response.json()
                answer = result["answer"]

                st.write(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except requests.exceptions.RequestException as e:
                st.error(f"API Error: {e}")

            except (ValueError, KeyError):
                st.error("Invalid response received from the backend.")
