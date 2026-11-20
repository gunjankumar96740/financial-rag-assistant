import streamlit as st
import requests
import uuid

import os

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="Financial RAG",
    page_icon="📊"
)

st.title("Financial RAG Assistant")
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
                    f"{API_BASE_URL}/upload",
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

                st.success("PDF uploaded and indexed successfully!")

                # Clear previous conversation
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())

            except requests.RequestException as e:
                st.error(f"Upload Error: {e}")


st.divider()


# Chat Section
st.subheader("💬 Ask Financial Questions")

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
                    f"{API_BASE_URL}/chat",
                    json={
                        "question": question,
                        "thread_id": st.session_state.thread_id
                    },
                    timeout=180
                )

                response.raise_for_status()

                answer = response.json()["answer"]

                st.write(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except requests.RequestException as e:
                st.error(f"API Error: {e}")
