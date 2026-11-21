
import streamlit as st
import requests
import uuid

# API Configuration
API_URL = st.secrets["API_URL"].rstrip("/")
API_KEY = st.secrets["API_KEY"]

HEADERS = {
    "X-API-Key": API_KEY
}

st.set_page_config(
    page_title="Financial RAG",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Financial RAG Assistant")
st.caption("Upload financial reports and ask questions about them.")

# Session Management
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# PDF Upload
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
                    headers=HEADERS,
                    timeout=600
                )

                response.raise_for_status()

                result = response.json()

                st.success(result.get("message", "PDF processed successfully!"))

                # Reset conversation
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())

            except requests.HTTPError:
                st.error(f"Upload failed: {response.status_code}")
                st.code(response.text)

            except requests.RequestException as e:
                st.error(f"Connection Error: {e}")

            except ValueError:
                st.error("Invalid response from backend.")


st.divider()


# Chat Section
st.subheader("💬 Ask Financial Questions")

# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input("Ask a financial question...")

if question:

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
                    headers=HEADERS,
                    timeout=180
                )

                response.raise_for_status()

                result = response.json()
                answer = result["answer"]

                st.write(answer)

                # Save successful conversation
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except requests.HTTPError:
                st.error(f"API Error: {response.status_code}")
                st.code(response.text)

            except requests.RequestException as e:
                st.error(f"Connection Error: {e}")

            except (ValueError, KeyError):
                st.error("Invalid response received from backend.")
