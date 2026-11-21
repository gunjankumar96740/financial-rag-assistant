
import streamlit as st
import requests

st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📊",
    layout="wide"
)

API_URL = st.secrets["API_URL"].rstrip("/")
API_KEY = st.secrets["API_KEY"]

HEADERS = {"X-API-Key": API_KEY}

# --------------------------------
# SESSION STATE
# --------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "report_name" not in st.session_state:
    st.session_state.report_name = None


# --------------------------------
# HEADER
# --------------------------------

st.title("📊 Financial RAG Assistant")

st.markdown("""
### Your AI-powered financial research assistant

Upload a company's annual report and ask questions about
revenue, profitability, financial performance and business risks.

Powered by Retrieval-Augmented Generation and AI agents.
""")

st.divider()


# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:

    st.title("About the Project")

    st.markdown("""
    **Core Technologies**

    - Python
    - LangChain and LangGraph
    - FAISS Vector Database
    - BM25 Hybrid Retrieval
    - CrossEncoder Reranking
    - Groq LLM
    - FastAPI
    - Streamlit
    - LangSmith

    **Key Capabilities**

    - Annual report analysis
    - Financial calculations
    - Financial comparisons
    - Context-aware conversations
    """)

    st.divider()

    st.caption(
        "Developed as a Financial RAG portfolio project."
    )


# --------------------------------
# PDF UPLOAD
# --------------------------------

st.subheader("📄 Upload Annual Report")

uploaded_file = st.file_uploader(
    "Choose your company's annual report",
    type=["pdf"],
    help="Maximum PDF size: 20 MB"
)

if uploaded_file:

    st.write(f"Selected report: **{uploaded_file.name}**")

    if st.button("Upload and Process Report", type="primary"):

        with st.spinner(
            "Processing your annual report. This may take a few minutes..."
        ):

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

                st.session_state.session_id = result["session_id"]
                st.session_state.report_name = uploaded_file.name
                st.session_state.messages = []

                st.success(
                    "Annual report successfully uploaded and indexed!"
                )

            except requests.HTTPError:

                try:
                    error = response.json().get("detail", response.text)
                except ValueError:
                    error = response.text

                st.error(f"Upload failed: {error}")

            except requests.RequestException as e:
                st.error(f"Connection error: {e}")


# --------------------------------
# CHAT SECTION
# --------------------------------

st.divider()
st.subheader("💬 Financial Research Chat")

if st.session_state.session_id:

    st.success(
        f"Analyzing: {st.session_state.report_name}"
    )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask about revenue, net profit, risks, etc."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Analyzing financial information..."):

                try:

                    response = requests.post(
                        f"{API_URL}/chat",

                        json={
                            "question": question,
                            "session_id": st.session_state.session_id
                        },

                        headers=HEADERS,
                        timeout=180
                    )

                    response.raise_for_status()

                    answer = response.json()["answer"]

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except requests.HTTPError:

                    try:
                        error = response.json().get(
                            "detail",
                            response.text
                        )
                    except ValueError:
                        error = response.text

                    st.error(f"API error: {error}")

                except requests.RequestException as e:
                    st.error(f"Connection error: {e}")

else:

    st.info(
        "Upload an annual report above to start asking financial questions."
    )

    st.markdown("**Try asking questions like:**")

    st.markdown("""
    - What was the company's total revenue?
    - What was its net profit?
    - Calculate its net profit margin.
    - How did revenue change compared with the previous year?
    - What are the major business risks?
    """)
