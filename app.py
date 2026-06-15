from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.chat import generate_answer
from src.pdf_processor import extract_and_chunk_pdf
from src.vectorstore import VectorStore

load_dotenv()

st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="wide")

st.title("📄 AskMyDocs")
st.caption("Upload a PDF and ask questions about it — powered by Claude AI")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Reading and indexing PDF..."):
                tmp_path = Path("data") / uploaded_file.name
                tmp_path.parent.mkdir(exist_ok=True)
                tmp_path.write_bytes(uploaded_file.getvalue())

                chunks = extract_and_chunk_pdf(str(tmp_path))

                vs = VectorStore()
                vs.add_documents(chunks, uploaded_file.name)

                st.session_state.vectorstore = vs
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []

            st.success(f"Indexed {len(chunks)} chunks from **{uploaded_file.name}**")

    if st.session_state.doc_name:
        st.info(f"Active document: **{st.session_state.doc_name}**")

    if st.session_state.messages and st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.vectorstore is None:
        st.error("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = generate_answer(
                    query=prompt,
                    vectorstore=st.session_state.vectorstore,
                    chat_history=st.session_state.messages[:-1],
                )
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
