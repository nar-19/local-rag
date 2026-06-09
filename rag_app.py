import streamlit as st
import os
from google import genai

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

GEMINI_API_KEY = st.secrets["API_KEY"]
client = genai.Client(api_key=st.secrets["API_KEY"])
VECTOR_DB_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"

st.set_page_config(page_title="Local Knowledge Base with Gemini", layout="wide")
st.title("📚 Local Knowledge Base with Gemini")
st.markdown("---")

with st.sidebar:
    st.header("Configuration")
    st.markdown("1. **Set up `GOOGLE_API_KEY` in `.env` file.**")
    st.markdown("2. **Specify your document directory.**")
    st.markdown("3. **App will attempt to load existing knowledge base on startup.**")
    st.markdown("4. **Click 'Process Documents' to rebuild or process a new directory.**")
    st.markdown("5. **Enter your query in the chat bar.**")

    if not GEMINI_API_KEY:
        st.warning("`GOOGLE_API_KEY` not found.")
        st.stop()

    st.subheader("Document Processing")
    doc_dir = st.text_input("Path to documents directory:", value="./docs")
    process_button = st.button("Process Documents (Rebuild/Process New Dir)")

# --- Session State ---
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'rag_chain' not in st.session_state:
    st.session_state['rag_chain'] = None
if 'initial_load_attempted' not in st.session_state:
    st.session_state['initial_load_attempted'] = False
if 'loaded_document_names' not in st.session_state:
    st.session_state['loaded_document_names'] = []

# --- Functions ---

def load_documents(directory):
    documents = []
    try:
        st.info(f"Loading documents from {directory}...")
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": TextLoader
        }
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension in loaders:
                    try:
                        loader = loaders[file_extension](file_path)
                        documents.extend(loader.load())
                        st.text(f"Loaded: {os.path.basename(file_path)}")
                    except Exception as e:
                        st.warning(f"Could not load {file_path}: {e}")
                else:
                    st.info(f"Skipping unsupported file: {file_path}")
        return documents
    except Exception as e:
        st.error(f"Error loading documents from {directory}: {e}")
        return []


def process_documents_and_create_vector_store(doc_directory):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # 1. Attempt to load existing vector store
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        try:
            st.session_state['vector_store'] = Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )
            count = st.session_state['vector_store']._collection.count()
            if count > 0:
                # Extract unique document names from stored metadata
                results = st.session_state['vector_store']._collection.get()
                unique_sources = sorted(set(
                    os.path.basename(m['source'])
                    for m in results['metadatas']
                    if m and 'source' in m
                ))
                st.session_state['loaded_document_names'] = unique_sources
                st.success(f"Loaded existing Chroma DB from '{VECTOR_DB_DIR}' with {count} chunks.")
                return True
            else:
                st.warning("Existing Chroma DB found but is empty. Rebuilding...")
        except Exception as e:
            st.error(f"Error loading existing Chroma DB: {e}. Rebuilding...")

    # 2. Process new documents
    documents = load_documents(doc_directory)
    if not documents:
        st.warning("No supported documents found.")
        return False

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    if not splits:
        st.warning("No text chunks generated from documents.")
        return False

    st.info(f"Creating embeddings for {len(splits)} chunks...")

    try:
        st.session_state['vector_store'] = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR,
            collection_name=COLLECTION_NAME
        )

        # Store unique document names from processed docs
        st.session_state['loaded_document_names'] = sorted(set(
            os.path.basename(doc.metadata.get('source', 'Unknown'))
            for doc in documents
        ))

        st.success(f"Knowledge base created with {len(splits)} chunks and saved to '{VECTOR_DB_DIR}'.")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return False


def setup_rag_chain():
    if st.session_state['vector_store'] is None:
        st.error("Vector store not initialized.")
        return

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2
    )
    retriever = st.session_state['vector_store'].as_retriever(search_kwargs={"k": 5})
    st.session_state['rag_chain'] = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    st.info("RAG chain initialized and ready for queries!")


# --- Main App Logic ---

if not st.session_state['initial_load_attempted']:
    st.session_state['initial_load_attempted'] = True
    with st.spinner("Attempting to load existing knowledge base..."):
        if process_documents_and_create_vector_store(doc_dir):
            setup_rag_chain()
        else:
            st.warning("Knowledge base not loaded. Please process documents.")

if process_button:
    if doc_dir:
        with st.spinner("Processing documents and building knowledge base..."):
            if process_documents_and_create_vector_store(doc_dir):
                setup_rag_chain()
    else:
        st.warning("Please enter a directory path.")

# --- Loaded Documents Display ---
st.subheader("📄 Loaded Documents")
if st.session_state['loaded_document_names']:
    cols = st.columns(3)
    for i, name in enumerate(st.session_state['loaded_document_names']):
        cols[i % 3].markdown(f"📄 `{name}`")
else:
    st.info("No documents loaded yet.")

st.markdown("---")

# --- Query ---
st.subheader("Ask a Question")
if st.session_state['rag_chain']:
    query = st.chat_input("Enter your query here...")
    if query:
        with st.spinner("Searching and generating answer..."):
            try:
                response = st.session_state['rag_chain'].invoke({"query": query})
                st.markdown(f"**Answer:** {response['result']}")
                if response.get('source_documents'):
                    st.markdown("---")
                    st.markdown("**Sources Used:**")
                    for i, doc in enumerate(response['source_documents']):
                        st.markdown(f"**{i+1}.** `{os.path.basename(doc.metadata.get('source', 'Unknown'))}`")
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Knowledge base not ready. Please check configuration and process documents.")
