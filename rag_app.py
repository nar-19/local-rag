import streamlit as st
import os
from google import genai
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic import hub # For loading the RAG prompt (optional, not strictly used in this version)


# Initialize the API key from secrets file
client = genai.Client(api_key = st.secrets["GEMINI_API_KEY"])

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Local Knowledge Base with Gemini", layout="wide")
st.title("📚 Local Knowledge Base with Gemini")
st.markdown("---")

# Sidebar for API Key input and instructions
with st.sidebar:
    st.header("Configuration")
    st.markdown("1.  **Set up `GOOGLE_API_KEY` in `.env` file.**")
    st.markdown("2.  **Specify your document directory.**")
    st.markdown("3.  **App will attempt to load existing knowledge base on startup.**") # Updated instruction
    st.markdown("4.  **Click 'Process Documents' to rebuild or process a new directory.**") # Updated instruction
    st.markdown("5.  **Enter your query in the chat bar.**")

    if not GEMINI_API_KEY:
        st.warning("`GOOGLE_API_KEY` not found. Please set it in a `.env` file or directly in the script.")
        st.stop() # Stop execution if API key is missing

    st.subheader("Document Processing")
    # Ensure doc_dir has a default value for initial load
    doc_dir = st.text_input("Path to documents directory:", value="./docs")
    process_button = st.button("Process Documents (Rebuild/Process New Dir)") # Updated button label

# --- Initialize Session State ---
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'rag_chain' not in st.session_state:
    st.session_state['rag_chain'] = None
if 'knowledge_tree_structure' not in st.session_state:
    st.session_state['knowledge_tree_structure'] = "No documents processed yet."
# New: Flag to ensure initial load/process attempt only happens once per session
if 'initial_load_attempted' not in st.session_state:
    st.session_state['initial_load_attempted'] = False

# --- Functions ---

def load_documents(directory):
    """Loads documents from the specified directory."""
    documents = []
    
    try:
        st.info(f"Loading documents from {directory}...")
        
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": TextLoader # Markdown files can be loaded as text
        }
        
        # Walk through the directory to find supported files
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
    """
    Attempts to load an existing Chroma DB. If not found or fails,
    it loads documents, splits them, creates embeddings, and stores them in Chroma DB.
    """
    # 1. Attempt to load existing vector store
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)
            st.session_state['vector_store'] = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
            
            # Verify if it actually contains data
            if st.session_state['vector_store']._collection.count() > 0: # Access internal count for verification
                st.success(f"Loaded existing Chroma DB from '{VECTOR_DB_DIR}' with {st.session_state['vector_store']._collection.count()} chunks.")
                st.session_state['knowledge_tree_structure'] = (
                    f"Loaded existing knowledge base from '{VECTOR_DB_DIR}'. "
                    f"Contains {st.session_state['vector_store']._collection.count()} chunks."
                )
                return True
            else:
                st.warning(f"Existing Chroma DB in '{VECTOR_DB_DIR}' found but appears empty. Rebuilding...")
                # Proceed to rebuild if empty
        except Exception as e:
            st.error(f"Error loading existing Chroma DB from '{VECTOR_DB_DIR}': {e}. Rebuilding...")
            # If loading fails, proceed to rebuild
    
    # 2. If no existing DB or loading failed, proceed to process new documents
    documents = load_documents(doc_directory)
    if not documents:
        st.warning("No supported documents found or loaded from the directory to build a new knowledge base.")
        return False

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    if not splits:
        st.warning("No text chunks generated from documents.")
        return False

    st.info(f"Creating embeddings for {len(splits)} chunks. This might take a moment...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)
    
    try:
        st.session_state['vector_store'] = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        st.session_state['vector_store'].persist() # Ensures data is written to disk
        st.success(f"Knowledge base created with {len(splits)} chunks and saved to '{VECTOR_DB_DIR}'.")

        # Update knowledge tree structure display
        file_counts = {}
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            file_counts[source] = file_counts.get(source, 0) + 1
        
        structure_display = "## Processed Documents:\n"
        for source, count in file_counts.items():
            structure_display += f"- `{os.path.basename(source)}`\n"
        structure_display += f"\nTotal original documents loaded: {len(file_counts)}\n"
        structure_display += f"Total chunks created: {len(splits)}\n"
        st.session_state['knowledge_tree_structure'] = structure_display
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return False

def setup_rag_chain():
    """Sets up the Retrieval Augmented Generation (RAG) chain."""
    if st.session_state['vector_store'] is None:
        # This case should ideally not be hit if initial loading/processing was successful
        st.error("Vector store not initialized. Cannot set up RAG chain.")
        return
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.2)
    # Retrieve top 5 relevant chunks by default
    retriever = st.session_state['vector_store'].as_retriever(search_kwargs={"k": 5}) 
    
    st.session_state['rag_chain'] = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" means combining all retrieved documents into one prompt
        retriever=retriever,
        return_source_documents=True # Return the source documents used for the answer
    )
    st.info("RAG chain initialized and ready for queries!")


# --- Main App Logic ---

# --- Automatic Initial Load Logic (runs only once per Streamlit session) ---
if not st.session_state['initial_load_attempted']:
    st.session_state['initial_load_attempted'] = True # Mark as attempted
    
    with st.spinner("Attempting to load existing knowledge base from disk or process documents for the first time..."):
        # `doc_dir` will already have the default value from `st.text_input` if not changed
        if process_documents_and_create_vector_store(doc_dir):
            setup_rag_chain()
        else:
            # This message appears if no existing DB was loaded AND initial processing failed/found no docs
            st.warning("Knowledge base not loaded or created. Please ensure your document directory is correct and click 'Process Documents' to build it.")

# Process documents button handler (for explicit rebuild or processing a different directory)
if process_button:
    if doc_dir:
        with st.spinner("Processing documents and building knowledge base..."):
            # This call will re-run the processing logic, potentially rebuilding the DB
            if process_documents_and_create_vector_store(doc_dir):
                setup_rag_chain()
    else:
        st.warning("Please enter a directory path.")

# Display Knowledge Tree structure (or processing status)
st.subheader("Knowledge Tree Structure")
with st.expander("Click to view processed documents"):
    st.markdown(st.session_state['knowledge_tree_structure'])

# Query bar
st.subheader("Ask a Question")
if st.session_state['rag_chain']:
    query = st.chat_input("Enter your query here...")
    if query:
        with st.spinner("Searching and generating answer..."):
            try:
                response = st.session_state['rag_chain'].invoke({"query": query})
                st.markdown(f"**Answer:** {response['result']}")

                # Display sources
                if response.get('source_documents'):
                    st.markdown("---")
                    st.markdown("**Sources Used:**")
                    for i, doc in enumerate(response['source_documents']):
                        st.markdown(f"**{i+1}.** `{os.path.basename(doc.metadata.get('source', 'Unknown'))}`")
                        # st.write(f"Content snippet: {doc.page_content[:200]}...") # Uncomment to see snippet
            except Exception as e:
                st.error(f"An error occurred during query processing: {e}")
                st.info("Please ensure your API key is correct and the knowledge base is properly initialized.")
else:
    st.info("Knowledge base not ready. Please check configuration and process documents.")
