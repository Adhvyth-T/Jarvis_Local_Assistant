"""
Jarvis AI Assistant - LangChain Implementation (FULLY FIXED)
"""

import streamlit as st
from pathlib import Path
from typing import List, Dict
import requests
import os
import tempfile

# LangChain imports
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone as LangchainPinecone

from config import Config

# Page configuration
st.set_page_config(
    page_title="Jarvis AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message { background-color: #e3f2fd; }
    .assistant-message { background-color: #f5f5f5; }
    .model-info {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def get_available_models() -> List[Dict]:
    """Fetch available models from Ollama"""
    try:
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [{
                'name': model['name'],
                'size': model.get('size', 0) / (1024**3)
            } for model in models]
    except:
        return []

def check_ollama_connection() -> bool:
    """Check if Ollama is running"""
    try:
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'messages': [],
        'qa_chain': None,
        'vector_store': None,
        'embeddings': None,
        'memory': None,
        'initialized': False,
        'selected_model': None,
        'available_models': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def initialize_components(model_name: str, temperature: float, pinecone_api_key: str):
    """Initialize LangChain components"""
    try:
        with st.spinner("Initializing Jarvis..."):
            # Initialize embeddings
            st.info("📊 Loading embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            
            # Initialize Ollama LLM
            st.info(f"🤖 Loading {model_name}...")
            llm = Ollama(
                model=model_name,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=temperature
            )
            
            # Initialize Pinecone
            st.info("🔍 Connecting to Pinecone...")
            from pinecone import Pinecone, ServerlessSpec
            
            pc = Pinecone(api_key=pinecone_api_key)
            index_name = Config.PINECONE_INDEX_NAME
            
            # Create index if needed
            existing = [idx.name for idx in pc.list_indexes()]
            if index_name not in existing:
                st.info(f"Creating index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            
            # Get index
            index = pc.Index(index_name)
            
            # Initialize vector store - FIXED METHOD
            st.session_state.vector_store = LangchainPinecone(
                index=index,
                embedding=embeddings,  # Pass embeddings object directly
                text_key="text"
            )
            
            # Initialize memory
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer'
            )
            
            # Create QA chain
            prompt_template = """You are Jarvis, a helpful AI assistant. Use the context to answer the question. If you don't know, say so.

Context: {context}

Question: {question}

Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": Config.MAX_CONTEXT_CHUNKS}
                ),
                memory=st.session_state.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": PROMPT}
            )
            
            st.session_state.initialized = True
            st.session_state.selected_model = model_name
            st.success("✅ Jarvis ready!")
            return True
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False

def display_chat_message(role: str, content: str):
    """Display chat message"""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    st.markdown(f'<div class="chat-message {css_class}"><strong>{icon} {role.title()}</strong><div style="margin-top:0.5rem">{content}</div></div>', unsafe_allow_html=True)

def process_document(file_path: str) -> bool:
    """Process and add document"""
    try:
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['.txt', '.md']:
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext in ['.docx', '.doc']:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            from langchain.schema import Document as LangchainDoc
            documents = [LangchainDoc(page_content=text, metadata={"source": file_path})]
            loader = None
        else:
            raise ValueError(f"Unsupported: {ext}")
        
        if loader:
            documents = loader.load()
        
        # Split and add
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(documents)
        st.session_state.vector_store.add_documents(chunks)
        return True
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def sidebar():
    """Sidebar"""
    with st.sidebar:
        st.title("⚙️ Jarvis Settings")
        
        # Connection
        st.subheader("🔌 Connection")
        if check_ollama_connection():
            st.success("✅ Ollama Connected")
        else:
            st.error("❌ Start Ollama: `ollama serve`")
            return
        
        st.divider()
        
        # Models
        st.subheader("🤖 Model Selection")
        if st.button("🔄 Refresh"):
            st.session_state.available_models = get_available_models()
        
        if not st.session_state.available_models:
            st.session_state.available_models = get_available_models()
        
        if st.session_state.available_models:
            options = [f"{m['name']} ({m['size']:.1f}GB)" for m in st.session_state.available_models]
            selected = st.selectbox("Model", options)
            model = selected.split(' (')[0]
        else:
            st.warning("No models! Run: `ollama pull llama2`")
            model = None
        
        st.divider()
        
        # Temperature
        st.subheader("🌡️ Settings")
        temp = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        st.divider()
        
        # Pinecone
        st.subheader("🔍 Pinecone")
        api_key = st.text_input(
            "API Key", 
            type="password", 
            value=Config.PINECONE_API_KEY,
            placeholder="Enter your Pinecone API key"
        )
        
        st.divider()
        
        # Initialize
        if st.button("🚀 Initialize", type="primary", use_container_width=True):
            if model and api_key:
                initialize_components(model, temp, api_key)
            else:
                st.error("Need model + API key")
        
        st.divider()
        
        # Upload
        st.subheader("📚 Documents")
        file = st.file_uploader("Upload", type=['txt', 'pdf', 'md', 'doc', 'docx'])
        
        if file and st.button("➕ Add"):
            if not st.session_state.initialized:
                st.error("Initialize first")
            else:
                with st.spinner("Processing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                        tmp.write(file.getbuffer())
                        if process_document(tmp.name):
                            st.success(f"✅ Added {file.name}")
        
        st.divider()
        
        # Controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.memory:
                    st.session_state.memory.clear()
                st.rerun()
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.clear()
                st.rerun()

def main():
    """Main"""
    initialize_session_state()
    
    st.title("🤖 Jarvis AI Assistant")
    st.markdown("*LangChain + Pinecone*")
    
    if st.session_state.initialized and st.session_state.selected_model:
        st.markdown(f'<div class="model-info">🤖 <strong>{st.session_state.selected_model}</strong></div>', unsafe_allow_html=True)
    
    sidebar()
    
    if not st.session_state.initialized:
        st.info("👈 Initialize Jarvis in sidebar")
        st.markdown("""
        ### Steps:
        1. Start Ollama: `ollama serve`
        2. Select model
        3. Enter Pinecone API key
        4. Click Initialize
        """)
        return
    
    # Chat history
    for msg in st.session_state.messages:
        display_chat_message(msg["role"], msg["content"])
    
    # Input
    if prompt := st.chat_input("Ask Jarvis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain.invoke({"question": prompt})
                response = result['answer']
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_chat_message("assistant", response)
                
                if result.get('source_documents'):
                    with st.expander("📄 Sources"):
                        for i, doc in enumerate(result['source_documents'][:3], 1):
                            st.write(f"**{i}:** {doc.page_content[:200]}...")
                            
            except Exception as e:
                st.error(f"❌ {str(e)}")

if __name__ == "__main__":
    main()