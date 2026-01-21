"""
Configuration settings for Jarvis AI Assistant - LangChain Implementation
Optimized for 12GB RAM with Pinecone
"""
import os
class Config:
    """Application configuration - LangChain + Pinecone"""
    
    # Ollama Configuration
    OLLAMA_BASE_URL = "http://localhost:11434"
    DEFAULT_TEMPERATURE = 0.7
    
    # Embedding Model Configuration
    # Using lightweight model optimized for semantic search
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions
    
    # Pinecone Configuration
    PINECONE_INDEX_NAME = "jarvis-knowledge-base"
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '')
    PINECONE_ENVIRONMENT = "us-east-1"  # AWS region
    
    # Document Processing - Optimized for 12GB RAM
    CHUNK_SIZE = 500  # Characters per chunk
    CHUNK_OVERLAP = 50  # Overlap between chunks
    MAX_CONTEXT_CHUNKS = 3  # Number of chunks to retrieve for context
    
    # LangChain Settings
    MAX_TOKENS = 512  # Maximum tokens in response
    VERBOSE = False  # Set to True for debugging
    
    # Memory Settings
    MAX_MEMORY_LENGTH = 10  # Keep last 10 exchanges in memory
    
    # Supported file formats
    SUPPORTED_FORMATS = ['.txt', '.pdf', '.md', '.doc', '.docx']
    
    # Recommended Models for 12GB RAM
    RECOMMENDED_MODELS = {
        'lightweight': [
            'tinydolphin:latest',      # 637MB - Ultra fast
            'phi:2.7b',          # 1.7GB - Fast, good quality
        ],
        'balanced': [
            'llama2:7b',      # 3.8GB - Best balance
            'mistral:7b',     # 4.1GB - High quality
        ],
        'specialized': [
            'codellama:instruct',   # 3.8GB - For coding
            'neural-chat',    # 4.1GB - Conversational
        ]
    }
    
    # Performance Tips
    PERFORMANCE_TIPS = """
    For optimal performance on 12GB RAM:
    1. Use 7B parameter models (not 13B or 70B)
    2. Set temperature between 0.5-0.8
    3. Keep chunk size at 500 characters
    4. Limit context chunks to 3
    5. Close other heavy applications
    """