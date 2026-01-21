# 🤖 Jarvis AI Assistant

A scalable, production-ready AI assistant built with **LangChain**, **Ollama**, and **Pinecone**. Features dynamic model selection, conversational memory, and enterprise-grade RAG implementation.

![Jarvis AI Chatbot](image.png)


## 🌟 Key Features

- ✅ **LangChain Framework** - Industry-standard, scalable architecture
- ✅ **Dynamic Model Selection** - Choose from any installed Ollama model
- ✅ **Pinecone Vector Store** - Cloud-native, scalable vector database
- ✅ **Conversational Memory** - Context-aware chat with history
- ✅ **RAG Pipeline** - Retrieval Augmented Generation
- ✅ **Multi-Format Support** - PDF, DOCX, TXT, MD documents
- ✅ **Source Attribution** - See which documents informed responses
- ✅ **Real-time Model Switching** - Change models on the fly
- ✅ **12GB RAM Optimized** - Works smoothly on modest hardware

---


> **⚠️ Important Prerequisites**
> - **Python 3.12.6** (Required)
> - Ollama installed and running
> - Pinecone account (free tier available)
---
## 🏗️ Architecture

```
┌─────────────┐
│   User UI   │ (Streamlit)
└──────┬──────┘
       │
┌──────▼──────────────────────────────┐
│      LangChain Framework            │
│  ┌────────────┬────────────────┐    │
│  │   Memory   │   Retriever    │    │
│  └────────────┴────────────────┘    │
└──────┬──────────────────┬───────────┘
       │                  │
┌──────▼──────┐    ┌──────▼──────────┐
│   Ollama    │    │    Pinecone     │
│   (LLM)     │    │  (Vectors)      │
└─────────────┘    └─────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | LangChain | Orchestration & chains |
| **UI** | Streamlit | Web interface |
| **LLM** | Ollama | Self-hosted models |
| **Vector DB** | Pinecone | Semantic search |
| **Embeddings** | HuggingFace | Text vectorization |
| **Memory** | LangChain Memory | Conversation history |

---

## 📦 Installation

### Prerequisites

1. **Python 3.12.6(important)**
2. **Ollama** installed and running
3. **Pinecone account** (free tier available at pinecone.io)

### Step 1: Install Ollama

**Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

### Step 2: Install & Start Ollama Models

```bash
# Start Ollama server (keep this running)
ollama serve

# In a new terminal, install models:

# Lightweight (recommended for 12GB RAM)
ollama pull phi:2.7b       # 1.7GB - Fast and efficient

# Balanced quality
ollama pull llama2:7b      # 3.8GB - Best balance

# High quality
ollama pull mistral:7b     # 4.1GB - Better responses

# For coding
ollama pull codellama:7b   # 3.8GB - Specialized

# You can install multiple models and switch between them!
```

### Step 3: Get Pinecone API Key

1. Go to [https://www.pinecone.io/](https://www.pinecone.io/)
2. Sign up for free account
3. Create a new project
4. Copy your API key from the dashboard

### Step 4: Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**For Python 3.12 compatibility issues:**
```bash
# Use the minimal requirements (faster, smaller)
pip install -r requirements_minimal.txt
```

---

## 🚀 Usage

### Starting Jarvis

```bash
# 1. Start Ollama (if not running)
ollama serve

# 2. In new terminal, activate venv
venv\Scripts\activate     # Windows
# or
source venv/bin/activate  # Linux/Mac

# 3. Run Jarvis
streamlit run app.py

# 4. Open browser at:
# http://localhost:8501
```

### First Time Setup

1. **Check Connection**
   - Sidebar should show "✅ Ollama Connected"
   - If not, start Ollama with `ollama serve`

2. **Select Model**
   - Click "🔄 Refresh Models" to see installed models
   - Choose from dropdown (e.g., "llama2:7b")
   - See model size and details

3. **Enter Pinecone API Key**
   - Paste your API key in the text field
   - Or set in config.py: `PINECONE_API_KEY = "your-key"`

4. **Initialize Jarvis**
   - Click "🚀 Initialize Jarvis"
   - Wait for setup (15-30 seconds first time)
   - Look for "✅ Jarvis ready!"

5. **Add Documents** (Optional)
   - Upload PDF, DOCX, TXT, or MD files
   - Click "➕ Add to Knowledge Base"
   - Documents are chunked and stored in Pinecone

6. **Start Chatting!**
   - Type your question in the chat input
   - Jarvis will search documents and respond
   - View sources used in expandable section

---

## 🎯 Features Guide

### Dynamic Model Selection

```
Available Models (Note: depends on your local ollama env):
├── phi:2.7b (1.7GB)        ← Recommended for demos
├── llama2:7b (3.8GB)       ← Best balance
├── mistral:7b (4.1GB)      ← High quality
└── codellama:7b (3.8GB)    ← For coding
```

**To add more models:**
```bash
ollama pull <model-name>
# Then click "Refresh Models" in sidebar
```

### Conversational Memory

Jarvis remembers your conversation:
- Last 10 exchanges kept in memory
- Context maintained across questions
- Click "Clear" to start fresh

### Source Attribution

Every response shows:
- 📄 Which documents were used
- Relevant excerpts from sources
- Expandable source viewer

### Temperature Control

Adjust creativity:
- **0.0-0.3**: Focused, factual
- **0.4-0.7**: Balanced (default)
- **0.8-1.0**: Creative, diverse

---

## 📊 Performance on 12GB RAM

### Model Performance

| Model | Load Time | Response Time | Memory Usage | Quality |
|-------|-----------|---------------|--------------|---------|
| phi:2.7b | 5-8s | 1-3s | 2-3GB | Very Good |
| llama2:7b | 10-15s | 3-8s | 5-6GB | Excellent |
| mistral:7b | 10-15s | 3-8s | 5-6GB | Excellent |

**Recommendation for <= 16GB RAM:** 
- **Demo/Testing:** phi:2.7b
- **Production:** llama2:7b or mistral:7b

---

## 🔧 Configuration

### config.py Settings

```python
# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.7

# Pinecone
PINECONE_API_KEY = ""  # Add your key here
PINECONE_INDEX_NAME = "jarvis-knowledge-base"

# Document Processing
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_CONTEXT_CHUNKS = 3
```

### Environment Variables (Optional)

Create `.env` file:
```bash
PINECONE_API_KEY=your_api_key_here
```

---

## 🎓 Use Cases

### 1. Research Assistant
```
Upload: Research papers, articles
Ask: "Summarize the main findings"
     "Compare methodologies"
```

### 2. Code Helper
```
Upload: Documentation, code files
Ask: "How do I implement X?"
     "Debug this error"
```

### 3. Study Buddy
```
Upload: Textbooks, notes
Ask: "Explain concept X"
     "Quiz me on topic Z"
```

### 4. Business Intelligence
```
Upload: Reports, memos
Ask: "What are key metrics?"
     "List action items"
```

---

## 🐛 Troubleshooting

### "Ollama Not Running"
```bash
ollama serve
```

### "No models found"
```bash
ollama list
ollama pull llama2:7b
# Click "Refresh Models" in sidebar
```

### "Pinecone connection failed"
- Check API key in config.py
- Verify internet connection
- Check Pinecone dashboard

### "Out of memory"
- Use phi:2.7b instead of llama2
- Close other applications
- Restart: `pkill ollama && ollama serve`

### "ImportError" or package issues
```bash
# Reinstall with minimal requirements
pip install -r requirements_minimal.txt
```

---

## 📝 Project Structure

```
jarvis-ai-assistant/
├── app.py                 # Main application
├── config.py             # Configuration
├── requirements.txt      # Full dependencies
├── Readme.md            # This file
└── .env.example         # Environment template
```

---

## 🎯 Key Talking Points

### "Why LangChain?"
- Industry standard framework
- Modular and extensible
- Large community support
- Built-in best practices

### "Why Pinecone?"
- Cloud-native, auto-scaling
- Low latency (< 100ms)
- Managed service
- Enterprise-grade security

### "Dynamic Model Selection?"
- Flexibility for different use cases
- Test new models easily
- Users choose speed vs quality
- No code changes needed

---

## 📚 Resources

- **LangChain:** https://python.langchain.com/docs/
- **Ollama:** https://ollama.ai/library
- **Pinecone:** https://docs.pinecone.io/
- **Streamlit:** https://docs.streamlit.io/

---

## ✅ Summary

**Production-ready AI assistant with:**
- ✅ LangChain framework
- ✅ Dynamic model selection
- ✅ Pinecone vector storage
- ✅ 12GB RAM optimized
- ✅ Enterprise-grade architecture


---

**Built with ❤️ using LangChain**