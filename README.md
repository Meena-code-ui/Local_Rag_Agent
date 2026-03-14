# RIO:The Local Rag Agent

> Local-First AI Agent with RAG and Autonomous Tool Selection  
> Optimized for 8GB RAM | CPU Inference | 100% Private | Zero API Costs

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1e3a8a?style=for-the-badge)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-ff6b35?style=for-the-badge)](https://ollama.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?style=for-the-badge)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## Overview

RIO is a lightweight, privacy-first AI agent that combines Retrieval-Augmented Generation (RAG) with autonomous tool selection to answer questions from your documents or the live web — all running locally on your machine.

Built specifically for resource-constrained environments (tested on 8GB RAM laptops), RIO demonstrates that powerful AI does not require cloud APIs or expensive GPUs.

### Key Features

| Feature | Description |
|---------|-------------|
| Document RAG | Ingest PDFs, chunk intelligently, embed locally, and query with context-aware retrieval |
| Autonomous Agent | Dynamically chooses between local knowledge base and live web search based on query intent |
| CPU-Optimized | Runs entirely on CPU using quantized Phi-3-mini (3.8B) via Ollama — no GPU required |
| Privacy-First | No data leaves your machine; no API keys; fully offline-capable |
| Resume-Aware | Specialized prompt engineering for structured extraction from resumes and CVs |
| Memory Management | One-click database clear, chunk limiting, and persistent disk storage to prevent RAM overflow |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM Engine | Ollama + Phi-3-mini (3.8B) | Local inference with 4-bit quantization |
| Framework | LangChain 0.2+ | Agent orchestration, tool integration, prompt management |
| Vector DB | ChromaDB (persistent) | Local embedding storage with disk persistence |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Lightweight CPU-friendly embeddings (~80MB) |
| UI | Streamlit | Interactive chat interface with file upload |
| Web Search | DuckDuckGo Search API | Free, no-key web search tool for agent |
| PDF Processing | PyPDFLoader + RecursiveCharacterTextSplitter | Document ingestion with structure-preserving chunking |

---

## Hardware Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | Intel i5 / AMD Ryzen 5 (4 cores) | Intel i7 / Ryzen 7 (6+ cores) |
| Storage | 5 GB free | 10 GB free (for models and database) |
| GPU | Not required | Integrated GPU helps inference speed |
| OS | Windows 10/11, macOS, Linux | Windows 11 (tested) |

Tested on: Intel i5-1135G7, 8GB RAM, Windows 11 — average inference: 2-4 words/sec on CPU

---

## Quick Start

### Prerequisites
1. Install [Ollama](https://ollama.com) and pull the model:
   ```bash
   ollama pull phi3:mini
   ```
2. Ensure Python 3.9+ is installed.

### Installation
```bash
# Clone the repository
git clone https://github.com/Meena-code-ui/edgemind-local-rag-agent.git
cd edgemind-local-rag-agent

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application
```bash
# Launch the Streamlit UI
streamlit run app.py
```

The application will open at `http://localhost:8501`.

### First-Time Setup
1. Upload a PDF (e.g., your resume) via the sidebar
2. Click "Process Document" to index it
3. Ask questions like:
   - "What are my top technical skills?"
   - "Summarize my work experience"
   - "What is the latest news about AI agents?" (triggers web search)

---

## Project Structure

```
edgemind-local-rag-agent/
│
├── app.py                 # Streamlit UI with RIO branding and agent logic
├── rag_core.py            # Standalone RAG pipeline (CLI)
├── agent_core.py          # Command-line agent with tool selection
├── requirements.txt       # Python dependencies
├── .gitignore            # Excludes venv/, db/, *.pdf, secrets
├── README.md             # This file
│
├── db/                   # ChromaDB vector store (auto-created, git-ignored)
├── sample.pdf            # Example document for testing (optional, git-ignored)
│
└── LICENSE               # MIT License
```

---

## Usage Examples

### Query Your Resume
```
User: "What programming languages am I proficient in?"
RIO [Knowledge Base]: 
- Python (Flask, Pandas, LangChain)
- SQL (MySQL schema design, migrations)
- JavaScript (web scraping with Selenium)
- Prompt Engineering (ChatGPT, GroqAI integration)
```

### Web-Enabled Research
```
User: "What are the latest developments in local LLMs?"
RIO [Web Search]: 
Based on recent articles (March 2026):
- Phi-3-mini continues to lead in CPU efficiency
- Ollama 0.2 adds improved Windows support
- New quantization techniques reduce VRAM by 30%
```

### Hybrid Query
```
User: "Compare my experience with current industry trends in data engineering"
RIO [Knowledge Base + Web Search]: 
Your experience includes: ETL pipelines, API integration, OCR...
Current trends (from web): Real-time streaming (Kafka), dbt for transformations...
Alignment: Strong foundation; consider adding cloud data warehouse exposure.
```


---

## Configuration and Customization

### Adjust Retrieval Sensitivity
Edit `app.py` to change how many chunks are retrieved:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Increase for more context
```

### Change the LLM Model
Update `LLM_MODEL` in `app.py`:
```python
LLM_MODEL = "llama3:8b"  # Requires more RAM (~6GB)
# or
LLM_MODEL = "phi3:mini"  # Default, optimized for 8GB RAM
```


---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Contact

Meenalochani Selvam   
LinkedIn: https://linkedin.com/in/meenalochani-selvam  
Location: Chennai, India  

Project Link: https://github.com/Meena-code-ui/Local_Rag_Agent

---
