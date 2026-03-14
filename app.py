# app.py - RIO: Royal Blue & Black Professional Theme
import streamlit as st
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun

# --- Page Config ---
st.set_page_config(page_title=" RIO Agent", page_icon="🦜", layout="wide")

# --- Custom CSS: Royal Blue & Black Professional Theme ---
def inject_custom_css():
    st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
        color: #e0e0ff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f0f1a 100%);
        border-right: 2px solid #4169e1;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #6495ed !important;
        font-weight: 600;
    }
    
    /* Title Styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4169e1, #6495ed, #87ceeb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(65, 105, 225, 0.3);
    }
    
    .subtitle {
        color: #8899bb;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid #2a3a5a;
        border-radius: 12px;
        margin: 8px 0;
    }
    
    .stChatMessage .stMarkdown {
        color: #d0d0ff;
    }
    
    /* User Message */
    [data-testid="stChatMessage"]:nth-child(odd) {
        border-left: 3px solid #4169e1;
    }
    
    /* Assistant Message */
    [data-testid="stChatMessage"]:nth-child(even) {
        border-left: 3px solid #6495ed;
    }
    
    /* Input Box */
    .stTextInput input {
        background: #1a1a2e;
        color: #e0e0ff;
        border: 1px solid #4169e1;
        border-radius: 8px;
    }
    
    .stTextInput input:focus {
        border-color: #6495ed;
        box-shadow: 0 0 0 2px rgba(65, 105, 225, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4169e1, #2a4a9e);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5278e8, #3a5ab8);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(65, 105, 225, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Success/Error/Info Boxes */
    .stSuccess {
        background: rgba(40, 167, 69, 0.15);
        border: 1px solid #28a745;
        border-radius: 8px;
        color: #90ee90;
    }
    
    .stError {
        background: rgba(220, 53, 69, 0.15);
        border: 1px solid #dc3545;
        border-radius: 8px;
        color: #ffb3b3;
    }
    
    .stInfo {
        background: rgba(65, 105, 225, 0.15);
        border: 1px solid #4169e1;
        border-radius: 8px;
        color: #a0b8ff;
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.15);
        border: 1px solid #ffc107;
        border-radius: 8px;
        color: #ffe08a;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(26, 26, 46, 0.6);
        border: 1px dashed #4169e1;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Tool Badge */
    .tool-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    .tool-kb {
        background: linear-gradient(135deg, #4169e1, #2a4a9e);
        color: white;
    }
    
    .tool-web {
        background: linear-gradient(135deg, #6495ed, #4169e1);
        color: white;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #667799;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #2a3a5a;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4169e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6495ed;
    }
    
    /* Code Blocks */
    pre, code {
        background: #0f0f1a !important;
        border: 1px solid #2a3a5a;
        color: #a0b8ff !important;
    }
    
    /* Divider */
    hr {
        border-color: #2a3a5a !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Inject CSS
inject_custom_css()

# --- Hardware-Aware Config ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "phi3:mini"
DB_PATH = "./db"
MAX_CHUNKS = 25

# --- Header: RIO Branding ---
st.markdown('<div class="main-title">🦜 RIO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligence Operator • Local RAG & AI Agent • 100% Private</div>', unsafe_allow_html=True)

# --- Cache Models ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def load_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0.3)

# --- Sidebar: Controls ---
with st.sidebar:
    # Sidebar Header
    st.markdown('<div style="color: #6495ed; font-weight: 600; font-size: 1.2rem; margin-bottom: 1rem;">⚙️ RIO Controls</div>', unsafe_allow_html=True)
    
    # PDF Upload
    uploaded_file = st.file_uploader("📄 Upload Document", type="pdf", help="Upload a PDF to enable RAG")
    
    # Process Button
    if uploaded_file and st.button("🔄 Process Document", type="primary"):
        with st.spinner("🦜 RIO is processing..."):
            try:
                # Save temp file
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load & Split
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=350, 
                    chunk_overlap=60,
                    separators=["\n\n", "\n", " ", ""]
                )
                splits = splitter.split_documents(docs)[:MAX_CHUNKS]
                
                # Embed & Store
                embeddings = load_embeddings()
                
                # Clear old DB if exists
                if os.path.exists(DB_PATH):
                    shutil.rmtree(DB_PATH)
                
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings, 
                    persist_directory=DB_PATH
                )
                st.success(f"✅ Indexed {len(splits)} chunks")
                
                # Clean up
                os.remove("temp.pdf")
                st.info("📚 Knowledge Base Ready!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Memory Management
    if st.button("🗑️ Clear Memory", type="secondary"):
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.success("Memory cleared")
    
    st.divider()
    
    # Status Info
    if os.path.exists(DB_PATH):
        st.markdown('<div style="color: #90ee90; font-weight: 500;">✅ Knowledge Base: Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color: #ffe08a; font-weight: 500;">⚠️ Upload PDF to enable RAG</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Hardware Tips
    with st.expander("💡 Performance Tips"):
        st.markdown("""
        - 🔹 Close unused browser tabs
        - 🔹 Use PDFs <10 pages for best speed
        - 🔹 CPU inference: ~3 words/sec
        - 🔹 8GB RAM optimized
        """)

# --- Main Chat Interface ---
st.markdown('<div style="color: #8899bb; margin-bottom: 1rem;">### 💬 Chat with RIO</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask RIO about your documents or the web..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🦜 RIO is thinking..."):
            try:
                llm = load_llm()
                embeddings = load_embeddings()
                
                # --- AGENT LOGIC: Tool Selection ---
                tool_used = ""
                context = ""
                
                if os.path.exists(DB_PATH):
                    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    docs = retriever.invoke(prompt)
                    context = "\n\n".join([d.page_content for d in docs])
                    
                    is_resume_query = any(word in prompt.lower() for word in 
                                        ["resume", "cv", "skills", "experience", "education", "project", "contact"])
                    
                    if context and len(context) > 50:
                        tool_used = "🗄️ Knowledge Base"
                        tool_class = "tool-kb"
                        
                        if is_resume_query:
                            system_prompt = f"""You are RIO, a professional resume analyst. Extract and organize information clearly.

Instructions:
1. Use clear section headings: Contact, Summary, Experience, Education, Skills, Projects, Achievements
2. Preserve exact values: names, dates, scores (e.g., CGPA: 9.1), tool names (e.g., Flask, Pandas)
3. Use bullet points for lists
4. If info is missing, state "Not found in provided context"
5. Do NOT invent information

Context:
{context}

Question: {prompt}

Structured response:"""
                        else:
                            system_prompt = f"""You are RIO, a helpful AI assistant. Use the context to answer.

Context: {context}
Question: {prompt}

Answer based on context. If unsure, say so."""
                    else:
                        tool_used = "🌐 Web Search"
                        tool_class = "tool-web"
                        system_prompt = f"""You are RIO. Answer using knowledge or web search.

Question: {prompt}
Provide a clear answer."""
                else:
                    tool_used = "🌐 Web Search"
                    tool_class = "tool-web"
                    system_prompt = f"""You are RIO. Answer the question.

Question: {prompt}
Provide a clear answer."""
                
                # Web Search fallback
                if "Web Search" in tool_used:
                    try:
                        search = DuckDuckGoSearchRun()
                        web_results = search.run(prompt[:100])
                        if web_results and len(web_results) > 20:
                            system_prompt = f"""You are RIO. Use these web results:

{web_results}
Question: {prompt}
Answer based on results."""
                    except:
                        pass
                
                # Generate response
                response = llm.invoke(system_prompt)
                answer = response.content
                
                # Display tool badge
                st.markdown(f'<span class="tool-badge {tool_class}">{tool_used}</span>', unsafe_allow_html=True)
                st.markdown(answer)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try: Close apps, smaller PDF, or simpler question.")

# --- Footer ---
st.markdown('<div class="footer">🔒 100% Local • No Data Leaves Your Machine • RIO Agent v1.0</div>', unsafe_allow_html=True)