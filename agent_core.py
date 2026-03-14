# agent_core.py - STABLE VERSION
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun

# --- Configuration ---
PDF_PATH = "sample.pdf"
DB_PATH = "./db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "phi3:mini"

def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def load_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0.3)

def get_resume_prompt(context, query):
    """Resume-aware structured prompt"""
    return f"""You are a professional resume analyst. Extract and organize information clearly.

Instructions:
1. Use clear section headings: Contact, Summary, Experience, Education, Skills, Projects, Achievements
2. Preserve exact values: names, dates, scores (e.g., CGPA: 9.1), tool names (e.g., Flask, Pandas)
3. Use bullet points for lists
4. If info is missing from context, state "Not found in provided context"
5. Do NOT invent or assume information

Context from document:
{context}

User question: {query}

Provide a well-structured, complete response with clear sections:"""

def main():
    print("🤖 Starting EdgeMind Agent...")
    
    llm = load_llm()
    embeddings = load_embeddings()
    
    # Setup RAG if DB exists
    if os.path.exists(DB_PATH):
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Higher k for resumes
        print("✅ Knowledge Base Loaded")
    else:
        retriever = None
        print("⚠️ No Knowledge Base (run rag_core.py first)")
    
    # Web Search Tool
    search = DuckDuckGoSearchRun()
    print("✅ Web Search Tool Loaded")
    
    print("\n--- Agent Ready! Type 'quit' to exit ---")
    print("💡 Try: 'What are my top skills?' or 'Summarize my experience'")
    
    while True:
        query = input("\n🗣️ You: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        print("🤖 Agent: ", end="", flush=True)
        try:
            tool_used = ""
            context = ""
            
            # Try RAG first
            if retriever:
                docs = retriever.invoke(query)
                context = "\n\n".join([d.page_content for d in docs])
                if context and len(context) > 50:
                    tool_used = "Knowledge Base"
            
            # Fallback to Web Search
            if not context or len(context) < 50:
                try:
                    web_results = search.run(query[:100])
                    if web_results and len(web_results) > 20:
                        context = f"Web Search:\n{web_results}"
                        tool_used = "Web Search"
                except:
                    pass
            
            # Build prompt
            is_resume = any(w in query.lower() for w in ["resume", "cv", "skills", "experience", "education"])
            if context and is_resume:
                prompt = get_resume_prompt(context, query)
            elif context:
                prompt = f"""Use the following context to answer.
Context: {context}
Question: {query}
Answer:"""
            else:
                prompt = f"""Answer the question.
Question: {query}
Answer:"""
            
            response = llm.invoke(prompt)
            print(f"\n[{tool_used}] {response.content}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Agent stopped.")