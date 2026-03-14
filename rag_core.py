# rag_core.py - STABLE VERSION
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama

# --- Configuration for 8GB RAM ---
PDF_PATH = "sample.pdf"
DB_PATH = "./db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "phi3:mini"

def main():
    print("🚀 Starting EdgeMind RAG Pipeline...")
    
    # 1. Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found. Please add a PDF to the folder.")
        return

    # 2. Load Document
    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"   Loaded {len(docs)} pages.")

    # 3. Split Text (Chunking)
    print("✂️ Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350, 
        chunk_overlap=60,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = splitter.split_documents(docs)
    print(f"   Created {len(splits)} chunks.")

    # 4. Create Embeddings
    print("🧠 Generating Embeddings (This may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # 5. Store in Vector DB
    print("💾 Saving to Vector Database...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print("   Database saved to './db'")

    # 6. Setup LLM
    print("🤖 Connecting to Local LLM (Ollama)...")
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)

    # 7. Query Test
    print("\n--- Ready to Query ---")
    query = "What are the main skills and experience mentioned?"
    
    # Retrieve relevant chunks (k=5 for completeness)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)
    
    # Create context string
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Resume-aware prompt
    prompt = f"""You are a professional resume analyst. Extract and organize information clearly.

Instructions:
1. Use clear section headings: Contact, Summary, Experience, Education, Skills, Projects
2. Preserve exact values: names, dates, scores, tool names
3. Use bullet points for lists
4. If info is missing, state "Not found in context"

Context: {context}
Question: {query}

Structured response:"""
    
    print(f"🗣️ Asking: {query}")
    response = llm.invoke(prompt)
    print(f"💡 Answer:\n{response.content}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        print("💡 Tip: Close other apps to free up RAM.")