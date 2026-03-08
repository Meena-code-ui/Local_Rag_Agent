import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama

PDF_PATH = "Meenalochani_CV.pdf"
DB_PATH = "./db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "phi3:mini"

def main():
    print("Starting  RAG Pipeline")

    if not os.path.exists(PDF_PATH):
        print("PDF PATH Not exists")
        return
    
    print("Loading PDF")
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    print("splitting text")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    splits = splitter.split_documents(docs)
    print(f"create {len(splits)} chunks")

    print("generating embeddings")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("storing to vector db")
    vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings,persist_directory=DB_PATH)
    print("database saved to .db/")

    print("connecting to local llm model")
    llm = ChatOllama(model=LLM_MODEL,temperature=0.3)

    print("ready to query")
    query = "what is this document about?"

    retriever = vectorstore.as_retriever(search_kwargs={"k":2})
    relevant_docs= retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""Use the following context to answer the question
    Context: {context}
    Question : {query}
    Answer"""

    print(f"asking : {query}")
    response = llm.invoke(prompt)
    print(f"Answer :{response.content}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical Error:{e}")
  


