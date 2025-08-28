import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")

# Embeddings setup
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

INDEX_PATH = "faiss_index"

def load_documents(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return loader.load()

def create_or_load_vectorstore(documents=None):
    if os.path.exists(INDEX_PATH):
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.split_documents(documents)
            vectorstore.add_documents(docs)
            vectorstore.save_local(INDEX_PATH)
    else:
        if not documents:
            raise ValueError("No documents provided for new FAISS index")
        print("Creating new FAISS index...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(INDEX_PATH)
    return vectorstore

def query_rag(vectorstore, query):
    llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant context found."

    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer concisely based on the following context:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)

    return response.text if hasattr(response, "text") else response

if __name__ == "__main__":
    file_path = "sample_rental.pdf"  # change to your doc
    docs = load_documents(file_path)
    vectorstore = create_or_load_vectorstore(docs)

    print("\n Document loaded. Start asking questions (type 'exit' or 'quit' to stop).")

    while True:
        query = input("\n Your Question: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print(" Exiting. Goodbye!")
            break
        answer = query_rag(vectorstore, query)
        print(f" Answer: {answer}")
