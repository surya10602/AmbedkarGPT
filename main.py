import sys

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def main():
    print("--- Initialize AmbedkarGPT ---")

    try:
        loader = TextLoader("speech.txt")
        documents = loader.load()
    except FileNotFoundError:
        print("Error: 'speech.txt' not found. Please create the file first.")
        return

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print("Loading Embeddings model... (This may take a moment)")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating Vector Store...")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    llm = Ollama(model="mistral")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=False
    )

    print("\n System Ready! Ask a question based on the text.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("User: ")
        if query.lower() in ['exit', 'quit']:
            break
        
        try:
            response = qa_chain.invoke({"query": query})
            print(f"AmbedkarGPT: {response['result']}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()