from langchain.vectorstores import Chroma

def create_and_persist_vectorstore(documents: list, embeddings, persist_directory: str):
    """
    Create and persist a Chroma vector store from provided documents and embeddings.

    Args:
        documents (list): List of LangChain `Document` objects to store.
        embeddings: Embeddings model used to convert text into vector representations.
        persist_directory (str): Path to the directory where the vector store will be persisted.

    Returns:
        Chroma: A persisted Chroma vector store containing the documents and their embeddings.
    """
    print("📦 Guardando documentos en Chroma DB...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    print(f"✅ Chroma DB persistido exitosamente en '{persist_directory}'")
    return vectorstore