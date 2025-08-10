from langchain.vectorstores import Chroma

def create_and_persist_vectorstore(documents: list, embeddings, persist_directory: str):
    print("📦 Guardando documentos en Chroma DB...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    print(f"✅ Chroma DB persistido exitosamente en '{persist_directory}'")
    return vectorstore