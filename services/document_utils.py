from langchain.schema import Document

def create_documents_with_metadata(chunks: list, source: str, document_id: str) -> list:
    print("🧾 Creando objetos Document con metadatos...")
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_id": i,
                "source": source,
                "document_id": document_id,
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    print("✅ Documentos con metadatos creados.")
    return documents