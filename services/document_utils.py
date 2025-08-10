from langchain.schema import Document

def create_documents_with_metadata(chunks: list, source: str, document_id: str) -> list:
    """
    Create LangChain `Document` objects with associated metadata.

    Args:
        chunks (list): List of text chunks to be converted into `Document` objects.
        source (str): Source of the document (e.g., file path, URL).
        document_id (str): Unique identifier for the document.

    Returns:
        list: A list of `Document` objects, each containing the chunk text and metadata:
            - chunk_id (int): Sequential index of the chunk.
            - source (str): Provided source of the document.
            - document_id (str): Provided unique document identifier.
    """
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