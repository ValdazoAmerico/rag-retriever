from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_text_chunks(text: str, chunk_size: int = 400, chunk_overlap: int = 0) -> list:
    """
    Split a text string into smaller chunks using a recursive character-based approach.

    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int, optional): Maximum number of characters per chunk. Defaults to 400.
        chunk_overlap (int, optional): Number of overlapping characters between consecutive chunks. Defaults to 0.

    Returns:
        list: List of text chunks as strings.
    """
    print("🔧 Dividiendo el texto en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    print(f"📄 Total de chunks generados: {len(chunks)}")
    return chunks