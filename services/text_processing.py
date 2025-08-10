from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_text_chunks(text: str, chunk_size: int = 400, chunk_overlap: int = 0) -> list:
    print("🔧 Dividiendo el texto en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    print(f"📄 Total de chunks generados: {len(chunks)}")
    return chunks