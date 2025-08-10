from langchain_openai import OpenAIEmbeddings

def create_embeddings_model():
    """
    Initialize an OpenAI embeddings model.

    Returns:
        OpenAIEmbeddings: Configured embeddings model using the
        "text-embedding-3-small" architecture for vector representation of text.
    """
    print("🧠 Configurando modelo de embeddings...")
    return OpenAIEmbeddings(model="text-embedding-3-small")