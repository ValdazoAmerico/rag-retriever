from langchain_openai import OpenAIEmbeddings

def create_embeddings_model():
    print("🧠 Configurando modelo de embeddings...")
    return OpenAIEmbeddings(model="text-embedding-3-small")