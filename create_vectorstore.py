from config.settings import PERSIST_DIRECTORY, SOURCE_FILE, DOCUMENT_ID
from services.text_processing import create_text_chunks
from services.document_utils import create_documents_with_metadata
from services.embeddings_utils import create_embeddings_model
from services.vectorstore_utils import create_and_persist_vectorstore

# Cargar texto base
with open(SOURCE_FILE, "r", encoding="utf-8") as f:
    knowledge_base_text = f.read()

print("✅ Iniciando procesamiento del texto base...")

# Paso 1
chunks = create_text_chunks(knowledge_base_text)

# Paso 2
documents = create_documents_with_metadata(chunks, SOURCE_FILE, DOCUMENT_ID)

# Paso 3
embeddings = create_embeddings_model()

# Paso 4
vectorstore = create_and_persist_vectorstore(documents, embeddings, PERSIST_DIRECTORY)

print("🏁 Proceso finalizado correctamente.")