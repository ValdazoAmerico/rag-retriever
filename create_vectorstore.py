# Imports - Librerías estándar
import os

# Imports - Librerías de terceros
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# ========================
# CONFIGURACIÓN INICIAL
# ========================

# Cargar variables de entorno
load_dotenv()

# Configurar API key de OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Configuración de archivos y directorios
PERSIST_DIRECTORY = "chroma_db"
SOURCE_FILE = "knowledge_base_text.txt"
DOCUMENT_ID = "123"

# ========================
# DATOS DE CONOCIMIENTO BASE
# ========================

knowledge_base_text = """
El éxito de un proyecto en nuestra agencia depende de tres pilares fundamentales: la comunicación clara, la gestión del tiempo y la rentabilidad.

La comunicación clara se logra manteniendo reuniones de seguimiento semanales y utilizando el canal de Slack #proyectos-activos para actualizaciones diarias. Es crucial documentar todas las decisiones importantes en la plataforma de COR.

Para la gestión del tiempo, cada tarea debe ser estimada en horas antes de iniciar un sprint. Utilizamos la técnica Pomodoro para mantener el enfoque y registramos el tiempo de cada actividad en COR para asegurar que nos mantenemos dentro del presupuesto. El Project Manager es responsable de ajustar el cronograma si surgen desvíos.

La rentabilidad se monitorea constantemente a través del dashboard de COR. Cualquier proyecto que caiga por debajo del 20% de margen de ganancia requiere una revisión inmediata. Los costos inesperados deben ser aprobados por el cliente antes de ser incurridos para evitar sorpresas en la facturación.
"""

# ========================
# FUNCIONES PRINCIPALES
# ========================

def create_text_chunks(text: str, chunk_size: int = 400, chunk_overlap: int = 0) -> list:
    """Divide el texto en chunks usando RecursiveCharacterTextSplitter."""
    print("🔧 Dividiendo el texto en chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    print(f"📄 Total de chunks generados: {len(chunks)}")
    
    return chunks

def create_documents_with_metadata(chunks: list, source: str, document_id: str) -> list:
    """Crea objetos Document con metadatos a partir de los chunks."""
    print("🧾 Creando objetos Document con metadatos...")
    
    documents = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id": i,
            "source": source,
            "document_id": document_id,
            # Podés agregar más campos si los tenés
        }
        documents.append(Document(page_content=chunk, metadata=metadata))
    
    print("✅ Documentos con metadatos creados.")
    return documents

def create_embeddings_model():
    """Configura el modelo de embeddings de OpenAI."""
    print("🧠 Configurando modelo de embeddings...")
    
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        # dimensions=1024  # podés descomentar si querés controlar dimensiones
    )

def create_and_persist_vectorstore(documents: list, embeddings, persist_directory: str):
    """Crea el vectorstore de Chroma y lo persiste."""
    print("📦 Guardando documentos en Chroma DB...")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    
    # Persistir vectorstore
    vectorstore.persist()
    print(f"✅ Chroma DB persistido exitosamente en '{persist_directory}'")
    
    return vectorstore

# ========================
# PROCESO PRINCIPAL
# ========================

print("✅ Iniciando procesamiento del texto base...")
    
    # Paso 1: Dividir el texto en chunks
text_chunks = create_text_chunks(knowledge_base_text)
    
    # Paso 2: Crear documentos con metadatos
documents = create_documents_with_metadata(
        text_chunks, 
        SOURCE_FILE, 
        DOCUMENT_ID
    )
    
    # Paso 3: Crear modelo de embeddings
embeddings = create_embeddings_model()
    
    # Paso 4: Crear y persistir vectorstore
vectorstore = create_and_persist_vectorstore(
        documents, 
        embeddings, 
        PERSIST_DIRECTORY
    )
    
print("🏁 Proceso finalizado correctamente.")
