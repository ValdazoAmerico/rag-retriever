# Imports - Librerías estándar primero
import os

# Imports - Librerías de terceros
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Imports - Google ADK
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm

# ========================
# CONFIGURACIÓN
# ========================

# Cargar variables de entorno
load_dotenv()

# Acceder a las variables de entorno
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configurar variables de entorno
os.environ["OPENAI_API_KEY"] = openai_api_key

# ========================
# CONFIGURACIÓN DE EMBEDDINGS Y VECTOR STORE
# ========================

# Configurar embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # dimensions=1024  # podés descomentar si querés controlar dimensiones
)

# Configurar path de la base de datos
path = "chroma_db"

# Cargar la base Chroma existente
vectorstore = Chroma(
    persist_directory=path,
    embedding_function=embeddings
)

# ========================
# VERIFICACIÓN DE FUNCIONAMIENTO
# ========================

# Verificar recuperación de chunks similares
results = vectorstore.similarity_search("exito", k=1)
for i, doc in enumerate(results):
    print(f"\n🔹 Resultado #{i + 1}")
    print("Contenido:", doc.page_content[:200], "...")
    print("Metadatos:", doc.metadata)

# ========================
# DEFINICIÓN DE HERRAMIENTAS
# ========================

def retriever(query: str) -> dict:
    """Retrieve relevant documents from the vector store based on the query."""
    results = vectorstore.similarity_search(query, k=1)
    if results:
        return {"status": "success", "content": results[0].page_content}
    else:
        return {"status": "error", "message": "No relevant documents found."}

# Crear la herramienta
retriever_tool = FunctionTool(func=retriever)

# ========================
# CONFIGURACIÓN DEL AGENTE
# ========================

# Crear el agente
root_agent = Agent(
    model=LiteLlm(model="openai/gpt-4.1-mini", temperature=0),
    name='project_agent',
    instruction="""Usa la tool `retriever` para obtener información relevante de la base de conocimiento sobre gestión de proyectos.""",
    tools=[retriever_tool]
)