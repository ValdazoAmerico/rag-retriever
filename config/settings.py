import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

PERSIST_DIRECTORY = "chroma_db"
SOURCE_FILE = "data/knowledge_base_text.txt"
DOCUMENT_ID = "123"