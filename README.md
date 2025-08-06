# 🧠 Proyecto RAG + ADK

Este proyecto implementa un sistema de **Recuperación Aumentada por Generación (RAG)** utilizando una base de datos vectorial (Chroma) y un agente conversacional mediante Google ADK.

## 📋 Tabla de Contenidos

- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)

## 🚀 Requisitos

- **Python 3.10+**
- **Acceso a la API de OpenAI** (`OPENAI_API_KEY`)
- Dependencias listadas en `requirements.txt`

## Instalación

Linux/Mac:

```bash
git clone https://github.com/ValdazoAmerico/rag-retriever.git && \
cd rag-retriever && \
python -m venv venv && \
source venv/bin/activate && \
pip install --upgrade pip && \
pip install -r requirements.txt
```

## 🔐 Configuración

### Variables de entorno

Crea un archivo `.env` en el directorio raíz del proyecto:

```env
OPENAI_API_KEY=tu_clave_openai_aqui
```

## 🛠️ Uso

### Paso 1: Crear la base de datos vectorial

Ejecuta el script para procesar los documentos y construir la base de datos vectorial:

```bash
python create_vectorstore.py
```

Este comando:
- ✅ Procesa el texto de la base de conocimientos
- ✅ Genera embeddings usando OpenAI
- ✅ Crea una base de datos Chroma persistente

### Paso 2: Ejecutar el agente

Una vez construida la base de datos, lanza la interfaz para interactuar con el agente:

```bash
adk web
```

💡 Una vez que se abra la interfaz gráfica de ADK en tu navegador, selecciona "agent" en la pestaña superior izquierda para interactuar con el agente configurado.

## 🔧 Funcionalidades

- **🔍 Búsqueda semántica** en base de conocimientos
- **💬 Agente conversacional** inteligente
- **📊 Embeddings** con OpenAI text-embedding-3-small
- **💾 Persistencia** de datos vectoriales con Chroma
- **🌐 Interfaz web** para interacción