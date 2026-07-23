import os 

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Validar que la API Key de Cohere exista
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("No se encontró COHERE_API_KEY en el archivo .env. Agrégala para continuar.")


def load_documents(docs_path="docs"):
    """Carga documentos PDF y Excel del directorio indicado."""
    print(f"Buscando documentos en el directorio: {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"El directorio '{docs_path}' no existe. Créalo y agrega archivos.")
    
    loaders = {
        ".pdf": PyMuPDFLoader,
        ".xlsx": UnstructuredExcelLoader,
    }
    
    documents = []
    for ext, loader_cls in loaders.items():
        loader = DirectoryLoader(
            docs_path,
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            show_progress=True
        )
        loaded_docs = loader.load()
        documents.extend(loaded_docs)
        print(f"Cargados {len(loaded_docs)} documentos con extensión {ext}")
        
    if len(documents) == 0:
        raise FileNotFoundError(f"No se encontraron documentos en '{docs_path}'.")  

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    """Divide los documentos en chunks para optimizar el contexto en RAG."""
    print(f"Dividiendo documentos en chunks (tamaño: {chunk_size}, solapamiento: {chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Se han creado un total de {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma"):
    """Crea un vector store persistente usando Cohere Embeddings Multilingüe."""
    print(f"Creando vector store en: {persist_directory}...")

    # Modelo multilingüe optimizado para búsqueda semántica / RAG
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    print("Generando embeddings e indexando en Chroma DB...")

    # Cohere procesa todos los chunks directamente gracias a su cuota de 1,000 req/min
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print("✅ Vector store creado e indexado con éxito.")
    return vector_store


def main():
    print("Iniciando proceso de ingesta con Cohere...")
    documents = load_documents(docs_path="docs")
    chunks = split_documents(documents)
    create_vector_store(chunks)


if __name__ == "__main__":
    main()