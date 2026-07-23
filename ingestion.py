import os 
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Validar que la API Key de Cohere exista
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("No se encontró COHERE_API_KEY en el archivo .env. Agrégala para continuar.")


def load_documents(docs_path="docs"):
    """Carga documentos PDF y Excel del directorio indicado de forma ligera."""
    print(f"Buscando documentos en el directorio: {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"El directorio '{docs_path}' no existe. Créalo y agrega archivos.")
    
    documents = []

    # 1. Cargar PDFs
    try:
        pdf_loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader, show_progress=True)
        docs_pdf = pdf_loader.load()
        documents.extend(docs_pdf)
        print(f"Cargados {len(docs_pdf)} documentos PDF.")
    except Exception as e:
        print(f"Error cargando PDFs: {e}")

    # 2. Cargar Excel con Pandas (Sin depender de unstructured)
    excel_files = [os.path.join(docs_path, f) for f in os.listdir(docs_path) if f.endswith('.xlsx')]
    for file in excel_files:
        try:
            df = pd.read_excel(file)
            # Convertir cada fila en un documento inyectando el nombre de la columna
            for index, row in df.iterrows():
                content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                documents.append(Document(page_content=content, metadata={"source": file, "row": index}))
            print(f"Cargado Excel {file} sumando {len(df)} filas como documentos.")
        except Exception as e:
            print(f"Error procesando el Excel {file}: {e}")
        
    if len(documents) == 0:
        raise FileNotFoundError(f"No se encontraron documentos procesables en '{docs_path}'.")  

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    """Divide los documentos en chunks para optimizar el contexto en RAG."""
    print(f"Dividiendo documentos en chunks (tamaño: {chunk_size}, solapamiento: {chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    print(f"Se han creado un total de {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma"):
    """Crea un vector store persistente usando Cohere Embeddings Multilingüe."""
    print(f"Creando vector store en: {persist_directory}...")
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    print("Generando embeddings e indexando en Chroma DB...")
    
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