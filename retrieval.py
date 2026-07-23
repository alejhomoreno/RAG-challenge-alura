import os
from dotenv import load_dotenv
import cohere
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.prompts import PromptTemplate

load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("No se encontró COHERE_API_KEY en el archivo .env.")

# Cliente nativo de Cohere para Reranking
co_client = cohere.Client(api_key=cohere_api_key)

# 1. Instanciar embeddings y Chroma DB
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
vector_store = Chroma(
    persist_directory="db/chroma",
    embedding_function=embeddings
)

# Traemos 10 chunks por consulta para darle un espectro más amplio al Reranker
base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# 2. Instanciar LLM de Cohere
llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)

prompt_template = PromptTemplate.from_template(
    "Eres un experto en cadenas de suministro. Reescribe la siguiente pregunta del usuario "
    "generando 3 variaciones usando vocabulario corporativo y sinónimos (ej. dispersión geográfica, "
    "cobertura territorial, proveedores, etc.) para mejorar una búsqueda vectorial. "
    "Responde SOLO con las 3 variaciones separadas por saltos de línea, sin viñetas ni introducciones:\n"
    "Pregunta: {question}"
)

def consultar_con_rerank(query_texto):
    # Generar variaciones
    prompt = prompt_template.format(question=query_texto)
    respuesta = llm.invoke(prompt).content.strip()
    variaciones = respuesta.split("\n")
    
    todas_las_consultas = [query_texto] + [v.strip("- *") for v in variaciones if v.strip()]
    
    print("🔍 Recuperando candidatos iniciales de Chroma DB...")
    
    docs_candidatos = []
    vistos = set()
    
    for q in todas_las_consultas:
        docs = base_retriever.invoke(q)
        for doc in docs:
            if doc.page_content not in vistos:
                vistos.add(doc.page_content)
                docs_candidatos.append(doc)
                
    print(f"📦 Total de chunks candidatos recuperados: {len(docs_candidatos)}")
    
    # Aplicar Cohere Rerank v3 para reordenar por relevancia real
    textos_candidatos = [d.page_content for d in docs_candidatos]
    
    print("⚡ Aplicando Cohere Rerank...")
    rerank_response = co_client.rerank(
        model="rerank-multilingual-v3.0",
        query=query_texto,
        documents=textos_candidatos,
        top_n=4  # Quedarnos solo con los 4 mejores tras el Rerank
    )
    
    docs_finales = []
    for hit in rerank_response.results:
        docs_finales.append(docs_candidatos[hit.index])
        
    return docs_finales

# 4. Ejecución principal
query = "Si un fabricante directo no puede cubrir la entrega en ciertas zonas alejadas, ¿qué alternativa propone el documento y qué documento debe presentar para registrarse?"

print("Iniciando pipeline con Reranking...\n")
retrieved_docs = consultar_con_rerank(query)

print(f"\n--- Contexto final (Top 4 reordenados por Rerank) ---")
for i, doc in enumerate(retrieved_docs):
    print(f"\nDocumento {i + 1}:")
    print(doc.page_content)
    print("-" * 40)