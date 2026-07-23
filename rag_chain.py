import os
import cohere
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Cargar variables de entorno
load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("No se encontró COHERE_API_KEY en el archivo .env.")

# Clientes e instancias
co_client = cohere.Client(api_key=cohere_api_key)
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
vector_store = Chroma(
    persist_directory="db/chroma",
    embedding_function=embeddings
)

base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)

# 2. Función de Retrieval + Reranking (nuestro motor probado)
def obtener_contexto_reranked(query_texto):
    # Generar variaciones para Multi-Query
    prompt_variaciones = ChatPromptTemplate.from_template(
        "Reescribe la siguiente consulta generando 3 variaciones profesionales con sinónimos en español "
        "para mejorar una búsqueda vectorial. Responde SOLO con las variaciones separadas por salto de línea:\n{question}"
    )
    cadena_variaciones = prompt_variaciones | llm | StrOutputParser()
    
    variaciones_raw = cadena_variaciones.invoke({"question": query_texto})
    variaciones = [v.strip("- *") for v in variaciones_raw.split("\n") if v.strip()]
    todas_las_consultas = [query_texto] + variaciones

    # Buscar candidatos
    docs_candidatos = []
    vistos = set()
    for q in todas_las_consultas:
        docs = base_retriever.invoke(q)
        for doc in docs:
            if doc.page_content not in vistos:
                vistos.add(doc.page_content)
                docs_candidatos.append(doc)

    # Aplicar Reranking
    textos = [d.page_content for d in docs_candidatos]
    rerank_response = co_client.rerank(
        model="rerank-multilingual-v3.0",
        query=query_texto,
        documents=textos,
        top_n=4
    )

    docs_finales = [docs_candidatos[hit.index] for hit in rerank_response.results]
    return "\n\n---\n\n".join([d.page_content for d in docs_finales])

# 3. Prompt de RAG para la generación de la respuesta
system_prompt = (
    "Eres un asistente virtual experto de Mercado Central 24h. "
    "Responde a la pregunta del usuario utilizando ÚNICA Y EXCLUSIVAMENTE el contexto proporcionado a continuación. "
    "Si la información no se encuentra explícitamente en el contexto, indica amablemente que no dispones de dicha información. "
    "Sé preciso, profesional y directo.\n\n"
    "--- CONTEXTO DISPONIBLE ---\n"
    "{context}"
)

prompt_rag = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# Cadena RAG completa
rag_chain = prompt_rag | llm | StrOutputParser()

def preguntar(pregunta_usuario):
    print(f"\n🔍 Procesando pregunta: '{pregunta_usuario}'")
    print("⏳ Recuperando y reordenando contexto...")
    
    contexto = obtener_contexto_reranked(pregunta_usuario)
    
    print("🤖 Generando respuesta final...")
    respuesta = rag_chain.invoke({
        "context": contexto,
        "question": pregunta_usuario
    })
    
    print("\n" + "="*50)
    print("RESPUESTA DEL SISTEMA RAG:")
    print("="*50)
    print(respuesta)
    print("="*50 + "\n")

if __name__ == "__main__":
    pregunta_prueba = "Si un fabricante directo no puede cubrir la entrega en ciertas zonas alejadas, ¿qué alternativa propone el documento y qué documento debe presentar para registrarse?"
    preguntar(pregunta_prueba)