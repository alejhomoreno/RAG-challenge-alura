import os
import cohere
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Cargar configuración y API Keys
load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("No se encontró COHERE_API_KEY en el archivo .env.")

# Instancias principales
co_client = cohere.Client(api_key=cohere_api_key)
embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
db = Chroma(persist_directory="db/chroma", embedding_function=embeddings)
base_retriever = db.as_retriever(search_kwargs={"k": 10})
llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)

# Memoria conversacional
chat_history = []

def obtener_contexto_avanzado(pregunta_usuario):
    """
    Combina Multi-Query y Reranking para obtener el contexto más relevante.
    """
    # 1. Generar variaciones de la pregunta (Multi-Query)
    prompt_variaciones = ChatPromptTemplate.from_template(
        "Reescribe la siguiente consulta generando 3 variaciones profesionales con sinónimos en español "
        "para mejorar una búsqueda vectorial. Responde SOLO con las variaciones separadas por salto de línea:\n{question}"
    )
    cadena_variaciones = prompt_variaciones | llm | StrOutputParser()
    variaciones_raw = cadena_variaciones.invoke({"question": pregunta_usuario})
    variaciones = [v.strip("- *") for v in variaciones_raw.split("\n") if v.strip()]
    todas_las_consultas = [pregunta_usuario] + variaciones

    # 2. Recuperar candidatos acumulados
    docs_candidatos = []
    vistos = set()
    for q in todas_las_consultas:
        docs = base_retriever.invoke(q)
        for doc in docs:
            if doc.page_content not in vistos:
                vistos.add(doc.page_content)
                docs_candidatos.append(doc)

    if not docs_candidatos:
        return ""

    # 3. Reordenar con Cohere Rerank v3
    textos = [d.page_content for d in docs_candidatos]
    rerank_response = co_client.rerank(
        model="rerank-multilingual-v3.0",
        query=pregunta_usuario,
        documents=textos,
        top_n=4
    )

    docs_finales = [docs_candidatos[hit.index] for hit in rerank_response.results]
    return "\n\n---\n\n".join([d.page_content for d in docs_finales])


def ask_question(user_question):
    print(f"\n💬 Pregunta recibida: '{user_question}'")

    # Paso 1: Reformular la pregunta usando el historial de chat si existe
    if chat_history:
        print("🧠 Reformulando pregunta con base en el historial conversacional...")
        prompt_reformular = (
            "Dada la siguiente conversación previa y una nueva pregunta del usuario, "
            "reformula la pregunta para que sea independiente y clara por sí misma. "
            "NO la respondas, solo devuélvela reformulada en español.\n\n"
            f"Pregunta nueva: {user_question}"
        )
        mensajes_reformulacion = [SystemMessage(content="Eres un asistente experto en comprensión de contexto.")] + chat_history + [HumanMessage(content=prompt_reformular)]
        
        pregunta_busqueda = llm.invoke(mensajes_reformulacion).content.strip()
        print(f"🔎 Pregunta independiente generada: '{pregunta_busqueda}'")
    else:
        pregunta_busqueda = user_question

    # Paso 2: Ejecutar el motor Multi-Query + Reranking
    print("⏳ Recuperando y reordenando contexto (Multi-Query + Rerank)...")
    contexto = obtener_contexto_avanzado(pregunta_busqueda)

    # Paso 3: Generar la respuesta usando el RAG estricto
    prompt_sistema = (
        "Eres un asistente virtual experto de Mercado Central 24h.\n"
        "Responde a la pregunta utilizando ÚNICA Y EXCLUSIVAMENTE el contexto proporcionado.\n"
        "Si la respuesta no se encuentra en el contexto, indica educadamente que no dispones de dicha información en el manual.\n"
        "Mantén un tono profesional, claro y directo.\n\n"
        f"--- CONTEXTO DISPONIBLE ---\n{contexto}"
    )

    mensajes_respuesta = [SystemMessage(content=prompt_sistema)] + chat_history + [HumanMessage(content=user_question)]

    print("🤖 Generando respuesta final...")
    respuesta = llm.invoke(mensajes_respuesta).content.strip()

    # Paso 4: Actualizar el historial conversacional
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=respuesta))

    print("\n" + "="*60)
    print("RESPUESTA:")
    print("="*60)
    print(respuesta)
    print("="*60 + "\n")
    
    return respuesta


def start_chat():
    print("\n🚀 ¡Sistema RAG Conversacional de Mercado Central 24h iniciado!")
    print("Escribe tus preguntas o presiona 'salir' para terminar.\n")

    while True:
        try:
            pregunta = input("Usuario: ")
            if pregunta.lower().strip() in ["salir", "exit", "quit"]:
                print("👋 Terminando la conversación. ¡Hasta luego!")
                break
            if not pregunta.strip():
                continue

            ask_question(pregunta)
        except KeyboardInterrupt:
            print("\n👋 Sesión interrumpida. ¡Hasta luego!")
            break

if __name__ == "__main__":
    start_chat()