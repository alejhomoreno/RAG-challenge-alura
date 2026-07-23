import os
import streamlit as st
import cohere
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Mercado Central 24h — Asistente Operativo",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INICIALIZACIÓN DE CACHÉ Y RECURSOS ---
@st.cache_resource
def iniciar_sistemas():
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        st.error("⚠️ No se encontró COHERE_API_KEY en el archivo .env.")
        st.stop()

    co_client = cohere.Client(api_key=cohere_api_key)
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    db = Chroma(persist_directory="db/chroma", embedding_function=embeddings)
    base_retriever = db.as_retriever(search_kwargs={"k": 10})
    llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)
    
    return co_client, base_retriever, llm

co_client, base_retriever, llm = iniciar_sistemas()

# --- FUNCIONES RAG CON MULTI-QUERY Y RERANKING ---
def obtener_contexto_avanzado(pregunta_usuario):
    prompt_variaciones = ChatPromptTemplate.from_template(
        "Reescribe la siguiente consulta generando 3 variaciones profesionales con sinónimos en español "
        "relacionados con retail, inventario, atención al cliente o proveedores "
        "para mejorar una búsqueda vectorial. Responde SOLO con las variaciones separadas por salto de línea:\n{question}"
    )
    cadena_variaciones = prompt_variaciones | llm | StrOutputParser()
    variaciones_raw = cadena_variaciones.invoke({"question": pregunta_usuario})
    variaciones = [v.strip("- *") for v in variaciones_raw.split("\n") if v.strip()]
    todas_las_consultas = [pregunta_usuario] + variaciones

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

    textos = [d.page_content for d in docs_candidatos]
    rerank_response = co_client.rerank(
        model="rerank-multilingual-v3.0",
        query=pregunta_usuario,
        documents=textos,
        top_n=4
    )

    docs_finales = [docs_candidatos[hit.index] for hit in rerank_response.results]
    return "\n\n---\n\n".join([d.page_content for d in docs_finales])

# --- BARRA LATERAL (SIDEBAR) CORPORATIVA ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart--v1.png", width=70)
    st.title("Mercado Central 24h")
    st.caption("Plataforma Inteligente de Operaciones y Soporte")
    
    st.divider()
    
    st.markdown("### 📚 Base de Conocimiento")
    st.markdown("""
    - 📦 **Inventario LATAM** (`.xlsx`)
    - 🤝 **Manual de Proveedores**
    - 🛎️ **Atención al Cliente y Devoluciones**
    - ❓ **Preguntas Frecuentes (FAQ)**
    - 📜 **Reglamento Interno Operativo**
    """)
    
    st.divider()
    
    st.markdown("### 💡 Ejemplos de Preguntas")
    st.info("""
    • *"¿Qué requisitos necesita un Distribuidor Autorizado?"*\n
    • *"¿Cuál es la política de devolución para Cliente VIP Central?"*\n
    • *"¿Cómo se gestionan las entregas fuera de horario?"*
    """)
    
    if st.button("🗑️ Limpiar Conversación"):
        st.session_state.mensajes_ui = []
        st.session_state.chat_history_llm = []
        st.rerun()

# --- CUERPO PRINCIPAL ---
st.title("🛒 Asistente Virtual Operativo")
st.markdown(
    "Bienvenido al portal de consulta de **Mercado Central 24h**. "
    "Resuelvo dudas sobre inventario, normativas de proveedores, servicio 24/7 y políticas de cliente."
)

# Inicializar historial
if "mensajes_ui" not in st.session_state:
    st.session_state.mensajes_ui = []
if "chat_history_llm" not in st.session_state:
    st.session_state.chat_history_llm = []

# Renderizar mensajes anteriores
for msg in st.session_state.mensajes_ui:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capturar entrada del usuario
if prompt := st.chat_input("Escribe tu consulta sobre operaciones, clientes o proveedores..."):
    
    # 1. UI usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.mensajes_ui.append({"role": "user", "content": prompt})

    # 2. Respuesta del Asistente
    with st.chat_message("assistant"):
        with st.status("Consultando base de conocimientos...", expanded=True) as status:
            
            # Reformular contexto si hay historial
            pregunta_busqueda = prompt
            if st.session_state.chat_history_llm:
                st.write("🧠 Contextualizando consulta...")
                prompt_reformular = (
                    "Dada la conversación previa, reformula esta pregunta para que sea independiente y clara. "
                    "NO la respondas, solo devuélvela reformulada.\n\n"
                    f"Pregunta nueva: {prompt}"
                )
                mensajes_ref = [SystemMessage(content="Eres un asistente contextual.")] + st.session_state.chat_history_llm + [HumanMessage(content=prompt_reformular)]
                pregunta_busqueda = llm.invoke(mensajes_ref).content.strip()

            # Búsqueda RAG
            st.write("🔎 Realizando búsqueda vectorial y Reranking...")
            contexto = obtener_contexto_avanzado(pregunta_busqueda)

            # Prompt especializado en Mercado Central 24h
            st.write("🤖 Sintetizando respuesta...")
            prompt_sistema = (
                "Eres el asistente virtual corporativo de Mercado Central 24h, un supermercado moderno de operación continua (24/7).\n"
                "Tu objetivo es responder a colaboradores, clientes o proveedores basándote ÚNICA Y EXCLUSIVAMENTE en el contexto oficial proporcionado.\n"
                "Si la respuesta no se encuentra en la documentación disponible (manuales, FAQ, inventario o políticas), "
                "indica de manera amable y profesional que no dispones de esa información en los registros actuales.\n"
                "Mantén un tono servicial, preciso y enfocado en la eficiencia operativa y en el programa Cliente VIP Central cuando aplique.\n\n"
                f"--- CONTEXTO OFICIAL DISPONIBLE ---\n{contexto}"
            )
            
            mensajes_respuesta = [SystemMessage(content=prompt_sistema)] + st.session_state.chat_history_llm + [HumanMessage(content=prompt)]
            respuesta = llm.invoke(mensajes_respuesta).content.strip()
            
            status.update(label="¡Consulta procesada!", state="complete", expanded=False)

        # Mostrar respuesta
        st.markdown(respuesta)
        
        # Guardar en memoria
        st.session_state.mensajes_ui.append({"role": "assistant", "content": respuesta})
        st.session_state.chat_history_llm.append(HumanMessage(content=prompt))
        st.session_state.chat_history_llm.append(AIMessage(content=respuesta))