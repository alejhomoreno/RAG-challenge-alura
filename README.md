# 🛒 Mercado Central 24h - Asistente Inteligente Operativo & RAG

> **Supermercado moderno de operación continua (24/7)** que integra la experiencia de tienda física con servicios de delivery y app propia. 

Este repositorio contiene la aplicación web inteligente basada en **Streamlit** y motores de Recuperación Aumentada por Generación (**RAG**), diseñada para asistir en la gestión operativa, consultas sobre políticas, normativas internas, inventarios y atención al cliente de **Mercado Central 24h**.

---

## 🚀 Despliegue en Producción
La aplicación se encuentra desplegada y accesible en la nube de Azure:
👉 **[https://app-mercado24h-final-alejhomoreno.azurewebsites.net/](https://app-mercado24h-final-alejhomoreno.azurewebsites.net/)**

---

## 🧠 Arquitectura y Stack Tecnológico

El proyecto combina un framework web interactivo de Python con un pipeline avanzado de Inteligencia Artificial Generativa y recuperación vectorial:

* **Frontend & Interfaz:** [Streamlit](https://streamlit.io/)
* **Orquestación de IA / LLM:** [LangChain](https://www.langchain.com/) (`langchain-community`, `langchain-text-splitters`)
* **Base de Datos Vectorial:** [ChromaDB](https://www.trychroma.com/) (`langchain-chroma`) para la indexación y búsqueda semántica de documentos corporativos.
* **Embeddings y Generación de Lenguaje:** [Cohere](https://cohere.com/) (`cohere`, `langchain-cohere`) para modelos de lenguaje y representación vectorial de alta precisión en español.
* **Procesamiento de Datos y Documentos:** 
  * `pymupdf` (FitZ) para la ingesta y extracción de texto en documentos PDF normativos.
  * `pandas` & `openpyxl` para el análisis, manipulación y consulta estructurada de bases de datos de inventario.
* **Configuración y Entorno:** `python-dotenv` para la gestión segura de credenciales y variables de entorno.

---

## 📚 Base Documental y Fuentes de Conocimiento (RAG)

El asistente inteligente indexa y procesa la siguiente documentación corporativa para dar respuestas precisas y contextuales:

1. **`inventario_de_supermercado_latam.xlsx`**: Control centralizado de stock, precios, categorías y disponibilidad de productos.
2. **`Política de Atención al Cliente y Devoluciones — Mercado Central 24h (México).pdf`**: Lineamientos para garantías, reembolsos, resolución de conflictos y satisfacción del programa de fidelidad *"Cliente VIP Central"*.
3. **`Preguntas Frecuentes (FAQ) — Mercado Central 24h (México).pdf`**: Respuestas estandarizadas a dudas recurrentes de usuarios y clientes de la app y tienda física.
4. **`Reglamento Interno y Procedimientos Operativos — Mercado Central 24h (México).pdf`**: Protocolos de seguridad, turnos 24/7, manejo de incidentes y normativas para el personal operativo.
5. **`Manual de Proveedores y Política de Compras — Mercado Central 24h (México).pdf`**: Estándares logísticos, tiempos de entrega, auditorías y acuerdos comerciales con la cadena de suministro.

---

## 🛠️ Instalación y Configuración Local

Si deseas clonar y ejecutar este proyecto en tu entorno de desarrollo local, sigue estos pasos:

### 1. Clonar el repositorio
```bash
git clone [https://github.com/alejhomoreno/app-mercado24h-final.git](https://github.com/alejhomoreno/app-mercado24h-final.git)
cd app-mercado24h-final