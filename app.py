import os
import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import streamlit.components.v1 as components

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="TechBot | AI Expert", page_icon="🤖", layout="wide")

# ==========================================
# 2. Advanced UI Design (CSS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
        font-family: 'Poppins', sans-serif;
    }

    /* Gradient Hero Title */
    .hero-title {
        font-size: 3.5rem !important;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #a78bfa, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 0px;
        margin-bottom: 30px;
    }

    /* Chat Input Box */
    .stChatInput textarea {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 2px solid #7c3aed !important;
        border-radius: 15px !important;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.2);
        transition: all 0.3s ease;
    }
    .stChatInput textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }

    /* User Chat Bubble */
    [data-testid="stChatMessage"]:has(div:contains("user")) {
        background-color: #4c1d95 !important;
        border-radius: 15px 15px 0 15px;
        padding: 15px;
        margin-left: 15%;
        border: 1px solid #7c3aed;
    }

    /* Assistant Chat Bubble */
    [data-testid="stChatMessage"]:has(div:contains("assistant")) {
        background-color: #1e293b !important;
        border-radius: 15px 15px 15px 0;
        padding: 15px;
        margin-right: 15%;
        border: 1px solid #334155;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #7c3aed, #3b82f6);
        color: white !important;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. API Key
# ==========================================
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ==========================================
# 4. Build Vector Database
# ==========================================
@st.cache_resource(show_spinner=False)
def build_vectorstore():
    with st.spinner("⏳ Preparing Tech Database..."):
        with open("data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            content = f"Question: {item['question_en']}\n{item['question_ar']}"
            metadata = {
                "answer_en": item["answer_en"],
                "answer_ar": item["answer_ar"],
                "category": item["category"]
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model)
    return vectorstore

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

# ==========================================
# 5. Bot Logic
# ==========================================
def get_response(query, vectorstore, llm):
    results = vectorstore.similarity_search(query, k=2)
    context = ""
    if results:
        for doc in results:
            context += f"Info: {doc.metadata['answer_en']}\n"
    
    prompt = f"""You are TechBot, a professional multilingual tech expert.
Answer the user's question in the SAME LANGUAGE they used (Arabic or English).
Use the provided context to answer. If the context doesn't have the info, use your general tech knowledge.
Keep the answer structured, helpful, and concise. Use bullet points for specs if needed.

Context:
{context}

User Question: {query}
Answer:"""
    
    try:
        response = llm.invoke(prompt)
        text = response.content
        if isinstance(text, list):
            text = " ".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in text])
        return text
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# 6. UI & Chat Interface (English Only)
# ==========================================
st.markdown('<h1 class="hero-title">🤖 TechBot</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Your Smart Multilingual Tech Expert</p>', unsafe_allow_html=True)

# Session Management
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 💬 Your Chats")
    
    if st.button("➕ New Chat"):
        new_chat_name = f"Chat {len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[new_chat_name] = []
        st.session_state.current_chat = new_chat_name
        st.rerun()
        
    st.markdown("---")
    
    selected_chat = st.radio("Select Chat:", list(st.session_state.chat_sessions.keys()), index=list(st.session_state.chat_sessions.keys()).index(st.session_state.current_chat))
    if selected_chat != st.session_state.current_chat:
        st.session_state.current_chat = selected_chat
        st.rerun()
        
    st.markdown("---")
    
    if st.button("🗑️ Clear Current Chat"):
        st.session_state.chat_sessions[st.session_state.current_chat] = []
        st.rerun()

# --- Display Messages ---
for msg in st.session_state.chat_sessions[st.session_state.current_chat]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- New Input ---
if prompt := st.chat_input("Ask anything about tech..."):
    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("TechBot is thinking..."):
            vectorstore = build_vectorstore()
            llm = get_llm()
            answer = get_response(prompt, vectorstore, llm)
            
            st.markdown(answer)
            
            # Copy Button
            st.button("📋 Copy Reply", key=f"copy_{len(st.session_state.chat_sessions[st.session_state.current_chat])}", help="Copy to clipboard", on_click=lambda: components.html(f"<script>navigator.clipboard.writeText(`{answer}`);</script>", height=0))
    
    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "assistant", "content": answer})