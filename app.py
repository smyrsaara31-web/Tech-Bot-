import os
import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# ==========================================
# 1. إعدادات الصفحة والواجهة المميزة (CSS)
# ==========================================
st.set_page_config(page_title="TechBot", page_icon="🤖", layout="wide")

# كود CSS عشان نخلي الواجهة شكلها مختلف وشيك (زي Dark Mode احترافي)
st.markdown("""
<style>
    /* الخلفية العامة */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    /* تنسيق العناوين */
    h1, h2, h3 {
        color: #a78bfa !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* تنسيق مربع الكتابة */
    .stChatInput textarea {
        background-color: #1e293b !important;
        color: white !important;
        border-radius: 15px !important;
    }
    /* تنسيق الرسائل */
    .stChatMessage {
        background-color: #1e293b !important;
        border-radius: 15px !important;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ==========================================
# 2. تحميل البيانات وبناء قاعدة البحث
# ==========================================
@st.cache_resource(show_spinner=False)
def build_vectorstore():
    with st.spinner("⏳ Preparing Tech Database (First time only)..."):
        with open("data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            # بنجمع العربي والإنجليزي مع بعض عشان البحث يشتغل بالاتنين
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
# 3. المنطق الأساسي للرد
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
# 4. واجهة الشات (تعدد المحادثات)
# ==========================================
st.title("🤖 TechBot")
st.markdown("### Your Multilingual Tech Expert | خبيرك التقني")

# التحكم في الجلسات (Chats)
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"

# Sidebar لإدارة الشاتات
with st.sidebar:
    st.header("💬 Your Chats")
    if st.button("➕ New Chat"):
        new_chat_name = f"Chat {len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[new_chat_name] = []
        st.session_state.current_chat = new_chat_name
        st.rerun()
    
    selected_chat = st.radio("Select Chat:", list(st.session_state.chat_sessions.keys()), index=list(st.session_state.chat_sessions.keys()).index(st.session_state.current_chat))
    if selected_chat != st.session_state.current_chat:
        st.session_state.current_chat = selected_chat
        st.rerun()

# عرض الرسائل السابقة
for msg in st.session_state.chat_sessions[st.session_state.current_chat]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# إدخال سؤال جديد
if prompt := st.chat_input("Ask about laptops, phones, software... | اسأل عن أي جهاز أو برنامج..."):
    # إضافة سؤال المستخدم
    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # توليد الرد
    with st.chat_message("assistant"):
        with st.spinner("Thinking... | جاري التفكير..."):
            vectorstore = build_vectorstore()
            llm = get_llm()
            answer = get_response(prompt, vectorstore, llm)
            st.markdown(answer)
    
    # حفظ الرد
    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "assistant", "content": answer})