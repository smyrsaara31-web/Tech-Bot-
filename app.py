import os
import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS

# ==========================================
# 1. إعدادات الصفحة
# ==========================================
st.set_page_config(page_title="TechBot", page_icon="🤖", layout="wide")

# ==========================================
# 2. التصميم (CSS عشان نخلي الشكل زي الصورة)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    /* الخلفية الرئيسية */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Poppins', sans-serif;
    }

    /* إخفاء القوائم العلوية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* الشريط الجانبي (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #1e1e2f !important;
        padding-top: 20px;
    }
    
    /* نصوص الشريط الجانبي */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    /* أزرار الشريط الجانبي */
    [data-testid="stSidebar"] button {
        background-color: transparent !important;
        color: #a0a0b0 !important;
        border: none !important;
        text-align: left !important;
        padding: 10px 15px !important;
        font-size: 16px !important;
        transition: 0.3s;
        width: 100%;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #6C5CE7 !important;
        color: white !important;
    }

    /* تصميم الشات (Message Bubbles) */
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* مربع الإدخال */
    .stChatInput textarea {
        border-radius: 20px !important;
        border: 2px solid #6C5CE7 !important;
        font-family: 'Poppins', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. إعداد الذكاء الاصطناعي
# ==========================================
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

@st.cache_resource(show_spinner=False)
def build_vectorstore():
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = []
    for item in data:
        content = f"Question: {item['question_en']}\n{item['question_ar']}"
        metadata = {"answer_en": item["answer_en"], "answer_ar": item["answer_ar"]}
        documents.append(Document(page_content=content, metadata=metadata))
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=documents, embedding=embedding_model)

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)

def get_response(query, vectorstore, llm):
    try:
        safe_query = query.encode('utf-8').decode('utf-8')
    except:
        safe_query = str(query)

    results = vectorstore.similarity_search_with_score(safe_query, k=2)
    context = ""
    if results and results[0][1] < 0.8:
        for doc, score in results:
            context += f"Local Info: {doc.metadata['answer_en']}\n"
    else:
        try:
            with DDGS() as ddgs:
                web_results = list(ddgs.text(f"tech specs {safe_query}", max_results=2))
                for r in web_results:
                    context += f"Web Info: {r['body']}\n"
        except Exception:
            context += "No external info found.\n"

    prompt = f"""You are TechBot, a professional tech expert.
Answer in the SAME language the user used (Arabic or English).
Be helpful, concise, and use bullet points if listing specs.

Context:
{context}

User: {safe_query}
TechBot:"""
    
    try:
        response = llm.invoke(prompt)
        text = response.content
        if isinstance(text, list):
            text = " ".join([t.get("text", "") if isinstance(t, dict) else str(t) for t in text])
        return text
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# 4. واجهة الشات (Streamlit)
# ==========================================
st.title("🤖 TechBot")
st.caption("Your Smart Tech Assistant")

# --- الشريط الجانبي ---
with st.sidebar:
    st.markdown("### 🤖 TechBot")
    st.markdown("---")
    
    # الأزرار التفاعلية (شبه اللي في الصورة)
    if st.button("🏠 Home"): st.session_state.current_cat = "Home"
    if st.button("💻 Laptops"): st.session_state.current_cat = "Laptops"
    if st.button("📱 Phones"): st.session_state.current_cat = "Phones"
    if st.button("📷 Cameras"): st.session_state.current_cat = "Cameras"
    if st.button("🛠️ Hardware"): st.session_state.current_cat = "Hardware"
    if st.button("💻 Software"): st.session_state.current_cat = "Software"
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- إدارة الجلسة ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm TechBot 👋\n\nI can help you choose the best laptop, phone, or accessory. What are you looking for?"}
    ]
if "current_cat" not in st.session_state:
    st.session_state.current_cat = "Home"

# عرض القسم الحالي
st.markdown(f"#### Current Category: **{st.session_state.current_cat}**")

# --- عرض الرسائل ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- إدخال سؤال جديد ---
if prompt := st.chat_input("Type your message here..."):
    # إضافة سؤال المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # توليد الرد
    with st.chat_message("assistant"):
        with st.spinner("TechBot is thinking..."):
            vs = build_vectorstore()
            llm = get_llm()
            answer = get_response(prompt, vs, llm)
            st.markdown(answer)
    
    # حفظ الرد
    st.session_state.messages.append({"role": "assistant", "content": answer})