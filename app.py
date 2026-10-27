import os
import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS

# ==========================================
# 1. إعدادات الصفحة والواجهة
# ==========================================
st.set_page_config(page_title="TechBot | Smart Assistant", page_icon="🤖", layout="wide")

# تصميم CSS يحاكي الواجهة اللي عملناها
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    
    .stApp {
        background: #f8f9fa;
        font-family: 'Poppins', sans-serif;
    }
    
    /* إخفاء عناصر Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* تنسيق الشريط الجانبي */
    [data-testid="stSidebar"] {
        background-color: #1e1e2f !important;
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* تنسيق رسائل الشات */
    [data-testid="stChatMessage"] {
        background-color: white !important;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    
    /* مربع الكتابة */
    .stChatInput textarea {
        border-radius: 20px !important;
        border: 2px solid #6C5CE7 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. المفتاح والداتا
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

# ==========================================
# 3. منطق الرد
# ==========================================
def get_response(query, vectorstore, llm):
    results = vectorstore.similarity_search_with_score(query, k=2)
    context = ""
    if results and results[0][1] < 0.8:
        for doc, score in results:
            context += f"Local Info: {doc.metadata['answer_en']}\n"
    else:
        try:
            with DDGS() as ddgs:
                web_results = list(ddgs.text(f"tech specs {query}", max_results=2))
                for r in web_results:
                    context += f"Web Info: {r['body']}\n"
        except Exception:
            context += "No external info found.\n"

    prompt = f"""You are TechBot, a professional tech expert.
Answer in the SAME language the user used (Arabic or English).
Be helpful, concise, and use bullet points if listing specs.

Context:
{context}

User: {query}
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
# 4. واجهة الشات (Streamlit Native)
# ==========================================
st.title("🤖 TechBot")
st.markdown("### Your Smart Tech Assistant")
st.markdown("---")

# إدارة الجلسات
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm TechBot 👋\n\nI'm here to help you find the best laptop, phone, or tech device. What are you looking for today?"}
    ]

# الشريط الجانبي
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)
    st.markdown("## 🤖 TechBot")
    st.markdown("### Categories")
    st.markdown("- 💻 Laptops\n- 📱 Phones\n- 📷 Cameras\n- 🛠️ Hardware\n- 💻 Software")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm TechBot 👋\n\nHow can I help you today?"}
        ]
        st.rerun()

# عرض الرسائل
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# إدخال سؤال جديد
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
    
    st.session_state.messages.append({"role": "assistant", "content": answer})