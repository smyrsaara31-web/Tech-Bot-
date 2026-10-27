import os
import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS

# 1. إعدادات الصفحة
st.set_page_config(page_title="TechBot", page_icon="🤖", layout="centered")

# 2. تصميم CSS بسيط وأنيق (زي ChatGPT)
st.markdown("""
<style>
    /* الخلفية العامة */
    .stApp {
        background-color: #0f172a;
    }
    /* الشريط الجانبي */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
        color: #e2e8f0 !important;
    }
    /* رسائل الشات */
    [data-testid="stChatMessage"] {
        background-color: #1e293b !important;
        border-radius: 15px;
        border: 1px solid #334155;
    }
    /* النصوص */
    p, h1, h2, h3, li, span {
        color: #f8fafc !important;
    }
    /* مربع الكتابة */
    .stChatInput textarea {
        background-color: #1e293b !important;
        color: white !important;
        border: 1px solid #6C5CE7 !important;
        border-radius: 15px !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. المفتاح
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# 4. تحميل الداتا
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

# 5. دالة الرد
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

# 6. الشريط الجانبي
with st.sidebar:
    st.title("🤖 TechBot")
    st.markdown("### Categories")
    st.markdown("- 💻 Laptops")
    st.markdown("- 📱 Phones")
    st.markdown("- 📷 Cameras")
    st.markdown("- 🛠️ Hardware")
    st.markdown("- 💻 Software")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm TechBot 👋 How can I help you today?"}
        ]
        st.rerun()

# 7. واجهة الشات
st.title("🤖 TechBot")
st.caption("Your Smart Tech Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm TechBot 👋\n\nI can help you with laptops, phones, software, and hardware. What do you need?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about any tech..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            vs = build_vectorstore()
            llm = get_llm()
            answer = get_response(prompt, vs, llm)
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})