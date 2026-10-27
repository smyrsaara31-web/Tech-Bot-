import os
import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS

st.set_page_config(page_title="TechBot", page_icon="🤖", layout="wide")
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.markdown("""
<style>
    .stApp { background: #f9fafb; }
    [data-testid="stSidebar"] { background: #1e1e2f; }
    [data-testid="stSidebar"] * { color: #fff !important; }
    [data-testid="stSidebar"] .stRadio label { padding: 8px 12px; border-radius: 8px; }
    [data-testid="stChatMessage"] { border-radius: 14px; padding: 12px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def build_vectorstore():
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = [
        Document(
            page_content=f"Question: {i['question_en']}\n{i['question_ar']}",
            metadata={"answer_en": i["answer_en"], "answer_ar": i["answer_ar"], "category": i.get("category", "")},
        )
        for i in data
    ]
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=docs, embedding=emb)

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)

def get_response(query, vs, llm, category):
    results = vs.similarity_search_with_score(query, k=2)
    context = ""
    if results and results[0][1] < 0.8:
        for doc, _ in results:
            context += f"Local Info: {doc.metadata['answer_en']}\n"
    else:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(f"tech specs {query}", max_results=2):
                    context += f"Web Info: {r['body']}\n"
        except Exception:
            context += "No external info found.\n"

    prompt = f"""You are TechBot, a professional tech expert. Topic: {category}.
Answer in the SAME language the user used (Arabic or English).

Context:
{context}

User: {query}
TechBot:"""
    text = llm.invoke(prompt).content
    if isinstance(text, list):
        text = " ".join(t.get("text", "") if isinstance(t, dict) else str(t) for t in text)
    return text

# الشريط الجانبي
with st.sidebar:
    st.title("🤖 TechBot")
    category = st.radio("القسم", ["Home", "Laptops", "Phones", "Cameras", "Hardware", "Software"], label_visibility="collapsed")

# الشات
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm TechBot 👋 What are you looking for today?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("TechBot is typing..."):
            try:
                answer = get_response(prompt, build_vectorstore(), get_llm(), category)
            except Exception as e:
                answer = f"حصل خطأ: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})