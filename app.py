import os
import json
import re
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS

st.set_page_config(page_title="TechBot", page_icon="🤖", layout="centered",
                   initial_sidebar_state="collapsed")
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.markdown("""
<style>
    .stApp { background: #f9fafb; }
    [data-testid="stChatMessage"] { border-radius: 14px; padding: 12px; margin-bottom: 8px; }
    .stButton>button { width: 100%; border-radius: 10px; }
    .rtl { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

CATEGORIES = ["Home", "Laptops", "Phones", "Cameras", "Hardware", "Software"]
ICONS = {"Home": "🏠", "Laptops": "💻", "Phones": "📱",
         "Cameras": "📷", "Hardware": "🛠️", "Software": "</>"}


@st.cache_resource(show_spinner=False)
def build_vectorstore():
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = [
        Document(
            page_content=f"Question: {i['question_en']}\n{i['question_ar']}",
            metadata={"answer_en": i["answer_en"], "answer_ar": i["answer_ar"],
                      "category": i.get("category", "")},
        )
        for i in data
    ]
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=docs, embedding=emb)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)


def is_arabic(text):
    return bool(re.search(r"[\u0600-\u06FF]", text))


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

Rules:
- If the user mentions a budget or price, ONLY recommend options within that budget (a small margin of 5-10% is OK). Never exceed it. If nothing fits, say so honestly and suggest the closest option.
- Do NOT use tables. Use short paragraphs and simple bullet points only.
- Keep model names and specs in English, but write the explanation in the user's language.
- Be concise: suggest 2-3 options max, each with key specs, approximate price, and who it's best for.

Context:
{context}

User: {query}
TechBot:"""
    text = llm.invoke(prompt).content
    if isinstance(text, list):
        text = " ".join(t.get("text", "") if isinstance(t, dict) else str(t) for t in text)
    return text


def render(content):
    if is_arabic(content):
        st.markdown(f"<div class='rtl'>\n\n{content}\n\n</div>", unsafe_allow_html=True)
    else:
        st.markdown(content)


# ---------- الأقسام (أزرار أفقية) ----------
if "category" not in st.session_state:
    st.session_state.category = "Home"

st.title("🤖 TechBot")
for row in (CATEGORIES[:3], CATEGORIES[3:]):
    cols = st.columns(3)
    for col, cat in zip(cols, row):
        label = f"{ICONS[cat]} {cat}"
        if col.button(label, key=f"btn_{cat}",
                      type="primary" if st.session_state.category == cat else "secondary"):
            st.session_state.category = cat
            st.rerun()

category = st.session_state.category

# ---------- الشات ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm TechBot 👋 What are you looking for today?"}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render(m["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        render(prompt)
    with st.chat_message("assistant"):
        with st.spinner("TechBot is typing..."):
            try:
                answer = get_response(prompt, build_vectorstore(), get_llm(), category)
            except Exception as e:
                answer = f"حصل خطأ: {e}"
        render(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})