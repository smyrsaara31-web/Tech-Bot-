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
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, .stApp { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(135deg, #fdf2f8 0%, #ede9fe 50%, #e0f2fe 100%); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem; max-width: 760px; }

/* الأزرار */
.stButton > button {
    width: 100%; border-radius: 12px; border: 1px solid #f9a8d4;
    background: #fff; color: #be185d; font-weight: 600; padding: 8px 4px;
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #ec4899, #8b5cf6); color: #fff; border: none;
    box-shadow: 0 6px 16px rgba(236,72,153,.35);
}
div[data-testid="stColumn"] { min-width: 30% !important; flex: 1 1 30% !important; }
div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; flex-direction: row !important; gap: .5rem; }
.st-key-chat_now { display: flex; justify-content: center; }
.st-key-chat_now button { width: auto; padding: 14px 44px; font-size: 18px; border-radius: 40px; }

/* صفحة الترحيب */
.logo-bar { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 22px; color: #1e1b3a; }
.logo-bar .ic { background: linear-gradient(135deg,#ec4899,#8b5cf6); border-radius: 12px; padding: 4px 8px; }
.hero-title { font-size: 40px; font-weight: 700; line-height: 1.2; color: #1e1b3a; margin: 24px 0 8px; text-align: center; }
.hero-title span { background: linear-gradient(90deg,#ec4899,#8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { text-align: center; color: #64748b; font-size: 15px; margin-bottom: 8px; }
.mascot-wrap { position: relative; height: 220px; display: flex; justify-content: center; align-items: center; }
.mascot { font-size: 90px; width: 150px; height: 150px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    background: radial-gradient(circle, #fff 30%, #fbcfe8 100%); box-shadow: 0 0 50px rgba(236,72,153,.45); animation: float 3s ease-in-out infinite; }
.fl { position: absolute; font-size: 34px; animation: float 4s ease-in-out infinite; }
.fl.a { top: 10px; left: 12%; } .fl.b { top: 20px; right: 12%; animation-delay: .5s; }
.fl.c { bottom: 10px; left: 18%; animation-delay: 1s; } .fl.d { bottom: 20px; right: 18%; animation-delay: 1.5s; }
.fl.e { top: 90px; left: 2%; animation-delay: .8s; } .fl.f { top: 90px; right: 2%; animation-delay: 1.2s; }
@keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-12px); } }
.features { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 28px; }
@media (min-width: 640px) { .features { grid-template-columns: repeat(4, 1fr); } }
.feat { background: rgba(255,255,255,.75); border-radius: 16px; padding: 14px 10px; text-align: center; box-shadow: 0 4px 14px rgba(139,92,246,.12); }
.feat .i { font-size: 26px; } .feat .t { font-size: 13px; font-weight: 600; color: #1e1b3a; margin-top: 4px; }
.feat .d { font-size: 11px; color: #64748b; }

/* صفحة الشات */
.chat-header { display: flex; align-items: center; gap: 12px; background: #1e1b3a; color: #fff; padding: 12px 16px; border-radius: 16px; margin-bottom: 12px; }
.chat-header .av { background: linear-gradient(135deg,#ec4899,#8b5cf6); width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; }
.chat-header .nm { font-weight: 700; font-size: 18px; }
.chat-header .st { font-size: 12px; color: #4ade80; }
[data-testid="stChatMessage"] { background: #fff; border-radius: 16px; padding: 12px; margin-bottom: 8px; box-shadow: 0 2px 10px rgba(139,92,246,.10); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: linear-gradient(90deg, #ec4899, #8b5cf6); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * { color: #fff !important; }
[data-testid="stChatInput"] { border: 2px solid #ec4899; border-radius: 16px; }
.rtl { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

CATEGORIES = ["Home", "Laptops", "Phones", "Cameras", "Hardware", "Software"]
ICONS = {"Home": "🏠", "Laptops": "💻", "Phones": "📱",
         "Cameras": "📷", "Hardware": "🛠️", "Software": "💾"}
WELCOME = "Hi! I'm TechBot 👋 What are you looking for today?"


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