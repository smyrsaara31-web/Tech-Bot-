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

.stButton > button {
    width: 100%; border-radius: 12px; border: 1px solid #f9a8d4;
    background: #fff; color: #be185d; font-weight: 600; padding: 8px 4px;
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #ec4899, #8b5cf6); color: #fff; border: none;
    box-shadow: 0 6px 16px rgba(236,72,153,.35);
}
div[data-testid="stColumn"] { min-width: 30% !important; flex: 1 1 30% !important; }
div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; flex-direction: row !important; gap: .5rem; }
.st-key-chat_now { display: flex; justify-content: center; }
.st-key-chat_now button { width: auto; padding: 14px 44px; font-size: 18px; border-radius: 40px; }

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

.chat-header { display: flex; align-items: center; gap: 12px; background: #1e1b3a; color: #fff; padding: 12px 16px; border-radius: 16px; margin-bottom: 12px; }
.chat-header .av { background: linear-gradient(135deg,#ec4899,#8b5cf6); width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; }
.chat-header .nm { font-weight: 700; font-size: 18px; }
.chat-header .st { font-size: 12px; color: #4ade80; }
[data-testid="stChatMessage"] { background: #fff; border-radius: 16px; padding: 12px; margin-bottom: 8px; box-shadow: 0 2px 10px rgba(139,92,246,.10); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: linear-gradient(90deg, #ec4899, #8b5cf6); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * { color: #fff !important; }
[data-testid="stForm"] { border: 2px solid #ec4899; border-radius: 16px; background: rgba(255,255,255,.7); }
.rtl { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

CATEGORIES = ["Home", "Laptops", "Phones", "Cameras", "Hardware", "Software"]
ICONS = {"Home": "🏠", "Laptops": "💻", "Phones": "📱",
         "Cameras": "📷", "Hardware": "🛠️", "Software": "💾"}


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

    prompt = f"""You are TechBot, a professional tech expert specialized ONLY in the category "{category}" (if the category is "Home", answer about any tech topic).
If the user asks about a different category, do not answer it. Politely tell them in their own language to switch to the right section from the buttons above, and mention which section fits their question.
Answer in the SAME language the user used (Arabic or English).

Rules:
- If the user mentions a budget or price, ONLY recommend options within that budget (5-10% margin is OK). Never exceed it. If nothing fits, say so honestly and suggest the closest option.
- Do NOT use tables. Use short paragraphs and simple bullet points only.
- Keep model names and specs in English, but write the explanation in the user's language.
- Be concise: 2-3 options max, each with key specs, approximate price, and who it's best for.

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


if "page" not in st.session_state:
    st.session_state.page = "home"
if "category" not in st.session_state:
    st.session_state.category = "Home"
if "answer" not in st.session_state:
    st.session_state.answer = None
if "last_q" not in st.session_state:
    st.session_state.last_q = None


# ================= صفحة الترحيب =================
if st.session_state.page == "home":
    st.markdown(
        '<div class="logo-bar"><span class="ic">🤖</span>TechBot</div>'
        '<h1 class="hero-title">Your <span>Smart Tech</span> Shopping Assistant</h1>'
        '<p class="hero-sub">Tell us what you need, and our AI chatbot will find the perfect device for you.</p>'
        '<div class="mascot-wrap"><div class="mascot">🤖</div>'
        '<span class="fl a">💻</span><span class="fl b">📱</span><span class="fl c">🎧</span>'
        '<span class="fl d">🎮</span><span class="fl e">📷</span><span class="fl f">⌚</span></div>',
        unsafe_allow_html=True,
    )
    if st.button("💬  Chat Now", key="chat_now", type="primary"):
        st.session_state.page = "chat"
        st.rerun()
    st.markdown(
        '<div class="features">'
        '<div class="feat"><div class="i">🎯</div><div class="t">Personalized</div><div class="d">Picks made for you</div></div>'
        '<div class="feat"><div class="i">💰</div><div class="t">Budget Friendly</div><div class="d">Respects your price</div></div>'
        '<div class="feat"><div class="i">⚡</div><div class="t">Instant Answers</div><div class="d">Fast and smart</div></div>'
        '<div class="feat"><div class="i">🌍</div><div class="t">Arabic & English</div><div class="d">Ask in any language</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ================= صفحة الشات =================
else:
    st.markdown(
        '<div class="chat-header"><div class="av">🤖</div>'
        '<div><div class="nm">TechBot</div><div class="st">● Online</div></div></div>',
        unsafe_allow_html=True,
    )

    for row in (CATEGORIES[:3], CATEGORIES[3:]):
        cols = st.columns(3)
        for col, cat in zip(cols, row):
            if col.button(f"{ICONS[cat]} {cat}", key=f"btn_{cat}",
                          type="primary" if st.session_state.category == cat else "secondary"):
                st.session_state.category = cat
                st.session_state.answer = None
                st.session_state.last_q = None
                st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("⬅️ Back", key="back"):
        st.session_state.page = "home"
        st.session_state.answer = None
        st.session_state.last_q = None
        st.rerun()
    if c2.button("🗑️ Clear chat", key="clear"):
        st.session_state.answer = None
        st.session_state.last_q = None
        st.rerun()

    st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#1e1b3a'>اسأل TechBot عن أي حاجة 👇</h3>",
                unsafe_allow_html=True)

    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input("سؤالك", placeholder="Type your message...",
                          label_visibility="collapsed")
        sent = st.form_submit_button("🚀 Send")

    if sent and q.strip():
        st.session_state.last_q = q.strip()
        with st.spinner("TechBot is typing..."):
            try:
                st.session_state.answer = get_response(
                    q.strip(), build_vectorstore(), get_llm(), st.session_state.category)
            except Exception as e:
                st.session_state.answer = f"حصل خطأ: {e}"

    if st.session_state.answer:
        with st.chat_message("user"):
            render(st.session_state.last_q)
        with st.chat_message("assistant", avatar="🤖"):
            render(st.session_state.answer)