import os
import json
import streamlit as st
import streamlit.components.v1 as components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS

# ==========================================
# 1. إعدادات الصفحة
# ==========================================
st.set_page_config(page_title="TechBot | Smart Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

# إخفاء عناصر Streamlit الافتراضية
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {padding-top: 0rem; padding-bottom: 0rem; padding-left: 0rem; padding-right: 0rem;}
            iframe {width: 100%; height: 100vh;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# 2. إعداد الذكاء الاصطناعي والداتا
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
    # البحث في الداتا المحلية
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
# 3. الواجهة (HTML + CSS + JS)
# ==========================================
# ملاحظة: الـ JS هنا بيبعت السؤال لـ Streamlit عن طريق الـ Custom Component
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TechBot</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
    :root {
        --primary: #6C5CE7;
        --bg-color: #f8f9fa;
        --text-color: #2d3436;
        --chat-user: #6C5CE7;
        --chat-bot: #ffffff;
    }
    body { margin: 0; padding: 0; font-family: 'Poppins', sans-serif; background-color: var(--bg-color); color: var(--text-color); overflow: hidden; }
    .app-container { display: flex; height: 100vh; width: 100vw; }
    
    /* Sidebar */
    .sidebar { width: 260px; background: #1e1e2f; color: white; display: flex; flex-direction: column; padding: 20px; }
    .sidebar-logo { font-size: 1.5rem; font-weight: 700; margin-bottom: 30px; display: flex; align-items: center; gap: 10px; }
    .sidebar-menu a { display: flex; align-items: center; color: #a0a0b0; text-decoration: none; padding: 12px 15px; border-radius: 10px; margin-bottom: 5px; transition: 0.3s; cursor: pointer; }
    .sidebar-menu a:hover, .sidebar-menu a.active { background: var(--primary); color: white; }
    .sidebar-menu i { margin-right: 15px; width: 20px; text-align: center; }
    
    /* Main Content */
    .main-content { flex: 1; display: flex; flex-direction: column; background: #ffffff; position: relative; }
    
    /* Screens */
    .screen { display: none; height: 100%; flex-direction: column; }
    .screen.active { display: flex; }

    /* Welcome Screen */
    .welcome-screen { justify-content: center; align-items: center; text-align: center; background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); padding: 40px; }
    .welcome-content h1 { font-size: 3rem; margin-bottom: 10px; background: -webkit-linear-gradient(45deg, var(--primary), #ff7675); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .welcome-content p { font-size: 1.2rem; color: #636e72; margin-bottom: 30px; }
    .start-btn { background: var(--primary); color: white; border: none; padding: 15px 40px; font-size: 1.1rem; border-radius: 30px; cursor: pointer; transition: 0.3s; box-shadow: 0 10px 20px rgba(108, 92, 231, 0.3); }
    .start-btn:hover { transform: translateY(-3px); box-shadow: 0 15px 25px rgba(108, 92, 231, 0.4); }
    
    /* Chat Screen */
    .chat-screen { background: #f1f2f6; }
    .chat-header { background: white; padding: 15px 30px; border-bottom: 1px solid #e1e1e1; display: flex; align-items: center; gap: 15px; }
    .bot-status { display: flex; align-items: center; gap: 10px; }
    .bot-status .status-dot { width: 10px; height: 10px; background: #00b894; border-radius: 50%; }
    .bot-info h3 { margin: 0; font-size: 1.1rem; }
    .bot-info p { margin: 0; font-size: 0.8rem; color: #00b894; }

    .chat-box { flex: 1; padding: 20px 30px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
    .message { max-width: 70%; padding: 12px 18px; border-radius: 18px; font-size: 0.95rem; line-height: 1.5; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .message.bot { background: var(--chat-bot); align-self: flex-start; border-bottom-left-radius: 5px; }
    .message.user { background: var(--chat-user); color: white; align-self: flex-end; border-bottom-right-radius: 5px; }

    /* Chat Input */
    .chat-input-area { background: white; padding: 20px 30px; border-top: 1px solid #e1e1e1; display: flex; align-items: center; gap: 15px; }
    .chat-input-area input { flex: 1; padding: 15px 20px; border-radius: 30px; border: 1px solid #e1e1e1; outline: none; font-family: 'Poppins', sans-serif; }
    .chat-input-area button { background: var(--primary); color: white; border: none; width: 50px; height: 50px; border-radius: 50%; cursor: pointer; display: flex; justify-content: center; align-items: center; font-size: 1.2rem; }
    
    /* Loading spinner */
    .loading { display: none; font-size: 0.9rem; color: #636e72; font-style: italic; }
</style>
</head>
<body>

<div class="app-container">
    <div class="sidebar">
        <div class="sidebar-logo">🤖 TechBot</div>
        <div class="sidebar-menu">
            <a class="active" onclick="showChat()"><i class="fas fa-home"></i> Home</a>
            <a onclick="showChat()"><i class="fas fa-laptop"></i> Laptops</a>
            <a onclick="showChat()"><i class="fas fa-mobile-alt"></i> Phones</a>
            <a onclick="showChat()"><i class="fas fa-camera"></i> Cameras</a>
            <a onclick="showChat()"><i class="fas fa-tools"></i> Hardware</a>
            <a onclick="showChat()"><i class="fas fa-code"></i> Software</a>
        </div>
    </div>

    <div class="main-content">
        <!-- Welcome Screen -->
        <div id="welcomeScreen" class="screen active welcome-screen">
            <div class="welcome-content">
                <h1>Your Smart Tech Assistant</h1>
                <p>Tell us what you need, and our AI chatbot will help you find the perfect device.</p>
                <button class="start-btn" onclick="startChat()">Start Chat <i class="fas fa-arrow-right"></i></button>
            </div>
        </div>

        <!-- Chat Screen -->
        <div id="chatScreen" class="screen chat-screen">
            <div class="chat-header">
                <div class="bot-status">
                    <div class="status-dot"></div>
                    <div class="bot-info">
                        <h3>TechBot</h3>
                        <p>Online</p>
                    </div>
                </div>
            </div>
            
            <div class="chat-box" id="chatBox">
                <div class="message bot">
                    Hi! I'm TechBot 👋<br><br>
                    I'm here to help you find the best laptop, phone, or tech device. What are you looking for today?
                </div>
            </div>
            <div class="loading" id="loadingIndicator">TechBot is thinking...</div>

            <div class="chat-input-area">
                <input type="text" id="userInput" placeholder="Type your message..." onkeypress="handleEnter(event)">
                <button onclick="sendMessage()"><i class="fas fa-paper-plane"></i></button>
            </div>
        </div>
    </div>
</div>

<script>
    // دالة عشان نستقبل الرد من بايثون
    function receiveResponse(responseText) {
        const chatBox = document.getElementById('chatBox');
        const loading = document.getElementById('loadingIndicator');
        
        loading.style.display = 'none';
        
        const botDiv = document.createElement('div');
        botDiv.className = 'message bot';
        botDiv.innerHTML = responseText.replace(/\\n/g, '<br>');
        chatBox.appendChild(botDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function startChat() {
        document.getElementById('welcomeScreen').classList.remove('active');
        document.getElementById('chatScreen').classList.add('active');
    }
    
    function showChat() {
        startChat();
    }

    function handleEnter(e) {
        if (e.key === 'Enter') sendMessage();
    }

    function sendMessage() {
        const input = document.getElementById('userInput');
        const text = input.value.trim();
        if (!text) return;

        const chatBox = document.getElementById('chatBox');
        const loading = document.getElementById('loadingIndicator');

        const userDiv = document.createElement('div');
        userDiv.className = 'message user';
        userDiv.innerText = text;
        chatBox.appendChild(userDiv);
        
        input.value = '';
        loading.style.display = 'block';
        chatBox.scrollTop = chatBox.scrollHeight;

        // هنا بنبعت السؤال لـ Streamlit
        window.parent.postMessage({
            isStreamlitMessage: true,
            type: "streamlit:setComponentValue",
            value: text
        }, "*");
    }
</script>
</body>
</html>
"""

# ==========================================
# 4. ربط الواجهة بـ بايثون
# ==========================================
# بنستخدم components.html عشان نعرض الواجهة
# وبنستقبل منها السؤال عن طريق Custom Component
component_value = components.html(html_code, height=800)

# لو المستخدم كتب حاجة، بنعالجها ونبعت الرد
if component_value:
    with st.spinner("Thinking..."):
        vs = build_vectorstore()
        llm = get_llm()
        answer = get_response(component_value, vs, llm)
        # بنبعت الرد للواجهة عن طريق JavaScript
        components.html(f"<script>receiveResponse(`{answer}`);</script>", height=0)