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
# 1. إعدادات الصفحة (إخفاء عناصر Streamlit)
# ==========================================
st.set_page_config(page_title="TechBot", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding: 0 !important;}
    iframe {width: 100vw; height: 100vh; border: none;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. إعداد الذكاء الاصطناعي (Groq + LangChain)
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
# 3. الواجهة (HTML + CSS)
# ==========================================
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
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Poppins', sans-serif; background-color: #f4f6f9; height: 100vh; overflow: hidden; }
    .app-container { display: flex; height: 100vh; }
    
    /* Sidebar */
    .sidebar { width: 260px; background: #1e1e2f; color: #fff; display: flex; flex-direction: column; padding: 20px; }
    .logo { font-size: 1.5rem; font-weight: 700; margin-bottom: 30px; display: flex; align-items: center; gap: 10px; }
    .menu a { display: flex; align-items: center; color: #a0a0b0; text-decoration: none; padding: 12px 15px; border-radius: 10px; margin-bottom: 5px; cursor: pointer; transition: 0.3s; }
    .menu a:hover, .menu a.active { background: #6C5CE7; color: #fff; }
    .menu i { margin-right: 15px; width: 20px; text-align: center; }
    
    /* Main Content */
    .main-content { flex: 1; display: flex; flex-direction: column; background: #fff; position: relative; }
    .screen { display: none; height: 100%; flex-direction: column; }
    .screen.active { display: flex; }

    /* Welcome Screen */
    .welcome-screen { justify-content: center; align-items: center; text-align: center; background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); padding: 40px; }
    .welcome-content h1 { font-size: 3rem; margin-bottom: 10px; color: #2d3436; }
    .welcome-content h1 span { color: #6C5CE7; }
    .welcome-content p { font-size: 1.2rem; color: #636e72; margin-bottom: 30px; }
    .start-btn { background: #6C5CE7; color: #fff; border: none; padding: 15px 40px; font-size: 1.1rem; border-radius: 30px; cursor: pointer; box-shadow: 0 10px 20px rgba(108, 92, 231, 0.3); transition: 0.3s; }
    .start-btn:hover { transform: translateY(-3px); }

    /* Chat Screen */
    .chat-header { background: #fff; padding: 15px 30px; border-bottom: 1px solid #e1e1e1; display: flex; align-items: center; gap: 15px; }
    .chat-header h3 { font-size: 1.1rem; margin: 0; }
    .chat-header p { font-size: 0.8rem; color: #00b894; margin: 0; }
    .chat-box { flex: 1; padding: 20px 30px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; background: #f9f9f9; }
    .msg { max-width: 70%; padding: 15px 20px; border-radius: 15px; font-size: 0.95rem; line-height: 1.5; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .msg.bot { background: #fff; align-self: flex-start; border-bottom-left-radius: 2px; }
    .msg.user { background: #6C5CE7; color: #fff; align-self: flex-end; border-bottom-right-radius: 2px; }
    
    /* Input Area */
    .input-area { background: #fff; padding: 20px 30px; border-top: 1px solid #e1e1e1; display: flex; gap: 15px; }
    .input-area input { flex: 1; padding: 15px 20px; border: 1px solid #ddd; border-radius: 30px; outline: none; font-family: inherit; }
    .input-area button { background: #6C5CE7; color: #fff; border: none; width: 50px; height: 50px; border-radius: 50%; cursor: pointer; display: flex; justify-content: center; align-items: center; }
    
    /* Typing Indicator */
    .typing { display: none; font-size: 0.9rem; color: #888; font-style: italic; padding: 10px 30px; }
</style>
</head>
<body>

<div class="app-container">
    <div class="sidebar">
        <div class="logo">🤖 TechBot</div>
        <div class="menu">
            <a class="active" onclick="startChat('Home')"><i class="fas fa-home"></i> Home</a>
            <a onclick="startChat('Laptops')"><i class="fas fa-laptop"></i> Laptops</a>
            <a onclick="startChat('Phones')"><i class="fas fa-mobile-alt"></i> Phones</a>
            <a onclick="startChat('Cameras')"><i class="fas fa-camera"></i> Cameras</a>
            <a onclick="startChat('Hardware')"><i class="fas fa-tools"></i> Hardware</a>
            <a onclick="startChat('Software')"><i class="fas fa-code"></i> Software</a>
        </div>
    </div>

    <div class="main-content">
        <!-- Welcome Screen -->
        <div id="welcomeScreen" class="screen active welcome-screen">
            <div class="welcome-content">
                <h1>Your <span>Smart Tech</span><br>Shopping Assistant</h1>
                <p>Tell us what you need, and our AI chatbot will help you find the perfect device.</p>
                <button class="start-btn" onclick="startChat('Home')">Start Chat <i class="fas fa-arrow-right"></i></button>
            </div>
        </div>

        <!-- Chat Screen -->
        <div id="chatScreen" class="screen">
            <div class="chat-header">
                <i class="fas fa-robot" style="font-size:1.5rem; color:#6C5CE7;"></i>
                <div>
                    <h3 id="headerTitle">TechBot</h3>
                    <p>Online</p>
                </div>
            </div>
            
            <div class="chat-box" id="chatBox">
                <div class="msg bot">Hi! I'm TechBot 👋<br><br>What are you looking for today?</div>
            </div>
            <div class="typing" id="typingIndicator">TechBot is typing...</div>

            <div class="input-area">
                <input type="text" id="userInput" placeholder="Type your message..." onkeypress="handleEnter(event)">
                <button onclick="sendMessage()"><i class="fas fa-paper-plane"></i></button>
            </div>
        </div>
    </div>
</div>

<script>
    function startChat(section) {
        document.getElementById('welcomeScreen').classList.remove('active');
        document.getElementById('chatScreen').classList.add('active');
        document.getElementById('headerTitle').innerText = "TechBot - " + section;
    }

    function handleEnter(e) { if (e.key === 'Enter') sendMessage(); }
    
    function sendMessage() {
        const input = document.getElementById('userInput');
        const text = input.value.trim();
        if (!text) return;

        const chatBox = document.getElementById('chatBox');
        const typing = document.getElementById('typingIndicator');

        // عرض رسالة المستخدم
        const userDiv = document.createElement('div');
        userDiv.className = 'msg user';
        userDiv.innerText = text;
        chatBox.appendChild(userDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        
        input.value = '';
        typing.style.display = 'block';

        // إرسال السؤال لـ Streamlit
        window.parent.postMessage({
            isStreamlitMessage: true,
            type: "streamlit:setComponentValue",
            value: text
        }, "*");
    }

    // استقبال الرد من بايثون
    window.receiveResponse = function(responseText) {
        const chatBox = document.getElementById('chatBox');
        const typing = document.getElementById('typingIndicator');
        typing.style.display = 'none';
        
        const botDiv = document.createElement('div');
        botDiv.className = 'msg bot';
        botDiv.innerHTML = responseText.replace(/\\n/g, '<br>');
        chatBox.appendChild(botDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    };
</script>
</body>
</html>
"""

# ==========================================
# 4. الربط (عشان الرد يشتغل)
# ==========================================
# هنا بنعرض الواجهة
component_value = components.html(html_code, height=850, scrolling=False)

if component_value:
    try:
        # بناء الداتا والـ LLM
        vs = build_vectorstore()
        llm = get_llm()
        
        # استقبال الإجابة
        answer = get_response(component_value, vs, llm)
        
        # تنظيف الرد من أي حاجة تكسر الـ JS
        safe_answer = answer.replace("`", "'").replace("$", "\\$")
        
        # إرسال الرد
        components.html(f"<script>window.parent.receiveResponse(`{safe_answer}`);</script>", height=0)
    except Exception as e:
        # لو حصل خطأ، نبعته للواجهة
        components.html(f"<script>window.parent.receiveResponse(`Error: {str(e)}`);</script>", height=0)