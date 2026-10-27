import os
import json
import streamlit as st
import streamlit.components.v1 as components

# ==========================================
# 1. إعدادات الصفحة
# ==========================================
st.set_page_config(page_title="TechBot | Smart Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

# إخفاء عناصر Streamlit الافتراضية عشان الواجهة تبان مخصصة
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {padding-top: 0rem; padding-bottom: 0rem; padding-left: 0rem; padding-right: 0rem;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# 2. بناء الواجهة (HTML + CSS + JS)
# ==========================================
# هنا بنكتب الواجهة كاملة، والـ JS هو اللي بيدير التنقل بين الشاشات
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
        --secondary: #a29bfe;
        --bg-color: #f8f9fa;
        --text-color: #2d3436;
        --chat-user: #6C5CE7;
        --chat-bot: #ffffff;
    }
    body {
        margin: 0;
        padding: 0;
        font-family: 'Poppins', sans-serif;
        background-color: var(--bg-color);
        color: var(--text-color);
        overflow-x: hidden;
    }
    .app-container {
        display: flex;
        height: 100vh;
        width: 100vw;
    }
    
    /* Sidebar */
    .sidebar {
        width: 260px;
        background: #1e1e2f;
        color: white;
        display: flex;
        flex-direction: column;
        padding: 20px;
    }
    .sidebar-logo {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .sidebar-menu a {
        display: flex;
        align-items: center;
        color: #a0a0b0;
        text-decoration: none;
        padding: 12px 15px;
        border-radius: 10px;
        margin-bottom: 5px;
        transition: 0.3s;
        cursor: pointer;
    }
    .sidebar-menu a:hover, .sidebar-menu a.active {
        background: var(--primary);
        color: white;
    }
    .sidebar-menu i { margin-right: 15px; width: 20px; text-align: center; }
    
    /* Main Content */
    .main-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        background: #ffffff;
        position: relative;
    }
    
    /* Screens */
    .screen { display: none; height: 100%; flex-direction: column; }
    .screen.active { display: flex; }

    /* Welcome Screen */
    .welcome-screen {
        justify-content: center;
        align-items: center;
        text-align: center;
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        padding: 40px;
    }
    .welcome-content h1 {
        font-size: 3rem;
        margin-bottom: 10px;
        background: -webkit-linear-gradient(45deg, var(--primary), #ff7675);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .welcome-content p { font-size: 1.2rem; color: #636e72; margin-bottom: 30px; }
    .start-btn {
        background: var(--primary);
        color: white;
        border: none;
        padding: 15px 40px;
        font-size: 1.1rem;
        border-radius: 30px;
        cursor: pointer;
        transition: 0.3s;
        box-shadow: 0 10px 20px rgba(108, 92, 231, 0.3);
    }
    .start-btn:hover { transform: translateY(-3px); box-shadow: 0 15px 25px rgba(108, 92, 231, 0.4); }
    .features-row { display: flex; gap: 30px; margin-top: 50px; }
    .feature-item { display: flex; flex-direction: column; align-items: center; color: #636e72; }
    .feature-item i { font-size: 1.5rem; color: var(--primary); margin-bottom: 10px; }

    /* Chat Screen */
    .chat-screen {
        background: #f1f2f6;
    }
    .chat-header {
        background: white;
        padding: 15px 30px;
        border-bottom: 1px solid #e1e1e1;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .bot-status { display: flex; align-items: center; gap: 10px; }
    .bot-status .status-dot { width: 10px; height: 10px; background: #00b894; border-radius: 50%; }
    .bot-info h3 { margin: 0; font-size: 1.1rem; }
    .bot-info p { margin: 0; font-size: 0.8rem; color: #00b894; }

    .chat-box {
        flex: 1;
        padding: 20px 30px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    .message {
        max-width: 70%;
        padding: 12px 18px;
        border-radius: 18px;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .message.bot {
        background: var(--chat-bot);
        align-self: flex-start;
        border-bottom-left-radius: 5px;
    }
    .message.user {
        background: var(--chat-user);
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 5px;
    }
    .quick-replies {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 10px;
    }
    .quick-reply {
        background: white;
        border: 1px solid var(--primary);
        color: var(--primary);
        padding: 8px 15px;
        border-radius: 20px;
        cursor: pointer;
        font-size: 0.85rem;
        transition: 0.2s;
    }
    .quick-reply:hover { background: var(--primary); color: white; }

    /* Chat Input */
    .chat-input-area {
        background: white;
        padding: 20px 30px;
        border-top: 1px solid #e1e1e1;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .chat-input-area input {
        flex: 1;
        padding: 15px 20px;
        border-radius: 30px;
        border: 1px solid #e1e1e1;
        outline: none;
        font-family: 'Poppins', sans-serif;
    }
    .chat-input-area button {
        background: var(--primary);
        color: white;
        border: none;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        cursor: pointer;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 1.2rem;
    }
</style>
</head>
<body>

<div class="app-container">
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-logo">🤖 TechBot</div>
        <div class="sidebar-menu">
            <a class="active"><i class="fas fa-home"></i> Home</a>
            <a><i class="fas fa-laptop"></i> Laptops</a>
            <a><i class="fas fa-mobile-alt"></i> Phones</a>
            <a><i class="fas fa-camera"></i> Cameras</a>
            <a><i class="fas fa-headphones"></i> Accessories</a>
            <a><i class="fas fa-tools"></i> Hardware</a>
            <a><i class="fas fa-code"></i> Software</a>
        </div>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        
        <!-- Welcome Screen -->
        <div id="welcomeScreen" class="screen active welcome-screen">
            <div class="welcome-content">
                <h1>Your Smart Tech Assistant</h1>
                <p>Tell us what you need, and our AI chatbot will help you find the perfect device.</p>
                <button class="start-btn" onclick="startChat()">Start Chat <i class="fas fa-arrow-right"></i></button>
                
                <div class="features-row">
                    <div class="feature-item"><i class="fas fa-check-circle"></i> Trusted Recommendations</div>
                    <div class="feature-item"><i class="fas fa-bolt"></i> Save Time & Effort</div>
                    <div class="feature-item"><i class="fas fa-heart"></i> Smart Choices Just for You</div>
                </div>
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

            <div class="chat-input-area">
                <input type="text" id="userInput" placeholder="Type your message..." onkeypress="handleEnter(event)">
                <button onclick="sendMessage()"><i class="fas fa-paper-plane"></i></button>
            </div>
        </div>

    </div>
</div>

<script>
    function startChat() {
        document.getElementById('welcomeScreen').classList.remove('active');
        document.getElementById('chatScreen').classList.add('active');
    }

    function handleEnter(e) {
        if (e.key === 'Enter') sendMessage();
    }

    function sendMessage() {
        const input = document.getElementById('userInput');
        const text = input.value.trim();
        if (!text) return;

        const chatBox = document.getElementById('chatBox');

        // إضافة رسالة المستخدم
        const userDiv = document.createElement('div');
        userDiv.className = 'message user';
        userDiv.innerText = text;
        chatBox.appendChild(userDiv);
        
        input.value = '';

        // محاكاة رد البوت (مؤقتاً)
        setTimeout(() => {
            const botDiv = document.createElement('div');
            botDiv.className = 'message bot';
            botDiv.innerHTML = "I'm searching for that right now... (Backend logic will be connected here soon!)";
            chatBox.appendChild(botDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }, 1000);

        chatBox.scrollTop = chatBox.scrollHeight;
    }
</script>

</body>
</html>
"""

# عرض الواجهة داخل Streamlit
components.html(html_code, height=900, scrolling=True)