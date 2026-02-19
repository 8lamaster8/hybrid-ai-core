"""
Modern Streamlit чат-интерфейс для AI Assistant - чистая Streamlit версия
"""
import streamlit as st
import requests
import json
import time
import pandas as pd
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime
import os
import sys
from pathlib import Path

# Создаем папку для логов
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "streamlit_chat.log"

# Проверяем, есть ли доступ к существующей системе логирования
try:
    # Пытаемся найти app в родительской директории
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.append(str(project_root))
    
    from app.core.logging import logger
    print(f"✅ Используем существующий логгер. Логи в: {LOG_FILE}")
    
except ImportError as e:
    # Если не нашли, создаем простой но эффективный логгер для Streamlit
    import logging
    
    class StreamlitLogger:
        """Простой адаптер логгера для Streamlit"""
        def __init__(self, log_file: Path):
            self.log_file = log_file
            self.logger = logging.getLogger("streamlit_chat_app")
            self.logger.setLevel(logging.INFO)
            
            # Очищаем старые обработчики
            self.logger.handlers.clear()
            
            # Консольный вывод
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - streamlit_chat - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # Файловый вывод в logs/streamlit_chat.log
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # В файл пишем все
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"Логгер инициализирован. Логи в: {log_file}")
        
        def info(self, msg, *args, **kwargs):
            self.logger.info(msg, *args, **kwargs)
            # Для удобства также выводим в консоль с emoji
            print(f"ℹ️ {msg}")
            
        def warning(self, msg, *args, **kwargs):
            self.logger.warning(msg, *args, **kwargs)
            print(f"⚠️ {msg}")
            
        def error(self, msg, *args, **kwargs):
            self.logger.error(msg, *args, **kwargs)
            print(f"❌ {msg}")
            
        def debug(self, msg, *args, **kwargs):
            self.logger.debug(msg, *args, **kwargs)
            print(f"🔍 {msg}")
            
        def exception(self, msg, *args, **kwargs):
            """Логирование исключений с трейсбеком"""
            self.logger.exception(msg, *args, **kwargs)
            print(f"🔥 {msg}")
    
    logger = StreamlitLogger(LOG_FILE)
    print(f"✅ Создан Streamlit-логгер. Логи в: {LOG_FILE}")

# Конфигурация
API_URL = "http://localhost:8000"
UPLOAD_FOLDER = Path("./uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Инициализация состояния
def init_session_state(url_session_id: str = None):
    """Инициализация состояния сессии с возможностью восстановления"""
    
    # Если есть session_id из URL, используем его
    if url_session_id:
        st.session_state.session_id = url_session_id
        logger.info(f"✅ Восстановлена сессия из URL: {url_session_id[:20]}...")
    elif "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"✅ Создана новая сессия: {st.session_state.session_id[:20]}...")
    
    # Остальные значения по умолчанию
    defaults = {
        "chat_history": [],
        "conversation_id": None,
        "processing": False,
        "knowledge_stats": {"total_chunks": 0},
        "feedback_history": [],
        "last_assistant_message": None,
        "feedback_submitted": False,
        "show_suggestions": True,
        "auto_scroll": True,
        "user_question": None,
        "session_to_conversation": {}  # Кэш для сопоставления
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Получаем или создаем беседу для этой сессии
    if st.session_state.conversation_id is None:
        result = get_or_create_conversation()
        if "conversation" in result and "id" in result["conversation"]:
            st.session_state.conversation_id = result["conversation"]["id"]
            logger.info(f"📝 Беседа для сессии: {result['conversation']['id']}")

def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Вызов API"""
    try:
        url = f"{API_URL}{endpoint}"
        
        headers = {"Content-Type": "application/json"}
        
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=10)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error {response.status_code}: {response.text}"}
    
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def send_message(message: str, session_id: str) -> Dict:
    """Отправка сообщения в чат"""
    endpoint = "/api/v1/chat/ask"
    data = {"message": message, "session_id": session_id}
    result = call_api(endpoint, "POST", data)
    
    if "error" in result:
        return {
            "answer": f"❌ Ошибка: {result['error']}",
            "confidence": 0.0,
            "session_id": session_id,
            "suggestions": []
        }
    
    return result

def get_or_create_conversation():
    """Получение существующей беседы или создание новой"""
    endpoint = "/api/v1/chat/conversations/get_or_create"
    data = {
        "session_id": st.session_state.session_id,
        "title": f"Беседа от {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    }
    
    result = call_api(endpoint, "POST", data)
    
    if "conversation" in result and "id" in result["conversation"]:
        conversation_id = result["conversation"]["id"]
        st.session_state.conversation_id = conversation_id
        
        if result.get("created"):
            logger.info(f"Создана новая беседа: {conversation_id}")
        else:
            logger.info(f"Найдена существующая беседа: {conversation_id}")
    
    return result


def create_conversation() -> Dict:
    """Создание новой беседы в БД"""
    endpoint = "/api/v1/chat/conversations"
    data = {
        "session_id": st.session_state.session_id,
        "title": f"Беседа от {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    }
    result = call_api(endpoint, "POST", data)
    
    if "error" not in result and "id" in result:
        st.session_state.conversation_id = result["id"]
        st.session_state.session_to_conversation[st.session_state.session_id] = result["id"]
    
    return result

def add_feedback(message_id: int, rating: int, helpful: bool = None, comment: str = None):
    """Отправка обратной связи"""
    # Если нет conversation_id, создаем новую беседу
    if st.session_state.conversation_id is None:
        result = create_conversation()
        if "error" not in result and "id" in result:
            st.session_state.conversation_id = result["id"]
        else:
            return {"error": "Не удалось создать беседу для обратной связи"}
    
    endpoint = "/api/v1/chat/feedback"
    data = {
        "conversation_id": st.session_state.conversation_id,  # Используем integer ID
        "message_id": message_id,
        "rating": rating,
        "helpful": helpful,
        "comment": comment,
    }
    return call_api(endpoint, "POST", data)

def upload_file(file_bytes: bytes, filename: str) -> Dict:
    """Загрузка файла"""
    try:
        url = f"{API_URL}/api/v1/knowledge/upload"
        
        files = {'file': (filename, file_bytes)}
        data = {'source': 'streamlit_upload'}
        
        response = requests.post(url, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload error: {response.text}"}
    
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}

def clear_knowledge_base() -> Dict:
    """Очистка базы знаний"""
    endpoint = "/api/v1/knowledge/clear"
    return call_api(endpoint, "DELETE")

def get_knowledge_info() -> Dict:
    """Получение информации о базе знаний"""
    endpoint = "/api/v1/knowledge/info"
    return call_api(endpoint, "GET")

def get_system_stats() -> Dict:
    """Получение статистики системы"""
    endpoint = "/api/v1/system/stats"
    return call_api(endpoint, "GET")


def get_or_create_conversation() -> Dict:
    """Получение существующей беседы или создание новой"""
    # Если уже есть кэшированная беседа для этой сессии
    if st.session_state.session_id in st.session_state.session_to_conversation:
        conversation_id = st.session_state.session_to_conversation[st.session_state.session_id]
        logger.info(f"📂 Используем кэшированную беседу: {conversation_id}")
        return {"conversation": {"id": conversation_id}}
    
    # Пытаемся найти существующую беседу
    try:
        endpoint = f"/api/v1/chat/conversations/by_session/{st.session_state.session_id}"
        result = call_api(endpoint, "GET")
        
        if "error" not in result and "id" in result:
            conversation_id = result["id"]
            st.session_state.session_to_conversation[st.session_state.session_id] = conversation_id
            logger.info(f"🔍 Найдена существующая беседа: {conversation_id}")
            return result
    except Exception as e:
        logger.warning(f"Ошибка при поиске беседы: {e}")
    
    # Создаем новую беседу
    endpoint = "/api/v1/chat/conversations"
    data = {
        "session_id": st.session_state.session_id,
        "title": f"Беседа от {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    }
    result = call_api(endpoint, "POST", data)
    
    if "error" not in result and "id" in result:
        conversation_id = result["id"]
        st.session_state.session_to_conversation[st.session_state.session_id] = conversation_id
        logger.info(f"✨ Создана новая беседа: {conversation_id}")
    
    return result   


def create_sidebar():
    """Создание боковой панели"""
    with st.sidebar:
        # Заголовок
        st.markdown("## 🤖 AI Assistant")
        
        # Информация о сессии
        st.markdown("### 📊 Сессия")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input("ID сессии", value=st.session_state.session_id[:20] + "...", disabled=True, key="session_id_display")
        # В функции create_sidebar()
        with col2:
            if st.button("🔄", help="Новая сессия", key="new_session_btn"):
                # Генерируем новый session_id
                new_session_id = str(uuid.uuid4())
                
                # Сбрасываем состояние
                st.session_state.session_id = new_session_id
                st.session_state.conversation_id = None
                st.session_state.chat_history = []
                st.session_state.last_assistant_message = None
                st.session_state.feedback_submitted = False
                
                # Обновляем URL
                st.query_params["session_id"] = new_session_id
                
                # Создаем новую беседу
                result = get_or_create_conversation()
                if "conversation" in result and "id" in result["conversation"]:
                    st.session_state.conversation_id = result["conversation"]["id"]
                
                st.success("✅ Новая сессия создана!")
                st.rerun()
        
        st.divider()
        
        # Статистика
        st.markdown("### 📈 Статистика")
        if st.button("🔄 Обновить", key="refresh_stats", use_container_width=True):
            stats = get_system_stats()
            if "error" not in stats:
                st.session_state.system_stats = stats
        
        if "system_stats" in st.session_state:
            stats = st.session_state.system_stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Вопросы", stats.get("questions_processed", 0))
            with col2:
                st.metric("Ошибки", stats.get("errors", 0))
        
        # База знаний
        st.divider()
        st.markdown("### 🧠 База знаний")
        
        if st.button("📊 Получить информацию", key="get_kb_info", use_container_width=True):
            info = get_knowledge_info()
            if "knowledge_base" in info:
                st.session_state.knowledge_stats = info["knowledge_base"]
        
        st.metric("Фрагментов", st.session_state.knowledge_stats.get("total_chunks", 0))
        
        # Загрузка файлов
        st.divider()
        st.markdown("### 📁 Загрузка файлов")
        
        uploaded_files = st.file_uploader(
            "Выберите файлы",
            type=["txt", "pdf", "docx", "json", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            with st.expander(f"📂 {len(uploaded_files)} файлов выбрано", expanded=True):
                for file in uploaded_files:
                    st.info(f"**{file.name}** ({file.size / 1024:.1f} KB)")
                
                if st.button("📤 Загрузить все", use_container_width=True, key="upload_all"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        result = upload_file(uploaded_file.getvalue(), uploaded_file.name)
                        results.append((uploaded_file.name, result))
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    success_count = sum(1 for _, r in results if "error" not in r)
                    if success_count > 0:
                        st.success(f"✅ Загружено {success_count}/{len(uploaded_files)} файлов")
                    else:
                        st.error("❌ Не удалось загрузить файлы")
        
        # Опасная зона
        st.divider()
        with st.expander("⚠️ Опасная зона", expanded=False):
            st.warning("Эти действия нельзя отменить!")
            
            if st.button("🗑️ Очистить базу знаний", use_container_width=True, key="clear_kb"):
                result = clear_knowledge_base()
                if "success" in result:
                    st.error("✅ База знаний очищена!")
                else:
                    st.error("❌ Ошибка очистки")
            
            if st.button("🧹 Очистить историю чата", use_container_width=True, key="clear_chat"):
                st.session_state.chat_history = []
                st.success("✅ История очищена!")

def display_welcome():
    """Отображение приветственного экрана с кнопками"""
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #4F46E5;">🤖 AI Assistant</h1>
        <p style="color: #6B7280; font-size: 1.1rem;">Ваш персональный помощник для работы с информацией</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Кнопки быстрого старта
    st.markdown("### 🚀 Быстрый старт")
    st.write("Начните общение с одной из этих тем:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "👋 **Привет! Расскажи о себе**\n\n*Познакомиться с ассистентом*",
            use_container_width=True,
            key="welcome_btn1"
        ):
            st.session_state.user_question = "Привет! Расскажи о себе"
            st.rerun()
    
    with col2:
        if st.button(
            "💡 **Что ты умеешь?**\n\n*Узнать возможности*",
            use_container_width=True,
            key="welcome_btn2"
        ):
            st.session_state.user_question = "Что ты умеешь?"
            st.rerun()
    
    # Дополнительные кнопки
    st.markdown("### 🎯 Популярные темы")
    
    topics_cols = st.columns(3)
    topics = [
        ("🤖", "Об искусственном интеллекте"),
        ("💻", "О программировании"),
        ("📚", "О машинном обучении"),
        ("🔧", "О настройке системы"),
        ("📊", "Об анализе данных"),
        ("🌐", "О веб-разработке"),
    ]
    
    for idx, (icon, topic) in enumerate(topics):
        with topics_cols[idx % 3]:
            if st.button(
                f"{icon} {topic}",
                use_container_width=True,
                key=f"topic_btn_{idx}"
            ):
                st.session_state.user_question = topic
                st.rerun()

def display_message(message: Dict):
    """Отображение сообщения в чате"""
    role = message.get("role", "unknown")
    content = message.get("content", "")
    confidence = message.get("confidence", 1.0)
    processing_time = message.get("processing_time_ms", 0)
    
    with st.chat_message(role):
        # Основное сообщение
        st.markdown(content)
        
        if role == "assistant":
            # Индикатор уверенности
            col1, col2 = st.columns([4, 1])
            with col1:
                if confidence > 0.7:
                    color = "#10B981"
                elif confidence > 0.4:
                    color = "#F59E0B"
                else:
                    color = "#EF4444"
                
                st.markdown(
                    f"""
                    <div style="margin-top: 8px; padding: 8px 12px; background: #F3F4F6; border-radius: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #6B7280; font-size: 0.9rem;">Уверенность:</span>
                            <span style="color: {color}; font-weight: 600; font-size: 0.9rem;">{confidence:.1%}</span>
                        </div>
                        <div style="margin-top: 4px; height: 4px; background: #E5E7EB; border-radius: 2px; overflow: hidden;">
                            <div style="width: {confidence * 100}%; height: 100%; background: {color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.caption(f"⏱️ {processing_time:.0f}мс")

def display_suggestions(suggestions: List[str]):
    """Отображение предложенных вопросов"""
    if not suggestions:
        return
    
    st.markdown("---")
    
    with st.container():
        st.markdown("##### 💡 Возможно вас заинтересует:")
        
        # Отображаем максимум 3 предложения
        for idx, suggestion in enumerate(suggestions[:3]):
            if st.button(
                suggestion,
                key=f"suggestion_{uuid.uuid4().hex[:8]}",
                use_container_width=True,
                help="Нажмите, чтобы задать этот вопрос"
            ):
                st.session_state.user_question = suggestion
                st.rerun()

def display_feedback_buttons(message_id: int):
    """Отображение кнопок обратной связи"""
    if st.session_state.feedback_submitted:
        st.info("✅ Спасибо за обратную связь!")
        return
    
    st.markdown("---")
    
    with st.container():
        st.markdown("##### 📊 Оцените ответ:")
        
        # Создаем conversation для фидбека, если его нет
        if st.session_state.conversation_id is None:
            result = create_conversation()
            if "error" not in result and "id" in result:
                st.session_state.conversation_id = result["id"]
            else:
                st.error("Не удалось создать беседу для оценки")
                return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(
                "👍 Полезный ответ",
                use_container_width=True,
                key=f"like_{message_id}",
                type="primary"
            ):
                result = add_feedback(message_id, 5, True, "Полезный ответ")
                if "error" not in result:
                    st.session_state.feedback_submitted = True
                    st.success("Спасибо за положительную оценку!")
                    st.rerun()
                else:
                    st.error(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")
        
        with col2:
            if st.button(
                "👎 Не полезный",
                use_container_width=True,
                key=f"dislike_{message_id}"
            ):
                result = add_feedback(message_id, 1, False, "Не полезный ответ")
                if "error" not in result:
                    st.session_state.feedback_submitted = True
                    st.warning("Спасибо за обратную связь!")
                    st.rerun()
                else:
                    st.error(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")
        
        with col3:
            if st.button(
                "😐 Средний ответ",
                use_container_width=True,
                key=f"neutral_{message_id}"
            ):
                result = add_feedback(message_id, 3, None, "Средний ответ")
                if "error" not in result:
                    st.session_state.feedback_submitted = True
                    st.info("Спасибо за оценку!")
                    st.rerun()
                else:
                    st.error(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")

def main():
    """Основная функция приложения"""
    st.set_page_config(
        page_title="🤖 AI Assistant Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ========== 1. ВОССТАНОВЛЕНИЕ ИЛИ СОЗДАНИЕ СЕССИИ ==========
    # Пытаемся восстановить session_id из URL
    url_session_id = st.query_params.get("session_id", [None])[0]
    
    if url_session_id:
        logger.info(f"📌 Восстановлен session_id из URL: {url_session_id}")
    
    # Инициализация состояния с учетом URL
    init_session_state(url_session_id)
    
    # Сохраняем session_id в URL (если его там еще нет)
    if "session_id" in st.session_state and st.session_state.session_id:
        if not url_session_id or url_session_id != st.session_state.session_id:
            st.query_params["session_id"] = st.session_state.session_id
    
    # ========== 2. ПРИМЕНЯЕМ CSS СТИЛИ ==========
    st.markdown("""
    <style>
    /* Улучшаем внешний вид кнопок */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Улучшаем внешний вид сообщений */
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
    }
    
    /* Стили для прогресс-бара уверенности */
    .stProgress > div > div > div {
        border-radius: 4px;
    }
    
    /* Убираем лишние отступы */
    .main > div {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Сайдбар
    create_sidebar()
    
    # Основная область
    main_col = st.columns([1, 2, 1])[1]  # Центральная колонка
    
    with main_col:
        # Приветственный экран, если нет истории
        if not st.session_state.chat_history:
            display_welcome()
        
        # Отображение истории чата
        if st.session_state.chat_history:
            st.markdown("### 💬 История диалога")
            
            for message in st.session_state.chat_history:
                display_message(message)
            
            # Предложенные вопросы (для последнего сообщения ассистента)
            if st.session_state.last_assistant_message:
                suggestions = st.session_state.last_assistant_message.get("suggestions", [])
                if suggestions and st.session_state.show_suggestions:
                    display_suggestions(suggestions)
                
                # Обратная связь
                if not st.session_state.feedback_submitted:
                    display_feedback_buttons(st.session_state.last_assistant_message["id"])
        
        # Поле ввода
        st.markdown("---")
        
        if st.session_state.processing:
            with st.spinner("🤔 AI думает..."):
                time.sleep(0.1)
        
        # Обработка вопроса из кнопок
        if st.session_state.user_question:
            user_input = st.session_state.user_question
            del st.session_state.user_question
            
            # Добавляем сообщение пользователя
            user_message = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.chat_history.append(user_message)
            st.session_state.feedback_submitted = False
            
            # Отправляем запрос
            st.session_state.processing = True
            st.rerun()
        
        # Поле ввода
        user_input = st.chat_input(
            "Задайте вопрос...",
            key="chat_input",
            max_chars=500,
            disabled=st.session_state.processing
        )
        
        if user_input and not st.session_state.processing:
            # Добавляем сообщение пользователя
            user_message = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.chat_history.append(user_message)
            st.session_state.feedback_submitted = False
            
            # Отправляем запрос
            st.session_state.processing = True
            st.rerun()
    
    # Обработка запроса (отдельный блок для обработки)
    if st.session_state.processing and st.session_state.chat_history:
        last_message = st.session_state.chat_history[-1]
        if last_message.get("role") == "user":
            with st.spinner("Обрабатываю ваш запрос..."):
                try:
                    response = send_message(last_message["content"], st.session_state.session_id)
                    processing_time = 0
                    
                    if "answer" in response:
                        assistant_message = {
                            "role": "assistant",
                            "content": response["answer"],
                            "confidence": response.get("confidence", 0.5),
                            "processing_time_ms": processing_time,
                            "id": len(st.session_state.chat_history),
                            "timestamp": datetime.now().isoformat(),
                            "suggestions": response.get("suggestions", [])
                        }
                        
                        st.session_state.chat_history.append(assistant_message)
                        st.session_state.last_assistant_message = assistant_message
                    else:
                        error_message = {
                            "role": "assistant",
                            "content": f"❌ Ошибка: {response.get('error', 'Неизвестная ошибка')}",
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.chat_history.append(error_message)
                        
                except Exception as e:
                    error_message = {
                        "role": "assistant",
                        "content": f"❌ Ошибка соединения: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.chat_history.append(error_message)
                
                finally:
                    st.session_state.processing = False
                    st.rerun()

if __name__ == "__main__":
    main()