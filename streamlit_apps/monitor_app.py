"""
Streamlit дашборд для мониторинга AI Assistant
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import psutil
import json
import humanize
import numpy as np
from typing import List, Dict, Optional, Any
import plotly.express as px

# Конфигурация
API_URL = "http://localhost:8000"
PROMETHEUS_URL = "http://localhost:9090"

# Инициализация состояния
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

def get_system_info():
    """Получение информации о системе"""
    try:
        response = requests.get(f"{API_URL}/api/v1/system/info", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def get_health():
    """Получение статуса здоровья"""
    try:
        response = requests.get(f"{API_URL}/api/v1/system/health", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def get_resources():
    """Получение информации о ресурсах"""
    try:
        response = requests.get(f"{API_URL}/api/v1/system/resources", timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Если в данных нет used_gb, вычисляем его
            if 'memory' in data and 'used_gb' not in data['memory']:
                total = data['memory'].get('total_gb', 0)
                available = data['memory'].get('available_gb', 0)
                data['memory']['used_gb'] = total - available
            return data
    except Exception as e:
        st.warning(f"Не удалось получить данные ресурсов от API: {e}")
    
    # Если не удалось получить от API, то используем локальную функцию
    return get_system_resources_local()

def get_stats():
    """Получение статистики"""
    try:
        response = requests.get(f"{API_URL}/api/v1/system/stats", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def get_metrics():
    """Получение метрик Prometheus"""
    try:
        response = requests.get(f"{API_URL}/api/v1/system/metrics", timeout=5)
        if response.status_code == 200:
            return parse_prometheus_metrics(response.text)
    except:
        return {}

def get_knowledge_info():
    """Получение информации о базе знаний"""
    try:
        response = requests.get(f"{API_URL}/api/v1/knowledge/info", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def get_ab_testing_stats():
    """Получение статистики A/B тестирования"""
    try:
        # В реальной системе здесь запрос к API A/B тестирования
        # Пока возвращаем тестовые данные
        return {
            "response_template": {
                "variants": [
                    {"id": "A", "template": "{answer}", "participants": 450, "conversion_rate": 0.42, "satisfaction": 4.2},
                    {"id": "B", "template": "📚 {answer}", "participants": 350, "conversion_rate": 0.48, "satisfaction": 4.5},
                    {"id": "C", "template": "🔍 Нашел информацию: {answer}", "participants": 120, "conversion_rate": 0.52, "satisfaction": 4.7},
                    {"id": "D", "template": "💡 Вот что я узнал: {answer}", "participants": 80, "conversion_rate": 0.38, "satisfaction": 3.8}
                ],
                "total_participants": 1000,
                "best_variant": "C",
                "confidence_level": 0.95
            },
            "confidence_display": {
                "variants": [
                    {"id": "A", "show": False, "participants": 600, "satisfaction": 4.1, "engagement": 0.65},
                    {"id": "B", "show": True, "participants": 400, "satisfaction": 4.4, "engagement": 0.72}
                ],
                "total_participants": 1000,
                "best_variant": "B"
            },
            "timestamp": datetime.now().isoformat()
        }
    except:
        return {}

def get_followup_stats():
    """Получение статистики follow-up генератора"""
    try:
        # Тестовые данные
        return {
            "total_generated": 1250,
            "avg_per_session": 2.3,
            "click_through_rate": 0.28,
            "engagement_rate": 0.42,
            "top_topics": [
                {"topic": "python", "count": 320, "engagement": 0.68},
                {"topic": "машинное обучение", "count": 280, "engagement": 0.72},
                {"topic": "базы данных", "count": 195, "engagement": 0.61},
                {"topic": "web разработка", "count": 150, "engagement": 0.55},
                {"topic": "devops", "count": 120, "engagement": 0.49}
            ],
            "effectiveness": {
                "high_engagement": 0.65,
                "medium_engagement": 0.25,
                "low_engagement": 0.10
            },
            "trend": "positive",
            "timestamp": datetime.now().isoformat()
        }
    except:
        return {}

def get_rl_agent_stats():
    """Получение статистики RL агента"""
    try:
        # Тестовые данные
        return {
            "q_table_updates": 1250,
            "exploration_rate": 0.25,
            "learning_rate": 0.1,
            "rewards_received": 850,
            "explorations": 315,
            "exploitations": 935,
            "states": {
                "factual_with_knowledge": {"high": 0.85, "medium": 0.10, "low": 0.05, "cautious": 0.00},
                "factual_no_knowledge": {"high": 0.15, "medium": 0.35, "low": 0.50, "cautious": 0.00},
                "conversational_with_knowledge": {"high": 0.70, "medium": 0.25, "low": 0.05, "cautious": 0.00},
                "conversational_no_knowledge": {"high": 0.20, "medium": 0.45, "low": 0.35, "cautious": 0.00},
                "ambiguous": {"high": 0.30, "medium": 0.40, "low": 0.30, "cautious": 0.00}
            },
            "performance": {
                "avg_reward": 0.68,
                "success_rate": 0.82,
                "improvement_trend": "positive",
                "avg_confidence_adjustment": 0.12
            },
            "timestamp": datetime.now().isoformat()
        }
    except:
        return {}

def get_feedback_stats():
    """Получение статистики обратной связи"""
    try:
        # В реальной системе - запрос к API
        # Пока тестовые данные с динамикой
        current_time = datetime.now()
        hour = current_time.hour
        
        # Динамическое распределение для демонстрации
        base_ratings = {"1": 10, "2": 25, "3": 150, "4": 400, "5": 665}
        # Добавляем немного динамики в зависимости от времени
        time_factor = 1 + 0.1 * np.sin(hour / 24 * 2 * np.pi)
        
        dynamic_ratings = {k: int(v * time_factor) for k, v in base_ratings.items()}
        total_feedback = sum(dynamic_ratings.values())
        
        # Рассчитываем средний рейтинг
        weighted_sum = sum(int(k) * v for k, v in dynamic_ratings.items())
        average_rating = weighted_sum / total_feedback if total_feedback > 0 else 0
        
        return {
            "total_feedback": total_feedback,
            "average_rating": round(average_rating, 2),
            "distribution": dynamic_ratings,
            "helpfulness": {
                "helpful": int(850 * time_factor),
                "not_helpful": int(150 * time_factor),
                "no_feedback": int(250 * time_factor)
            },
            "trend": "positive",
            "satisfaction_rate": 0.85,
            "recent_feedback": [
                {"rating": 5, "helpful": True, "comment": "Отличный ответ!", "timestamp": (current_time - timedelta(minutes=5)).isoformat()},
                {"rating": 4, "helpful": True, "comment": "Полезно, но можно подробнее", "timestamp": (current_time - timedelta(minutes=15)).isoformat()},
                {"rating": 3, "helpful": None, "comment": "Средне", "timestamp": (current_time - timedelta(minutes=30)).isoformat()},
                {"rating": 5, "helpful": True, "comment": "Спасибо, очень помогло!", "timestamp": (current_time - timedelta(minutes=45)).isoformat()},
                {"rating": 2, "helpful": False, "comment": "Не то, что я искал", "timestamp": (current_time - timedelta(minutes=60)).isoformat()}
            ],
            "timestamp": current_time.isoformat()
        }
    except Exception as e:
        st.error(f"Ошибка получения статистики фидбека: {e}")
        return {}

def parse_prometheus_metrics(metrics_text):
    """Парсинг метрик Prometheus из текста"""
    metrics = {}
    for line in metrics_text.split('\n'):
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                try:
                    value = float(parts[1])
                    metrics[name] = value
                except ValueError:
                    pass
    return metrics

def get_system_resources_local():
    """Локальное получение ресурсов системы"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.5)
        disk = psutil.disk_usage('/')
        
        # Информация о сети
        net_io = psutil.net_io_counters()
        
        return {
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": memory.percent,
                "used_gb": round(memory.used / (1024**3), 2)
            },
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                "load_avg": [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()][:3] if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": disk.percent
            },
            "network": {
                "bytes_sent_mb": round(net_io.bytes_sent / (1024**2), 2),
                "bytes_recv_mb": round(net_io.bytes_recv / (1024**2), 2),
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        }
    except Exception as e:
        st.error(f"Ошибка при получении локальных ресурсов: {e}")
        return {
            "memory": {"total_gb": 0, "available_gb": 0, "used_percent": 0, "used_gb": 0},
            "cpu": {"percent": 0, "count": 0, "freq": 0, "load_avg": [0, 0, 0]},
            "disk": {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent": 0},
            "network": {"bytes_sent_mb": 0, "bytes_recv_mb": 0, "packets_sent": 0, "packets_recv": 0}
        }

def create_gauge_chart(value, max_value, title, color):
    """Создание кругового графика-датчика"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, max_value * 0.6], 'color': "#2E7D32"},
                {'range': [max_value * 0.6, max_value * 0.8], 'color': "#F9A825"},
                {'range': [max_value * 0.8, max_value], 'color': "#C62828"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(size=12)
    )
    return fig

def create_bar_chart(labels, values, title, colors=None, horizontal=False):
    """Создание столбчатой диаграммы"""
    if horizontal:
        fig = go.Figure(data=[go.Bar(
            y=labels,
            x=values,
            orientation='h',
            marker_color=colors if colors else 'steelblue',
            text=values,
            textposition='auto'
        )])
    else:
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            marker_color=colors if colors else 'steelblue',
            text=values,
            textposition='auto'
        )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_pie_chart(labels, values, title):
    """Создание круговой диаграммы"""
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='percent+label',
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    return fig

def create_heatmap(data, title, x_labels, y_labels):
    """Создание тепловой карты"""
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        text=[[f'{val:.2f}' for val in row] for row in data],
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Действия",
        yaxis_title="Состояния"
    )
    return fig

def create_trend_chart(timestamps, values, title, y_title):
    """Создание графика тренда"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='lines+markers',
        name='Тренд',
        line=dict(color='#2196F3', width=2),
        marker=dict(size=6)
    ))
    
    # Добавляем скользящее среднее
    if len(values) > 5:
        window_size = min(5, len(values))
        moving_avg = pd.Series(values).rolling(window=window_size).mean().tolist()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=moving_avg,
            mode='lines',
            name=f'Среднее ({window_size} точек)',
            line=dict(color='#FF5722', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Время",
        yaxis_title=y_title,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0.02)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_radar_chart(categories, values, title):
    """Создание радар-чарта"""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Производительность',
        line_color='#4CAF50'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title=dict(text=title, font=dict(size=14)),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_histogram(data, title, x_title, bins=20):
    """Создание гистограммы"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        marker_color='#3F51B5',
        opacity=0.7
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=x_title,
        yaxis_title="Частота",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        bargap=0.1
    )
    return fig

def format_number(num):
    """Форматирование чисел для отображения"""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f} млрд"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f} млн"
    elif num >= 1_000:
        return f"{num / 1_000:.1f} тыс"
    return str(num)

def main():
    """Основная функция дашборда"""
    st.set_page_config(
        page_title="📊 AI Assistant Monitor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Заголовок
    st.title("📊 AI Assistant Monitoring Dashboard")
    st.caption("Real-time monitoring of AI system performance and metrics")
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Автообновление
        auto_refresh = st.checkbox("🔄 Автообновление", value=st.session_state.auto_refresh)
        if auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = auto_refresh
            st.rerun()
        
        refresh_rate = st.slider("Частота обновления (сек)", 5, 60, 10, disabled=not auto_refresh)
        
        # Информация о сервере
        st.divider()
        st.header("🌐 Сервер")
        server_status = st.selectbox(
            "Статус сервера",
            ["🟢 Онлайн", "🟡 Деградирован", "🔴 Оффлайн"],
            index=0
        )
        
        st.metric("API URL", API_URL)
        
        # Ссылки на другие инструменты
        st.divider()
        st.header("🔗 Быстрые ссылки")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Prometheus", use_container_width=True):
                st.markdown(f"[Открыть Prometheus]({PROMETHEUS_URL})", unsafe_allow_html=True)
        with col2:
            if st.button("🤖 Чат", use_container_width=True):
                st.markdown("[Открыть Чат](http://localhost:8501)", unsafe_allow_html=True)
        
        # Информация о версии
        st.divider()
        st.caption("Версия 2.1.0")
        st.caption(f"Обновлено: {datetime.now().strftime('%H:%M:%S')}")
    
    # Получение данных
    with st.spinner("🔄 Загрузка данных..."):
        system_info = get_system_info()
        health_status = get_health()
        resources = get_resources()
        stats = get_stats()
        metrics = get_metrics()
        knowledge_info = get_knowledge_info()
        ab_stats = get_ab_testing_stats()
        followup_stats = get_followup_stats()
        rl_stats = get_rl_agent_stats()
        feedback_stats = get_feedback_stats()
        
        st.session_state.last_refresh = datetime.now()
    
    # Создаем вкладки
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Общее состояние", 
        "🧠 База знаний", 
        "🔬 A/B Тестирование",
        "💭 Follow-up",
        "🤖 RL Агент",
        "⚙️ Система",
        "👍 Обратная связь"  # Новая вкладка
    ])
    
    # Вкладка 1: Общее состояние
    with tab1:
        st.header("🏥 Состояние системы")
        
        # Статус здоровья
        health_cols = st.columns(4)
        with health_cols[0]:
            status = health_status.get("status", "unknown")
            if status == "healthy":
                st.success("✅ Здоров")
            elif status == "degraded":
                st.warning("⚠️ Деградирован")
            else:
                st.error("❌ Не здоров")
            st.metric("Статус", status.capitalize())
        
        with health_cols[1]:
            total_questions = stats.get("questions_processed", 0)
            st.metric("Вопросов обработано", format_number(total_questions))
        
        with health_cols[2]:
            avg_time = stats.get("average_processing_time_ms", 0)
            st.metric("Среднее время ответа", f"{avg_time:.0f} мс")
        
        with health_cols[3]:
            cache_hits = stats.get("cache_hits", 0)
            cache_misses = stats.get("questions_processed", 0) - cache_hits
            total = max(1, cache_hits + cache_misses)
            hit_rate = (cache_hits / total) * 100
            st.metric("Попадания в кэш", f"{hit_rate:.1f}%", 
                     f"{cache_hits}/{cache_misses}")
        
        # Использование ресурсов
        st.header("💾 Использование ресурсов")
        
        if resources:
            resource_cols = st.columns(4)
            
            with resource_cols[0]:
                mem_percent = resources["memory"]["used_percent"]
                st.plotly_chart(create_gauge_chart(
                    mem_percent, 100, "Память", "#2196F3"
                ), use_container_width=True)
                
                used_gb = resources['memory'].get('used_gb', 0)
                total_gb = resources['memory'].get('total_gb', 0)
                st.metric("Память", f"{used_gb:.1f} ГБ", f"/ {total_gb:.1f} ГБ")
            
            with resource_cols[1]:
                cpu_percent = resources["cpu"]["percent"]
                st.plotly_chart(create_gauge_chart(
                    cpu_percent, 100, "CPU", "#4CAF50"
                ), use_container_width=True)
                st.metric("CPU", f"{cpu_percent:.1f}%", f"{resources['cpu']['count']} ядер")
            
            with resource_cols[2]:
                disk_percent = resources["disk"]["percent"]
                st.plotly_chart(create_gauge_chart(
                    disk_percent, 100, "Диск", "#FF9800"
                ), use_container_width=True)
                st.metric("Диск", f"{disk_percent:.1f}%", 
                         f"{resources['disk']['used_gb']:.1f}/{resources['disk']['total_gb']:.1f} ГБ")
            
            with resource_cols[3]:
                # Network usage
                if 'network' in resources:
                    net_up = resources['network']['bytes_sent_mb']
                    net_down = resources['network']['bytes_recv_mb']
                    net_total = net_up + net_down
                    
                    fig = go.Figure(go.Indicator(
                        mode="number+gauge",
                        value=net_total,
                        number={"suffix": " МБ"},
                        title={"text": "Сеть"},
                        gauge={
                            'axis': {'range': [None, max(net_total * 2, 100)]},
                            'bar': {'color': "#9C27B0"},
                            'steps': [
                                {'range': [0, net_total * 0.6], 'color': "lightgray"},
                                {'range': [net_total * 0.6, net_total * 0.8], 'color': "gray"},
                                {'range': [net_total * 0.8, net_total], 'color': "darkgray"}
                            ]
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"↑ {net_up:.1f} МБ | ↓ {net_down:.1f} МБ")
        else:
            st.warning("Не удалось получить данные о ресурсах")
        
        # Статистика работы
        st.header("📈 Статистика работы")
        
        if stats:
            stat_cols = st.columns(2)
            
            with stat_cols[0]:
                labels = ["Обработано", "Кэш попаданий", "Ошибок", "Сессий"]
                values = [
                    stats.get("questions_processed", 0),
                    stats.get("cache_hits", 0),
                    stats.get("errors", 0),
                    stats.get("sessions_created", 0)
                ]
                colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800']
                
                st.plotly_chart(
                    create_bar_chart(labels, values, "Ключевые метрики", colors),
                    use_container_width=True
                )
            
            with stat_cols[1]:
                # Динамика трендов
                hours = list(range(24))
                time_labels = [f"{h:02d}:00" for h in hours]
                
                # Генерация тестовых данных с трендом
                base_questions = 100
                trend_factor = [1 + 0.3 * np.sin(h/24 * 2 * np.pi) + 0.1 * np.random.rand() for h in hours]
                questions_per_hour = [int(base_questions * f) for f in trend_factor]
                
                response_times = [max(50, 200 - h * 2 + 20 * np.sin(h/12 * np.pi)) for h in hours]
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Вопросы в час", "Время ответа (мс)"),
                    vertical_spacing=0.15
                )
                
                fig.add_trace(
                    go.Scatter(x=time_labels, y=questions_per_hour, mode='lines+markers', 
                             name='Вопросы', line=dict(color='#2196F3', width=2)),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=time_labels, y=response_times, mode='lines+markers',
                             name='Время ответа', line=dict(color='#4CAF50', width=2)),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=400, 
                    showlegend=True, 
                    margin=dict(l=20, r=20, t=50, b=20),
                    plot_bgcolor='rgba(0,0,0,0.02)'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Статистика временно недоступна")
    
    # Вкладка 2: База знаний
    with tab2:
        st.header("🧠 База знаний")
        
        if knowledge_info and "knowledge_base" in knowledge_info:
            kb_info = knowledge_info["knowledge_base"]
            
            kb_cols = st.columns([2, 1])
            
            with kb_cols[0]:
                st.subheader("📊 Основные метрики")
                
                metrics_data = {
                    "Метрика": [
                        "Всего фрагментов",
                        "Модель эмбеддингов", 
                        "Размерность эмбеддингов",
                        "Поисков выполнено",
                        "Добавлений",
                        "Удалений",
                        "Обновлений"
                    ],
                    "Значение": [
                        str(kb_info.get("total_chunks", 0)),
                        str(kb_info.get("embedding_model", "Неизвестно")),
                        str(kb_info.get("embedding_dimension", 0)),
                        str(kb_info.get("stats", {}).get("searches", 0)),
                        str(kb_info.get("stats", {}).get("additions", 0)),
                        str(kb_info.get("stats", {}).get("deletions", 0)),
                        str(kb_info.get("stats", {}).get("updates", 0))
                    ]
                }
                
                st.dataframe(
                    pd.DataFrame(metrics_data),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Метрика": st.column_config.Column(width="medium"),
                        "Значение": st.column_config.Column(width="small")
                    }
                )
            
            with kb_cols[1]:
                st.subheader("📂 Распределение по источникам")
                
                # Тестовые данные с динамикой
                sources = ["Файлы", "API", "Веб", "Ручной ввод", "Импорт"]
                counts = [120, 85, 65, 30, 45]
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFD166']
                
                fig = go.Figure(data=[go.Pie(
                    labels=sources,
                    values=counts,
                    hole=0.3,
                    marker=dict(colors=colors),
                    textinfo='label+percent',
                    textposition='inside'
                )])
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Дополнительные метрики качества
                st.subheader("🎯 Качество знаний")
                quality_cols = st.columns(2)
                with quality_cols[0]:
                    st.metric("Релевантность", "78.5%", "↑2.3%")
                with quality_cols[1]:
                    st.metric("Дубликаты", "4.2%", "↓1.1%")
        else:
            st.warning("Информация о базе знаний недоступна")
        
        # Детализация знаний
        st.header("📚 Детализация знаний")
        
        detail_cols = st.columns(3)
        with detail_cols[0]:
            # По типу контента
            content_types = ["Текст", "Код", "Таблицы", "Изображения", "Другое"]
            content_counts = [320, 150, 85, 45, 65]
            st.plotly_chart(
                create_bar_chart(content_types, content_counts, "По типу контента", horizontal=True),
                use_container_width=True
            )
        
        with detail_cols[1]:
            # По языку
            languages = ["Python", "SQL", "JavaScript", "Документация", "Другое"]
            lang_counts = [280, 120, 85, 150, 130]
            st.plotly_chart(
                create_pie_chart(languages, lang_counts, "По языку/типу"),
                use_container_width=True
            )
        
        with detail_cols[2]:
            # Активность обновлений
            days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
            updates = [45, 52, 48, 60, 55, 30, 25]
            st.plotly_chart(
                create_bar_chart(days, updates, "Активность по дням"),
                use_container_width=True
            )
    
    # Вкладка 3: A/B Тестирование
    with tab3:
        st.header("🔬 A/B Тестирование")
        
        if ab_stats:
            # Шаблоны ответов
            st.subheader("📝 Шаблоны ответов")
            
            template_data = ab_stats.get("response_template", {})
            variants = template_data.get("variants", [])
            
            if variants:
                # Основные метрики
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Участники", template_data.get('total_participants', 0))
                with metric_cols[1]:
                    best_variant = template_data.get('best_variant', 'N/A')
                    st.metric("Лучший вариант", best_variant)
                with metric_cols[2]:
                    confidence = template_data.get('confidence_level', 0)
                    st.metric("Уровень доверия", f"{confidence:.0%}")
                with metric_cols[3]:
                    best_rate = max((v['conversion_rate'] for v in variants), default=0)
                    st.metric("Лучшая конверсия", f"{best_rate:.1%}")
                
                # Детализация вариантов
                st.subheader("📊 Детализация вариантов")
                
                tab_cols = st.columns(2)
                with tab_cols[0]:
                    # Таблица с результатами
                    df_variants = pd.DataFrame(variants)
                    display_df = df_variants[['id', 'template', 'participants', 'conversion_rate', 'satisfaction']].copy()
                    display_df.columns = ['Вариант', 'Шаблон', 'Участники', 'Конверсия', 'Удовлетворенность']
                    
                    # Форматирование
                    display_df['Конверсия'] = display_df['Конверсия'].apply(lambda x: f"{x:.1%}")
                    display_df['Удовлетворенность'] = display_df['Удовлетворенность'].apply(lambda x: f"{x:.1f}/5.0")
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                
                with tab_cols[1]:
                    # Визуализация конверсии
                    labels = [f"Вариант {v['id']}" for v in variants]
                    conversions = [v['conversion_rate'] * 100 for v in variants]
                    participants = [v['participants'] for v in variants]
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Конверсия по вариантам (%)", "Распределение участников"),
                        vertical_spacing=0.2
                    )
                    
                    fig.add_trace(
                        go.Bar(x=labels, y=conversions, name='Конверсия',
                             marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(x=labels, y=participants, name='Участники',
                             marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=500, showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            
            # Отображение уверенности
            st.subheader("🎯 Отображение уверенности")
            
            confidence_data = ab_stats.get("confidence_display", {})
            confidence_variants = confidence_data.get("variants", [])
            
            if confidence_variants:
                conf_cols = st.columns(3)
                
                with conf_cols[0]:
                    labels = [f"Вариант {v['id']}" for v in confidence_variants]
                    participants = [v['participants'] for v in confidence_variants]
                    st.plotly_chart(
                        create_pie_chart(labels, participants, "Распределение участников"),
                        use_container_width=True
                    )
                
                with conf_cols[1]:
                    labels = [f"Вариант {v['id']}" for v in confidence_variants]
                    satisfaction = [v.get('satisfaction', 0) for v in confidence_variants]
                    st.plotly_chart(
                        create_bar_chart(labels, satisfaction, "Удовлетворенность"),
                        use_container_width=True
                    )
                
                with conf_cols[2]:
                    labels = [f"Вариант {v['id']}" for v in confidence_variants]
                    engagement = [v.get('engagement', 0) * 100 for v in confidence_variants]
                    st.plotly_chart(
                        create_gauge_chart(engagement[1] if len(engagement) > 1 else 0, 
                                         100, "Вовлеченность варианта B", "#9C27B0"),
                        use_container_width=True
                    )
                
                best_conf_variant = confidence_data.get('best_variant', 'N/A')
                st.info(f"**Рекомендуемый вариант:** {best_conf_variant} (на основе A/B тестов)")
        else:
            st.info("Данные A/B тестирования временно недоступны")
    
    # Вкладка 4: Follow-up вопросы
    with tab4:
        st.header("💭 Follow-up вопросы")
        
        if followup_stats:
            # Основные метрики
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric("Всего сгенерировано", followup_stats.get('total_generated', 0))
            
            with metric_cols[1]:
                avg_per_session = followup_stats.get('avg_per_session', 0)
                st.metric("На сессию", f"{avg_per_session:.1f}")
            
            with metric_cols[2]:
                ctr = followup_stats.get('click_through_rate', 0)
                st.metric("CTR", f"{ctr:.1%}")
            
            with metric_cols[3]:
                engagement = followup_stats.get('engagement_rate', 0)
                st.metric("Вовлеченность", f"{engagement:.1%}")
            
            # Топ тем
            st.subheader("🔥 Популярные темы")
            
            top_topics = followup_stats.get('top_topics', [])
            if top_topics:
                topic_cols = st.columns(2)
                
                with topic_cols[0]:
                    df_topics = pd.DataFrame(top_topics)
                    df_topics['engagement_pct'] = df_topics['engagement'] * 100
                    
                    st.dataframe(
                        df_topics[['topic', 'count', 'engagement_pct']],
                        use_container_width=True,
                        column_config={
                            "topic": "Тема",
                            "count": "Количество",
                            "engagement_pct": st.column_config.NumberColumn(
                                "Вовлеченность (%)",
                                format="%.1f%%"
                            )
                        },
                        hide_index=True
                    )
                
                with topic_cols[1]:
                    # Радар-чарт эффективности по темам
                    topics = [t['topic'] for t in top_topics[:5]]
                    engagement_values = [t['engagement'] for t in top_topics[:5]]
                    
                    # Нормализуем значения для радар-чарта
                    if engagement_values:
                        max_val = max(engagement_values)
                        normalized_values = [v / max_val for v in engagement_values]
                    else:
                        normalized_values = [0] * len(topics)
                    
                    st.plotly_chart(
                        create_radar_chart(topics, normalized_values, "Эффективность по темам"),
                        use_container_width=True
                    )
            
            # Эффективность генерации
            st.subheader("📈 Эффективность генерации")
            
            effectiveness = followup_stats.get('effectiveness', {})
            
            eff_cols = st.columns(3)
            
            with eff_cols[0]:
                high_engagement = effectiveness.get('high_engagement', 0) * 100
                st.plotly_chart(
                    create_gauge_chart(
                        high_engagement,
                        100,
                        "Высокая вовлеченность",
                        "#4CAF50"
                    ),
                    use_container_width=True
                )
            
            with eff_cols[1]:
                medium_engagement = effectiveness.get('medium_engagement', 0) * 100
                st.plotly_chart(
                    create_gauge_chart(
                        medium_engagement,
                        100,
                        "Средняя вовлеченность",
                        "#FF9800"
                    ),
                    use_container_width=True
                )
            
            with eff_cols[2]:
                low_engagement = effectiveness.get('low_engagement', 0) * 100
                st.plotly_chart(
                    create_gauge_chart(
                        low_engagement,
                        100,
                        "Низкая вовлеченность",
                        "#F44336"
                    ),
                    use_container_width=True
                )
            
            # Тренд
            trend = followup_stats.get('trend', 'stable')
            if trend == 'positive':
                st.success("📈 Положительный тренд вовлеченности")
            elif trend == 'negative':
                st.error("📉 Отрицательный тренд вовлеченности")
            else:
                st.info("➡️ Стабильный тренд вовлеченности")
        else:
            st.info("Данные follow-up генератора временно недоступны")
    
    # Вкладка 5: RL Агент
    with tab5:
        st.header("🤖 RL Агент")
        
        if rl_stats:
            # Основные метрики
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                updates = rl_stats.get('q_table_updates', 0)
                st.metric("Обновления Q-таблицы", format_number(updates))
            
            with metric_cols[1]:
                exploration_rate = rl_stats.get('exploration_rate', 0)
                st.metric("Исследование", f"{exploration_rate:.1%}")
            
            with metric_cols[2]:
                learning_rate = rl_stats.get('learning_rate', 0)
                st.metric("Обучение", f"{learning_rate:.3f}")
            
            with metric_cols[3]:
                rewards = rl_stats.get('rewards_received', 0)
                st.metric("Получено наград", format_number(rewards))
            
            # Дополнительные метрики
            extra_cols = st.columns(2)
            with extra_cols[0]:
                explorations = rl_stats.get('explorations', 0)
                exploitations = rl_stats.get('exploitations', 0)
                total = explorations + exploitations
                if total > 0:
                    exploration_ratio = explorations / total
                    st.metric("Соотношение исследование/использование", 
                             f"{exploration_ratio:.1%}")
            
            with extra_cols[1]:
                avg_reward = rl_stats.get('performance', {}).get('avg_reward', 0)
                st.metric("Средняя награда", f"{avg_reward:.2f}")
            
            # Q-таблица состояния
            st.subheader("🎯 Распределение действий по состояниям")
            
            states_data = rl_stats.get('states', {})
            
            if states_data:
                # Подготавливаем данные для тепловой карты
                states = list(states_data.keys())
                actions = ['high', 'medium', 'low', 'cautious']
                
                heatmap_data = []
                for state in states:
                    row = []
                    for action in actions:
                        row.append(states_data[state].get(action, 0))
                    heatmap_data.append(row)
                
                # Перевод названий состояний
                state_labels = {
                    "factual_with_knowledge": "Факты + знания",
                    "factual_no_knowledge": "Факты без знаний",
                    "conversational_with_knowledge": "Диалог + знания",
                    "conversational_no_knowledge": "Диалог без знаний",
                    "ambiguous": "Неоднозначные"
                }
                
                translated_states = [state_labels.get(state, state) for state in states]
                translated_actions = ['Высокое', 'Среднее', 'Низкое', 'Осторожное']
                
                st.plotly_chart(
                    create_heatmap(
                        heatmap_data,
                        "Q-значения по состояниям и действиям",
                        translated_actions,
                        translated_states
                    ),
                    use_container_width=True
                )
            
            # Производительность агента
            st.subheader("📊 Производительность агента")
            
            performance = rl_stats.get('performance', {})
            
            perf_cols = st.columns(3)
            
            with perf_cols[0]:
                avg_reward = performance.get('avg_reward', 0)
                st.plotly_chart(
                    create_gauge_chart(
                        avg_reward * 100,
                        100,
                        "Средняя награда",
                        "#2196F3"
                    ),
                    use_container_width=True
                )
            
            with perf_cols[1]:
                success_rate = performance.get('success_rate', 0)
                st.plotly_chart(
                    create_gauge_chart(
                        success_rate * 100,
                        100,
                        "Успешность",
                        "#4CAF50"
                    ),
                    use_container_width=True
                )
            
            with perf_cols[2]:
                trend = performance.get('improvement_trend', 'stable')
                if trend == 'positive':
                    st.success("📈 Положительный тренд")
                elif trend == 'negative':
                    st.error("📉 Отрицательный тренд")
                else:
                    st.info("➡️ Стабильный тренд")
                st.metric("Тренд улучшения", trend.capitalize())
        else:
            st.info("Данные RL агента временно недоступны")
    
    # Вкладка 6: Система
    with tab6:
        st.header("⚙️ Системные компоненты")
        
        if system_info and "components" in system_info:
            components = system_info["components"]
            
            st.subheader("🔄 Состояние компонентов")
            
            comp_cols = st.columns(4)
            component_names = ["cache", "vector_store", "memory_manager", "orchestrator"]
            component_labels = ["Кэш", "Векторное хранилище", "Менеджер памяти", "Оркестратор"]
            
            for idx, (comp_name, comp_label) in enumerate(zip(component_names, component_labels)):
                with comp_cols[idx]:
                    comp_status = components.get(comp_name, {})
                    status = comp_status.get("status", "unknown")
                    
                    if status == "initialized":
                        st.success(f"✅ {comp_label}")
                    elif status == "not_initialized":
                        st.error(f"❌ {comp_label}")
                    else:
                        st.info(f"ℹ️ {comp_label}")
                    
                    # Дополнительная информация
                    if comp_name == "cache" and "stats" in comp_status:
                        size = comp_status['stats'].get('size', 0)
                        hit_rate = comp_status['stats'].get('hit_rate', 0)
                        st.caption(f"Размер: {size}")
                        st.caption(f"Hit Rate: {hit_rate:.1%}")
                    elif comp_name == "vector_store" and "total_chunks" in comp_status:
                        chunks = comp_status.get('total_chunks', 0)
                        st.caption(f"Чанков: {chunks}")
                    elif comp_name == "memory_manager":
                        sessions = comp_status.get('active_sessions', 0)
                        st.caption(f"Сессии: {sessions}")
                    elif comp_name == "orchestrator":
                        throughput = comp_status.get('throughput', 0)
                        st.caption(f"Пропускная способность: {throughput}/сек")
        
        # Настройки системы
        st.subheader("⚙️ Настройки системы")
        
        if system_info:
            setting_cols = st.columns(2)
            
            with setting_cols[0]:
                st.write("**📋 Конфигурация:**")
                
                config_data = {
                    "Кэш включен": system_info.get("settings", {}).get("cache_enabled", True),
                    "Режим ChromaDB": system_info.get("settings", {}).get("chroma_mode", "persistent"),
                    "Модель эмбеддингов": system_info.get("settings", {}).get("embedding_model", "unknown"),
                    "Размер чанков": system_info.get("settings", {}).get("chunk_size", 512),
                    "Перекрытие чанков": system_info.get("settings", {}).get("chunk_overlap", 50)
                }
                
                for key, value in config_data.items():
                    st.write(f"• **{key}:** `{value}`")
            
            with setting_cols[1]:
                st.write("**📊 Версия системы:**")
                st.info(f"**Версия:** {system_info.get('version', '1.0.0')}")
                st.info(f"**Статус:** {system_info.get('status', 'operational').capitalize()}")
                st.info(f"**Время работы:** {system_info.get('uptime', 'Неизвестно')}")
        
        # Сырые данные для отладки
        with st.expander("🔍 Сырые данные для отладки", expanded=False):
            debug_cols = st.columns(2)
            
            with debug_cols[0]:
                st.subheader("Системная информация")
                st.json(system_info or {})
            
            with debug_cols[1]:
                st.subheader("Метрики")
                if metrics:
                    # Отображаем только первые 10 метрик для читаемости
                    metric_items = list(metrics.items())[:10]
                    for name, value in metric_items:
                        st.text(f"{name}: {value}")
                else:
                    st.info("Метрики недоступны")
    
    # Вкладка 7: Обратная связь (НОВАЯ ВКЛАДКА)
    with tab7:
        st.header("👍 Обратная связь")
        
        feedback_stats = get_feedback_stats()
        
        if feedback_stats:
            # Основные метрики
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                total_feedback = feedback_stats["total_feedback"]
                st.metric("Всего оценок", format_number(total_feedback))
            
            with metric_cols[1]:
                average_rating = feedback_stats["average_rating"]
                st.metric("Средний рейтинг", f"{average_rating:.2f}")
            
            with metric_cols[2]:
                satisfaction_rate = feedback_stats["satisfaction_rate"]
                st.metric("Удовлетворенность", f"{satisfaction_rate:.1%}")
            
            with metric_cols[3]:
                trend = feedback_stats["trend"]
                if trend == "positive":
                    st.success("📈 Положительный тренд")
                elif trend == "negative":
                    st.error("📉 Отрицательный тренд")
                else:
                    st.info("➡️ Стабильный тренд")
                st.metric("Тренд", trend.capitalize())
            
            # Распределение по рейтингам
            st.subheader("⭐ Распределение по рейтингам")
            
            distribution = feedback_stats["distribution"]
            rating_cols = st.columns(2)
            
            with rating_cols[0]:
                labels = ["★☆☆☆☆ (1)", "★★☆☆☆ (2)", "★★★☆☆ (3)", "★★★★☆ (4)", "★★★★★ (5)"]
                values = [distribution.get("1", 0), distribution.get("2", 0), 
                         distribution.get("3", 0), distribution.get("4", 0), 
                         distribution.get("5", 0)]
                colors = ['#F44336', '#FF9800', '#FFC107', '#4CAF50', '#2196F3']
                
                fig = go.Figure(data=[go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors,
                    text=values,
                    textposition='auto'
                )])
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="Рейтинг",
                    yaxis_title="Количество",
                    plot_bgcolor='rgba(0,0,0,0.02)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with rating_cols[1]:
                # Круговая диаграмма распределения
                rating_labels = ["1 звезда", "2 звезды", "3 звезды", "4 звезды", "5 звезд"]
                st.plotly_chart(
                    create_pie_chart(rating_labels, values, "Распределение рейтингов"),
                    use_container_width=True
                )
            
            # Полезность ответов
            st.subheader("🎯 Полезность ответов")
            
            helpfulness = feedback_stats["helpfulness"]
            helpful_cols = st.columns(2)
            
            with helpful_cols[0]:
                labels = ["Полезные", "Не полезные", "Без оценки"]
                values = [
                    helpfulness.get("helpful", 0),
                    helpfulness.get("not_helpful", 0),
                    helpfulness.get("no_feedback", 0)
                ]
                colors = ['#4CAF50', '#F44336', '#9E9E9E']
                
                st.plotly_chart(
                    create_pie_chart(labels, values, "Распределение по полезности"),
                    use_container_width=True
                )
            
            with helpful_cols[1]:
                # Метрики полезности
                helpful_total = helpfulness.get("helpful", 0)
                not_helpful_total = helpfulness.get("not_helpful", 0)
                total_with_feedback = helpful_total + not_helpful_total
                
                if total_with_feedback > 0:
                    helpful_rate = helpful_total / total_with_feedback
                else:
                    helpful_rate = 0
                
                st.metric("Коэффициент полезности", f"{helpful_rate:.1%}")
                st.metric("Полезные ответы", helpful_total)
                st.metric("Не полезные ответы", not_helpful_total)
            
            # Последние отзывы
            st.subheader("💬 Последние отзывы")
            
            recent_feedback = feedback_stats.get("recent_feedback", [])
            if recent_feedback:
                for feedback in recent_feedback:
                    rating = feedback.get("rating", 0)
                    comment = feedback.get("comment", "")
                    helpful = feedback.get("helpful", None)
                    timestamp = feedback.get("timestamp", "")
                    
                    # Форматируем время
                    try:
                        time_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = time_obj.strftime("%H:%M:%S")
                    except:
                        time_str = timestamp
                    
                    # Создаем контейнер для отзыва
                    with st.container():
                        col1, col2, col3 = st.columns([1, 3, 1])
                        
                        with col1:
                            # Отображаем звезды
                            stars = "⭐" * rating + "☆" * (5 - rating)
                            st.write(f"**{stars}**")
                        
                        with col2:
                            st.write(comment)
                            st.caption(f"Время: {time_str}")
                        
                        with col3:
                            if helpful is True:
                                st.success("👍 Полезно")
                            elif helpful is False:
                                st.error("👎 Не полезно")
                            else:
                                st.info("🤔 Нет оценки")
                        
                        st.divider()
            else:
                st.info("Нет недавних отзывов")
            
            # Динамика фидбека за время
            st.subheader("📈 Динамика обратной связи")
            
            # Генерация тестовых данных динамики
            hours = list(range(24))
            time_labels = [f"{h:02d}:00" for h in hours]
            
            # Динамика рейтингов в течение дня
            base_rating = 4.2
            rating_trend = [base_rating + 0.3 * np.sin(h/12 * np.pi) + 0.1 * np.random.randn() for h in hours]
            
            # Динамика количества фидбеков
            base_feedback = 50
            feedback_trend = [int(base_feedback * (1 + 0.5 * np.sin(h/24 * 2 * np.pi) + 0.2 * np.random.rand())) for h in hours]
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Средний рейтинг по часам", "Количество оценок по часам"),
                vertical_spacing=0.2
            )
            
            fig.add_trace(
                go.Scatter(x=time_labels, y=rating_trend, mode='lines+markers',
                         name='Рейтинг', line=dict(color='#2196F3', width=2)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=time_labels, y=feedback_trend, name='Количество',
                     marker_color='#4CAF50'),
                row=2, col=1
            )
            
            fig.update_layout(
                height=500, 
                showlegend=False, 
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor='rgba(0,0,0,0.02)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Статистика обратной связи временно недоступна")
    
    # Футер
    st.markdown("---")
    footer_cols = st.columns(3)
    with footer_cols[0]:
        st.caption(f"🔄 Последнее обновление: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    with footer_cols[1]:
        st.caption(f"📊 Версия дашборда: 2.1.0")
    with footer_cols[2]:
        if st.button("🔄 Обновить данные", type="primary", use_container_width=True):
            st.rerun()
    
    # Автообновление
    if st.session_state.auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()