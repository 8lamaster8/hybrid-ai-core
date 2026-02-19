"""
Сбор и экспорт метрик для Prometheus
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY, start_http_server
import time
from typing import Dict, Any, Optional 
import threading
from datetime import datetime
import sys
import os

# Добавляем путь для корректных импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logging import logger

# ========== ОСНОВНЫЕ МЕТРИКИ ==========

QUESTIONS_PROCESSED = Counter(
    'ai_questions_processed_total',
    'Total number of questions processed'
)

PROCESSING_TIME = Histogram(
    'ai_question_processing_seconds',
    'Time spent processing question',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

CACHE_HITS = Counter(
    'ai_cache_hits_total',
    'Total number of cache hits'
)

CACHE_MISSES = Counter(
    'ai_cache_misses_total',
    'Total number of cache misses'
)

SESSION_COUNT = Gauge(
    'ai_active_sessions',
    'Number of active sessions'
)

KNOWLEDGE_CHUNKS = Gauge(
    'ai_knowledge_chunks_total',
    'Total number of knowledge chunks'
)

ERROR_COUNT = Counter(
    'ai_errors_total',
    'Total number of errors',
    ['error_type']
)

MEMORY_USAGE = Gauge(
    'ai_memory_usage_bytes',
    'Memory usage of the application'
)

REQUEST_DURATION = Histogram(
    'ai_request_duration_seconds',
    'Duration of HTTP requests',
    ['endpoint', 'method'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

# ========== МЕТРИКИ ОБРАТНОЙ СВЯЗИ ==========

FEEDBACK_TOTAL = Counter(
    'ai_feedback_total',
    'Total number of feedback submissions'
)

FEEDBACK_RATING = Counter(
    'ai_feedback_rating',
    'Feedback ratings by score',
    ['rating']
)

FEEDBACK_HELPFUL = Counter(
    'ai_feedback_helpful',
    'Feedback helpfulness',
    ['status']
)

FEEDBACK_COMMENT_LENGTH = Histogram(
    'ai_feedback_comment_length',
    'Length of feedback comments',
    buckets=[10, 50, 100, 200, 500, 1000]
)

# ========== МЕТРИКИ ДЛЯ RL AGENT ==========

RL_AGENT_UPDATES = Counter(
    'ai_rl_agent_updates_total',
    'Total number of RL agent updates'
)

RL_AGENT_EXPLORATIONS = Counter(
    'ai_rl_agent_explorations_total',
    'Total number of RL agent explorations'
)

RL_AGENT_REWARDS = Histogram(
    'ai_rl_agent_rewards',
    'RL agent rewards distribution',
    buckets=[-1.0, -0.5, 0, 0.5, 1.0]
)

# ========== МЕТРИКИ БАЗЫ ДАННЫХ ==========

DB_CONNECTIONS = Gauge(
    'ai_db_connections',
    'Database connections count'
)

DB_QUERY_DURATION = Histogram(
    'ai_db_query_duration_seconds',
    'Database query duration',
    ['operation'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
)

# ========== МЕТРИКИ КЭША ==========

CACHE_SIZE = Gauge(
    'ai_cache_size',
    'Current cache size'
)

CACHE_EVICTIONS = Counter(
    'ai_cache_evictions_total',
    'Total number of cache evictions'
)


class MetricsCollector:
    """Сборщик метрик системы"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics_cache = None
        self._last_update = None
        self._metrics_server_started = False
        self._start_time = time.time()
    
    def start_metrics_server(self, port: int = 8001):
        """Запуск сервера метрик Prometheus"""
        if not self._metrics_server_started:
            try:
                start_http_server(port)
                self._metrics_server_started = True
                logger.info(f"✅ Prometheus metrics server started on port {port}")
            except Exception as e:
                logger.error(f"❌ Failed to start metrics server: {e}")
                # Создаем заглушку для разработки
                self._create_stub_methods()
    
    def _create_stub_methods(self):
        """Создание заглушек для методов метрик"""
        logger.warning("Созданы заглушки для методов метрик")
        
        # Заглушки для основных методов
        self.record_question_processing = lambda *args, **kwargs: None
        self.record_error = lambda *args, **kwargs: None
        self.record_feedback = lambda *args, **kwargs: None
        self.record_request = lambda *args, **kwargs: None
    
    # ========== ОСНОВНЫЕ МЕТРИКИ ==========
    
    def record_question_processing(self, start_time: float, from_cache: bool = False):
        """Запись метрик обработки вопроса"""
        try:
            processing_time = time.time() - start_time
            
            QUESTIONS_PROCESSED.inc()
            PROCESSING_TIME.observe(processing_time)
            
            if from_cache:
                CACHE_HITS.inc()
            else:
                CACHE_MISSES.inc()
                
            logger.debug(f"Метрики вопроса записаны: time={processing_time:.2f}s, cache={from_cache}")
            
        except Exception as e:
            logger.warning(f"Не удалось записать метрики вопроса: {e}")
    
    def record_error(self, error_type: str = "general"):
        """Запись метрики ошибки"""
        try:
            ERROR_COUNT.labels(error_type=error_type).inc()
            logger.debug(f"Метрика ошибки записана: {error_type}")
        except Exception as e:
            logger.warning(f"Не удалось записать метрику ошибки: {e}")
    
    def update_session_count(self, count: int):
        """Обновление количества сессий"""
        try:
            SESSION_COUNT.set(count)
        except Exception as e:
            logger.warning(f"Не удалось обновить количество сессий: {e}")
    
    def update_knowledge_chunks(self, count: int):
        """Обновление количества чанков знаний"""
        try:
            KNOWLEDGE_CHUNKS.set(count)
        except Exception as e:
            logger.warning(f"Не удалось обновить количество чанков: {e}")
    
    def update_memory_usage(self):
        """Обновление использования памяти"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            MEMORY_USAGE.set(memory_info.rss)
            
        except ImportError:
            logger.debug("psutil не установлен, пропускаем метрики памяти")
        except Exception as e:
            logger.debug(f"Не удалось обновить использование памяти: {e}")
    
    # ========== МЕТРИКИ ОБРАТНОЙ СВЯЗИ ==========
    
    def record_feedback(self, rating: int, helpful: Optional[bool] = None, comment: Optional[str] = None):
        """Запись метрик обратной связи"""
        try:
            FEEDBACK_TOTAL.inc()
            FEEDBACK_RATING.labels(rating=str(rating)).inc()
            
            if helpful is not None:
                status = "helpful" if helpful else "not_helpful"
            else:
                status = "unknown"
            FEEDBACK_HELPFUL.labels(status=status).inc()
            
            if comment:
                FEEDBACK_COMMENT_LENGTH.observe(len(comment))
            
            logger.debug(f"Метрики фидбека записаны: rating={rating}, helpful={helpful}")
            
        except Exception as e:
            logger.warning(f"Не удалось записать метрики фидбека: {e}")
    
    # ========== МЕТРИКИ HTTP ЗАПРОСОВ ==========
    
    def record_request(self, endpoint: str, method: str, duration: float, status_code: int = 200):
        """Запись метрики HTTP запроса"""
        try:
            REQUEST_DURATION.labels(endpoint=endpoint, method=method).observe(duration)
            logger.debug(f"Метрика запроса: {method} {endpoint} - {duration:.2f}s")
        except Exception as e:
            logger.warning(f"Не удалось записать метрику запроса: {e}")
    
    # ========== МЕТРИКИ RL AGENT ==========
    
    def record_rl_update(self):
        """Запись обновления RL агента"""
        try:
            RL_AGENT_UPDATES.inc()
        except Exception as e:
            logger.debug(f"Не удалось записать метрику RL агента: {e}")
    
    def record_rl_exploration(self):
        """Запись исследования RL агента"""
        try:
            RL_AGENT_EXPLORATIONS.inc()
        except Exception as e:
            logger.debug(f"Не удалось записать метрику исследования RL: {e}")
    
    def record_rl_reward(self, reward: float):
        """Запись награды RL агента"""
        try:
            RL_AGENT_REWARDS.observe(reward)
        except Exception as e:
            logger.debug(f"Не удалось записать метрику награды RL: {e}")
    
    # ========== МЕТРИКИ БАЗЫ ДАННЫХ ==========
    
    def record_db_query(self, operation: str, duration: float):
        """Запись метрики запроса к БД"""
        try:
            DB_QUERY_DURATION.labels(operation=operation).observe(duration)
        except Exception as e:
            logger.debug(f"Не удалось записать метрику БД: {e}")
    
    def update_db_connections(self, count: int):
        """Обновление количества соединений с БД"""
        try:
            DB_CONNECTIONS.set(count)
        except Exception as e:
            logger.debug(f"Не удалось обновить метрику соединений БД: {e}")
    
    # ========== МЕТРИКИ КЭША ==========
    
    def update_cache_size(self, size: int):
        """Обновление размера кэша"""
        try:
            CACHE_SIZE.set(size)
        except Exception as e:
            logger.debug(f"Не удалось обновить метрику размера кэша: {e}")
    
    def record_cache_eviction(self):
        """Запись вытеснения из кэша"""
        try:
            CACHE_EVICTIONS.inc()
        except Exception as e:
            logger.debug(f"Не удалось записать метрику вытеснения кэша: {e}")
    
    # ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========
    
    def get_all_metrics(self) -> str:
        """Получение всех метрик в формате Prometheus"""
        with self._lock:
            try:
                # Обновляем динамические метрики
                self.update_memory_usage()
                
                metrics = generate_latest(REGISTRY)
                return metrics.decode('utf-8')
            except Exception as e:
                logger.error(f"Не удалось получить метрики: {e}")
                return ""
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Получение сводки метрик в JSON формате"""
        try:
            uptime = time.time() - self._start_time
            
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": uptime,
                "metrics": {
                    "questions_processed": self._get_counter_value(QUESTIONS_PROCESSED),
                    "cache_hits": self._get_counter_value(CACHE_HITS),
                    "cache_misses": self._get_counter_value(CACHE_MISSES),
                    "errors": self._get_counter_value(ERROR_COUNT),
                    "session_count": SESSION_COUNT._value.get(),
                    "knowledge_chunks": KNOWLEDGE_CHUNKS._value.get(),
                    "feedback_total": self._get_counter_value(FEEDBACK_TOTAL),
                    "rl_agent_updates": self._get_counter_value(RL_AGENT_UPDATES),
                }
            }
        except Exception as e:
            logger.error(f"Не удалось получить сводку метрик: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _get_counter_value(self, counter) -> float:
        """Безопасное получение значения счетчика"""
        try:
            return counter._value.get()
        except:
            return 0.0
    
    def reset_metrics(self):
        """Сброс всех метрик (для тестов)"""
        try:
            # Очищаем все коллекторы
            collectors = list(REGISTRY._collector_to_names.keys())
            for collector in collectors:
                REGISTRY.unregister(collector)
            
            logger.info("Метрики сброшены")
        except Exception as e:
            logger.error(f"Не удалось сбросить метрики: {e}")


# ========== ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР И УДОБНЫЕ ФУНКЦИИ ==========

metrics_collector = MetricsCollector()


def get_metrics() -> str:
    """Получение метрик в формате Prometheus"""
    return metrics_collector.get_all_metrics()


def get_metrics_json() -> Dict[str, Any]:
    """Получение метрик в JSON формате"""
    return metrics_collector.get_metrics_summary()


def record_question(start_time: float, from_cache: bool = False):
    """Упрощенный интерфейс для записи метрик вопроса"""
    metrics_collector.record_question_processing(start_time, from_cache)


def record_request(endpoint: str, method: str, duration: float, status_code: int = 200):
    """Упрощенный интерфейс для записи метрик HTTP запроса"""
    metrics_collector.record_request(endpoint, method, duration, status_code)


async def start_metrics_collection(port: int = 8001):
    """Запуск сбора метрик"""
    try:
        metrics_collector.start_metrics_server(port)
        logger.info(f"✅ Metrics collection started on port {port}")
        
        # Запускаем фоновое обновление метрик
        import asyncio
        asyncio.create_task(_background_metrics_update())
        
    except Exception as e:
        logger.error(f"❌ Failed to start metrics collection: {e}")
        # Создаем заглушку для продолжения работы
        metrics_collector._create_stub_methods()


async def _background_metrics_update():
    """Фоновое обновление метрик"""
    import asyncio
    
    while True:
        try:
            # Обновляем память каждые 30 секунд
            metrics_collector.update_memory_usage()
            
            # Можно добавить другие периодические обновления
            
            await asyncio.sleep(30)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"Ошибка фонового обновления метрик: {e}")
            await asyncio.sleep(30)


# ========== MIDDLEWARE ДЛЯ FASTAPI ==========

def create_metrics_middleware():
    """Создание middleware для автоматического сбора метрик"""
    from fastapi import Request
    import time
    
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Записываем метрику запроса
            metrics_collector.record_request(
                endpoint=str(request.url.path),
                method=request.method,
                duration=duration,
                status_code=response.status_code
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            metrics_collector.record_error("middleware")
            
            # Все равно записываем метрику
            metrics_collector.record_request(
                endpoint=str(request.url.path),
                method=request.method,
                duration=duration,
                status_code=500
            )
            
            raise
    
    return metrics_middleware


# ========== ДЕКОРАТОРЫ ДЛЯ ЛЕГКОГО ИСПОЛЬЗОВАНИЯ ==========

def track_time(metric_name: str = None):
    """Декоратор для отслеживания времени выполнения функции"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Записываем метрику
                if metric_name:
                    PROCESSING_TIME.observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_error(type(e).__name__)
                raise
        return wrapper
    return decorator


def count_calls(counter_name: str):
    """Декоратор для подсчета вызовов функции"""
    def decorator(func):
        # Создаем динамический счетчик
        counter = Counter(
            f'ai_{counter_name}_calls_total',
            f'Total calls of {func.__name__}'
        )
        
        def wrapper(*args, **kwargs):
            counter.inc()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator