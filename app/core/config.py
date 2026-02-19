# app/core/config.py
"""
Единая конфигурация проекта с валидацией
"""
from pydantic_settings import BaseSettings
from pydantic import Field, validator, PostgresDsn, ConfigDict
from typing import List, Optional, Union
from enum import Enum
from pathlib import Path


class Environment(str, Enum):
    """Окружения"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Уровни логирования"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Настройки приложения.
    Приоритет: env переменные > .env файл > значения по умолчанию
    """
    
    # ВАЖНО: Добавляем model_config с extra="ignore"
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # <-- Игнорировать лишние поля
    )
    
    # === ОКРУЖЕНИЕ ===
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = Field(default=False, description="Режим отладки")
    LOG_LEVEL: LogLevel = LogLevel.INFO
    
    # === API ===
    API_HOST: str = Field(default="0.0.0.0", description="Хост API")
    API_PORT: int = Field(default=8000, ge=1024, le=65535, description="Порт API")
    API_WORKERS: int = Field(default=1, ge=1, le=32, description="Количество воркеров")
    API_TITLE: str = "AI Knowledge Assistant"
    API_DESCRIPTION: str = "Продакшен-готовый AI ассистент с базой знаний"
    API_VERSION: str = "1.0.0"
    
    # === БАЗЫ ДАННЫХ ===
    # PostgreSQL для SQL данных
    DATABASE_URL: PostgresDsn = Field(
        default="postgresql://ai_user:password@localhost:5432/ai_core",
        description="URL PostgreSQL базы данных"
    )
    # ДОБАВЛЕНО:
    DATABASE_POOL_SIZE: int = Field(default=20, description="Размер пула соединений")
    DATABASE_MAX_OVERFLOW: int = Field(default=40, description="Максимальный переполнение пула")
    
    # ChromaDB для векторных данных
    CHROMA_HOST: str = Field(default="localhost", description="Хост ChromaDB")
    CHROMA_PORT: int = Field(default=8000, ge=1024, le=65535, description="Порт ChromaDB")
    CHROMA_COLLECTION: str = Field(default="knowledge_base", description="Коллекция ChromaDB")
    # ДОБАВЛЕНО:
    CHROMA_MODE: str = Field(default="persistent", description="Режим ChromaDB")
    CHROMA_PERSIST_DIR: str = Field(default="./chroma_data", description="Директория для хранения ChromaDB")
    
    # Redis для кэша
    REDIS_URL: Optional[str] = Field(
        default="redis://localhost:6379/0",
        description="URL Redis для кэширования"
    )
    
    # === МОДЕЛИ И ЭМБЕДДИНГИ ===
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="Модель для создания эмбеддингов"
    )
    EMBEDDING_DEVICE: str = Field(default="cpu", pattern="^(cpu|cuda)$")
    EMBEDDING_DIMENSION: int = Field(default=768, ge=128, le=4096)
    
    # LLM модель (опционально)
    LLM_MODEL: Optional[str] = Field(default=None, description="Модель для генерации текста")
    LLM_API_KEY: Optional[str] = Field(default=None, description="API ключ для LLM")
    
    # === КЭШИРОВАНИЕ ===
    CACHE_ENABLED: bool = Field(default=True, description="Включить кэширование")
    CACHE_TTL_SECONDS: int = Field(default=3600, ge=60, le=86400, description="Время жизни кэша")
    CACHE_MAX_SIZE: int = Field(default=10000, ge=100, le=100000, description="Максимальный размер кэша")
    
    # === СЕССИИ (ДОБАВЛЕНО) ===
    SESSION_TTL: int = Field(default=86400, description="Время жизни сессии (секунды)")
    MAX_HISTORY_LENGTH: int = Field(default=50, description="Максимальная длина истории")
    CONTEXT_WINDOW: int = Field(default=10, description="Размер окна контекста")
    
    # === УЛУЧШЕНИЯ ===
    ENABLE_AB_TESTING: bool = Field(default=True, description="Включить A/B тестирование")
    ENABLE_RL_AGENT: bool = Field(default=True, description="Включить RL агент")
    ENABLE_FOLLOWUP: bool = Field(default=True, description="Включить follow-up генератор")
    # ДОБАВЛЕНО (алиасы для совместимости):
    ENHANCEMENTS_ENABLED: bool = Field(default=True, description="Включить улучшения")
    AB_TESTING_ENABLED: bool = Field(default=True, description="Включить A/B тестирование")
    RL_LEARNING_ENABLED: bool = Field(default=True, description="Включить RL обучение")
    
    # === МОНИТОРИНГ ===
    METRICS_ENABLED: bool = Field(default=True, description="Включить сбор метрик")
    METRICS_PORT: int = Field(default=9090, description="Порт для метрик Prometheus")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, description="Интервал health checks (секунды)")
    
    # === БЕЗОПАСНОСТЬ ===
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        min_length=32,
        description="Секретный ключ для подписи"
    )
    API_KEYS: List[str] = Field(default=[], description="Список валидных API ключей")
    # ДОБАВЛЕНО:
    API_KEY_REQUIRED: bool = Field(default=False, description="Требовать API ключ")
    CORS_ORIGINS: List[str] = Field(default=["*"], description="Допустимые CORS origins")
    
    # === RATE LIMITING (ДОБАВЛЕНО) ===
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Максимальное количество запросов")
    RATE_LIMIT_PERIOD: int = Field(default=60, description="Период для rate limiting (секунды)")
    
    # === ЛОГИРОВАНИЕ (ДОБАВЛЕНО) ===
    LOG_FORMAT: str = Field(default="json", description="Формат логов")
    LOG_FILE: str = Field(default="./logs/ai_assistant.log", description="Файл для логов")
    
    # === ПУТИ ===
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = Field(default=BASE_DIR / "data")
    LOGS_DIR: Path = Field(default=BASE_DIR / "logs")
    MODELS_DIR: Path = Field(default=BASE_DIR / "models")
    UPLOADS_DIR: Path = Field(default=BASE_DIR / "uploads")
    
    @validator("DATA_DIR", "LOGS_DIR", "MODELS_DIR", "UPLOADS_DIR", pre=True)
    def create_directories(cls, v: Path) -> Path:
        """Создает директории при инициализации"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("API_KEYS", pre=True)
    def parse_api_keys(cls, v):
        """Парсит API ключи из строки"""
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return v
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Парсит CORS origins из строки"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @property
    def is_production(self) -> bool:
        """Проверка на продакшен окружение"""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Проверка на разработку"""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @property
    def chromadb_url(self) -> str:
        """URL для подключения к ChromaDB"""
        return f"http://{self.CHROMA_HOST}:{self.CHROMA_PORT}"
    
    def get_database_url(self, async_: bool = False) -> str:
        """Получить URL базы данных с учетом режима (async/sync)"""
        url = str(self.DATABASE_URL)
        if async_ and url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://")
        return url


# Глобальный экземпляр настроек
try:
    settings = Settings()
    print(f"✅ Настройки загружены. Режим: {settings.ENVIRONMENT}")
    print(f"📊 DEBUG: {settings.DEBUG}")
except Exception as e:
    print(f"❌ Ошибка загрузки настроек: {e}")
    raise

# Экспорт для быстрого доступа
__all__ = ["settings", "Environment", "LogLevel"]