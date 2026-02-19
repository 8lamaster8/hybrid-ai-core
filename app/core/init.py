"""
Ядро системы - центральная точка инициализации
"""
import sys
from pathlib import Path
from typing import Optional

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Импортируем настройки и логирование
from app.core.config import settings
from app.core.logging import logger, performance_logger

# Инициализируем глобальные переменные
brain_instance: Optional['AIBrain'] = None  # type: ignore
knowledge_base_instance: Optional['KnowledgeBase'] = None  # type: ignore
cache_instance: Optional['Cache'] = None  # type: ignore

# Structlog логгер (инициализируется в setup_logging)
struct_logger = None

# Версия
__version__ = settings.API_VERSION
__author__ = "AI Core Team"
__description__ = "Продакшен-готовый AI ассистент с базой знаний"


def get_brain():
    """Ленивое получение AI мозга"""
    global brain_instance
    if brain_instance is None:
        from app.core.brain import AIBrain
        brain_instance = AIBrain()
    return brain_instance


def get_knowledge_base():
    """Ленивое получение базы знаний"""
    global knowledge_base_instance
    if knowledge_base_instance is None:
        from app.services.knowledge.base import KnowledgeBase
        knowledge_base_instance = KnowledgeBase()
    return knowledge_base_instance


def get_cache():
    """Ленивое получение кэша"""
    global cache_instance
    if cache_instance is None:
        from app.infrastructure.cache import Cache
        cache_instance = Cache()
    return cache_instance


def initialize_system():
    """Инициализация всей системы"""
    logger.info("Начинаем инициализацию системы...")
    
    try:
        # Инициализируем компоненты
        brain = get_brain()
        knowledge_base = get_knowledge_base()
        cache = get_cache()
        
        logger.info("✅ Система инициализирована", extra={
            "context": {
                "components": ["brain", "knowledge_base", "cache"],
                "environment": settings.ENVIRONMENT.value
            }
        })
        
        return {
            "brain": brain,
            "knowledge_base": knowledge_base,
            "cache": cache
        }
    
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации системы: {e}", exc_info=True)
        raise


# Экспорт
__all__ = [
    "settings",
    "logger",
    "performance_logger",
    "struct_logger",
    "get_brain",
    "get_knowledge_base",
    "get_cache",
    "initialize_system",
    "__version__",
    "__author__",
    "__description__"
]

# Автоматическая инициализация в разработке
if settings.is_development:
    logger.info("Режим разработки: автоматическая инициализация...")
    try:
        initialize_system()
    except:
        logger.warning("Автоматическая инициализация не удалась")