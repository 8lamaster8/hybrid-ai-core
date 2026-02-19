"""
Инициализация приложения с ленивой загрузкой
"""

import sys
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Экспортируем только настройки и утилиты
from app.core.config import settings
from app.core.logging import setup_logging

# Настраиваем логирование при импорте
logger = setup_logging()

__version__ = "1.0.0"
__all__ = ["settings", "logger"]