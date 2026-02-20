"""
📝 Логирование системы - структурированные логи с ротацией
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Optional

# Форматы логов
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
JSON_FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'  # можно заменить на json

# Уровни логирования
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO, #info,WARNING
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Глобальный экземпляр логгера
_logger = None


def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    max_bytes: int = 10485760,  # 10 MB
    backup_count: int = 5,
    use_detailed_format: bool = False,
    json_format: bool = False
) -> logging.Logger:
    """
    Настройка корневого логгера.
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу лога (если None, только консоль)
        max_bytes: Максимальный размер файла до ротации
        backup_count: Количество файлов для ротации
        use_detailed_format: Использовать детальный формат (с именем файла)
        json_format: Использовать JSON-формат (заглушка)
        
    Returns:
        Настроенный корневой логгер
    """
    global _logger
    
    # Получаем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))
    
    # Удаляем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Выбираем формат
    if use_detailed_format:
        format_str = DETAILED_FORMAT
    else:
        format_str = DEFAULT_FORMAT
    
    formatter = logging.Formatter(format_str)
    
    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Обработчик для файла, если указан
    if log_file:
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    _logger = root_logger
    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Получение логгера с заданным именем.
    Если логгер не настроен, настраивается с параметрами по умолчанию.
    
    Args:
        name: Имя логгера (если None, возвращает корневой)
        
    Returns:
        Объект логгера
    """
    global _logger
    
    if _logger is None:
        # Автоматическая настройка с параметрами по умолчанию
        _logger = setup_logging(log_level='INFO')
    
    if name:
        return logging.getLogger(name)
    return _logger


# Создаём логгер по умолчанию для использования в модулях
logger = get_logger('autonomous_ai')


class LoggerMixin:
    """
    Миксин для добавления логгера в классы.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Логгер экземпляра класса"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger