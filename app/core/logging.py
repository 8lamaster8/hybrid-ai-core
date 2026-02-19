"""
Продвинутая система логирования для продакшена
"""
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import structlog
from datetime import datetime

from app.core.config import settings, LogLevel


class JSONFormatter(logging.Formatter):
    """Форматтер для логирования в JSON (удобно для ELK стека)"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
            "process": record.processName,
        }
        
        # Добавляем контекст из extra
        if hasattr(record, 'context'):
            log_data["context"] = record.context
        
        # Добавляем информацию об исключении
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Добавляем трейс для ошибок
        if record.levelno >= logging.ERROR:
            log_data["stack_trace"] = self._get_stack_trace()
        
        return json.dumps(log_data, ensure_ascii=False, default=str)
    
    def _get_stack_trace(self) -> str:
        """Получить стек вызовов"""
        import traceback
        return "".join(traceback.format_stack())


class ContextFilter(logging.Filter):
    """Фильтр для добавления контекста в логи"""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Добавляем контекст в запись лога
        if not hasattr(record, 'context'):
            record.context = {}
        record.context.update(self.context)
        return True


def setup_logging(
    name: str = "ai_core",
    log_level: Optional[LogLevel] = None,
    enable_json: Optional[bool] = None,
    enable_structlog: bool = True
) -> logging.Logger:
    """
    Настройка продвинутого логирования
    
    Args:
        name: Имя логгера
        log_level: Уровень логирования
        enable_json: Включить JSON формат
        enable_structlog: Включить structlog для структурированного логирования
    
    Returns:
        Настроенный логгер
    """
    # Определяем параметры из настроек
    log_level = log_level or settings.LOG_LEVEL
    enable_json = enable_json if enable_json is not None else settings.is_production
    
    # Создаем директорию для логов
    logs_dir = settings.LOGS_DIR
    logs_dir.mkdir(exist_ok=True)
    
    # Получаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(log_level.value)
    
    # Очищаем существующие хендлеры
    logger.handlers.clear()
    
    # 1. Консольный хендлер (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level.value)
    
    if enable_json:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    
    # 2. Файловый хендлер с ротацией по размеру
    file_handler = RotatingFileHandler(
        filename=logs_dir / f"{name}.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(console_formatter)
    
    # 3. Хендлер для ошибок (отдельный файл)
    error_handler = RotatingFileHandler(
        filename=logs_dir / f"{name}_error.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(console_formatter)
    
    # 4. Хендлер с ротацией по времени (ежедневно)
    daily_handler = TimedRotatingFileHandler(
        filename=logs_dir / f"{name}_daily.log",
        when='midnight',
        interval=1,
        backupCount=30,  # Храним 30 дней
        encoding='utf-8'
    )
    daily_handler.setLevel(logging.INFO)
    daily_handler.setFormatter(console_formatter)
    
    # Добавляем хендлеры
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(daily_handler)
    
    # Добавляем фильтр с контекстом
    context_filter = ContextFilter({
        "environment": settings.ENVIRONMENT.value,
        "service": name,
        "version": settings.API_VERSION
    })
    logger.addFilter(context_filter)
    
    # Настраиваем structlog для структурированного логирования
    if enable_structlog:
        setup_structlog(logger, enable_json)
    
    # Логируем успешную настройку
    logger.info(
        "Логирование инициализировано",
        extra={
            "context": {
                "log_level": log_level.value,
                "json_format": enable_json,
                "log_directory": str(logs_dir)
            }
        }
    )
    
    return logger


def setup_structlog(base_logger: logging.Logger, enable_json: bool = True):
    """
    Настройка structlog для структурированного логирования
    
    Args:
        base_logger: Базовый логгер
        enable_json: Использовать JSON формат
    """
    try:
        import structlog
        
        # Конфигурация structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
            wrapper_class=structlog.stdlib.BoundLogger,
        )
        
        # Создаем structlog логгер
        struct_logger = structlog.get_logger()
        
        # Биндим контекст
        struct_logger = struct_logger.bind(
            service="ai_core",
            environment=settings.ENVIRONMENT.value
        )
        
        # Сохраняем в настройках для глобального доступа
        import app.core.__init__ as core_module
        core_module.struct_logger = struct_logger
        
    except ImportError:
        base_logger.warning("Structlog не установлен, используется стандартное логирование")


class PerformanceLogger:
    """Логгер для отслеживания производительности"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def time_it(self, operation: str):
        """
        Декоратор для измерения времени выполнения
        
        Args:
            operation: Название операции для логирования
        """
        import functools
        import time
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Логируем начало операции
                self.logger.debug(f"Начало операции: {operation}", extra={
                    "operation": operation,
                    "stage": "start"
                })
                
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    
                    # Логируем успешное завершение
                    self.logger.info(f"Операция завершена: {operation}", extra={
                        "operation": operation,
                        "stage": "complete",
                        "duration_seconds": elapsed,
                        "status": "success"
                    })
                    
                    return result
                
                except Exception as e:
                    elapsed = time.time() - start_time
                    
                    # Логируем ошибку
                    self.logger.error(f"Ошибка в операции {operation}: {e}", extra={
                        "operation": operation,
                        "stage": "error",
                        "duration_seconds": elapsed,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
                    
                    raise
            
            return wrapper
        return decorator
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Логирование метрик производительности"""
        self.logger.info("Метрики производительности", extra={
            "context": {
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
        })


# Глобальный логгер
logger = setup_logging()

# Глобальный логгер производительности
performance_logger = PerformanceLogger(logger)

# Экспорт
__all__ = ["logger", "performance_logger", "setup_logging", "PerformanceLogger"]