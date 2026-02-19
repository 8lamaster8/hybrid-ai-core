"""
Кэш вопросов с поддержкой TTL и стратегиями инвалидации
"""
from typing import Optional, Dict, Any
import time
from collections import OrderedDict
import hashlib

from app.core.logging import logger


class QuestionCache:
    """
    Простой кэш вопросов-ответов с LRU стратегией
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
    
    def _generate_key(self, question: str) -> str:
        """Генерация ключа для вопроса"""
        # Нормализуем вопрос: нижний регистр, убираем лишние пробелы
        normalized = ' '.join(question.lower().strip().split())
        # Создаем хэш для экономии памяти
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Получение кэшированного ответа
        
        Args:
            question: Вопрос пользователя
        
        Returns:
            Кэшированный ответ или None
        """
        key = self._generate_key(question)
        
        if key not in self.cache:
            return None
        
        # Проверяем TTL
        timestamp = self.timestamps.get(key, 0)
        if time.time() - timestamp > self.ttl_seconds:
            # Удаляем просроченный кэш
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        # Обновляем порядок использования (LRU)
        value = self.cache.pop(key)
        self.cache[key] = value
        self.timestamps[key] = time.time()
        
        logger.debug(f"Cache hit for question: {question[:50]}...")
        return value
    
    def set(self, question: str, value: Dict[str, Any]):
        """
        Сохранение ответа в кэш
        
        Args:
            question: Вопрос пользователя
            value: Ответ для кэширования
        """
        key = self._generate_key(question)
        
        # Если превышен размер, удаляем самый старый элемент
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        # Сохраняем значение
        self.cache[key] = value
        self.timestamps[key] = time.time()
        
        logger.debug(f"Cached answer for question: {question[:50]}...")
    
    def clear(self):
        """Очистка кэша"""
        self.cache.clear()
        self.timestamps.clear()
        logger.info("Question cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "oldest_timestamp": min(self.timestamps.values()) if self.timestamps else None,
            "newest_timestamp": max(self.timestamps.values()) if self.timestamps else None
        }