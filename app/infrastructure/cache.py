"""
Продвинутая система кэширования с Redis и fallback на in-memory
"""
import asyncio
import pickle
import json
from typing import Any, Optional, Union, Dict, List
from datetime import datetime, timedelta
from functools import wraps
import hashlib

from app.core.config import settings
from app.core.logging import logger


class Cache:
    """
    Универсальный кэш с поддержкой Redis и in-memory fallback
    
    Стратегии кэширования:
    1. Redis (если доступен) - распределенный кэш
    2. In-memory (fallback) - локальный кэш
    3. No-cache (деградация) - если всё отвалилось
    """
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.redis_available = False
        self._initialized = False
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Асинхронная инициализация кэша"""
        if self._initialized:
            return
        
        async with self._lock:
            try:
                # Пробуем подключиться к Redis
                if settings.REDIS_URL and settings.CACHE_ENABLED:
                    import redis.asyncio as redis
                    
                    self.redis_client = redis.from_url(
                        str(settings.REDIS_URL),
                        encoding="utf-8",
                        decode_responses=False  # Работаем с bytes
                    )
                    
                    # Проверяем соединение
                    await self.redis_client.ping()
                    self.redis_available = True
                    logger.info("✅ Redis кэш подключен")
                
                self._initialized = True
                
            except Exception as e:
                logger.warning(f"Redis недоступен, используем in-memory кэш: {e}")
                self.redis_available = False
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Генерация ключа кэша на основе аргументов
        
        Args:
            prefix: Префикс ключа
            *args: Позиционные аргументы
            **kwargs: Именованные аргументы
        
        Returns:
            Хэшированный ключ
        """
        # Сериализуем аргументы
        data = {
            "args": args,
            "kwargs": {k: v for k, v in kwargs.items() if not k.startswith('_')}
        }
        
        # Преобразуем в строку и хэшируем
        key_str = f"{prefix}:{json.dumps(data, sort_keys=True, default=str)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Получить значение из кэша
        
        Args:
            key: Ключ
        
        Returns:
            Значение или None
        """
        if not settings.CACHE_ENABLED:
            return None
        
        await self.initialize()
        
        try:
            # Пробуем Redis
            if self.redis_available and self.redis_client:
                data = await self.redis_client.get(key)
                if data:
                    return pickle.loads(data)
            
            # Fallback: in-memory кэш
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if entry["expires_at"] > datetime.now():
                    return entry["value"]
                else:
                    del self.memory_cache[key]
        
        except Exception as e:
            logger.warning(f"Ошибка получения из кэша: {e}")
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Сохранить значение в кэш
        
        Args:
            key: Ключ
            value: Значение
            ttl_seconds: Время жизни в секундах
            tags: Теги для групповой инвалидации
        
        Returns:
            Успешность операции
        """
        if not settings.CACHE_ENABLED:
            return False
        
        await self.initialize()
        
        if ttl_seconds is None:
            ttl_seconds = settings.CACHE_TTL_SECONDS
        
        try:
            # Пробуем Redis
            if self.redis_available and self.redis_client:
                # Сериализуем значение
                serialized = pickle.dumps(value)
                
                # Сохраняем с TTL
                await self.redis_client.setex(key, ttl_seconds, serialized)
                
                # Сохраняем теги если есть
                if tags:
                    tag_key = f"tags:{key}"
                    await self.redis_client.setex(tag_key, ttl_seconds, json.dumps(tags))
                
                return True
            
            # Fallback: in-memory кэш
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            self.memory_cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": datetime.now()
            }
            
            # Очищаем старые записи если превысили лимит
            if len(self.memory_cache) > settings.CACHE_MAX_SIZE:
                self._cleanup_memory_cache()
            
            return True
        
        except Exception as e:
            logger.error(f"Ошибка сохранения в кэш: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Удалить значение из кэша"""
        await self.initialize()
        
        try:
            # Удаляем из Redis
            if self.redis_available and self.redis_client:
                await self.redis_client.delete(key)
                await self.redis_client.delete(f"tags:{key}")
            
            # Удаляем из memory
            self.memory_cache.pop(key, None)
            
            return True
        
        except Exception as e:
            logger.warning(f"Ошибка удаления из кэша: {e}")
            return False
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """
        Инвалидация кэша по тегам
        
        Args:
            tags: Список тегов
        
        Returns:
            Количество удаленных записей
        """
        await self.initialize()
        
        if not self.redis_available or not self.redis_client:
            return 0
        
        try:
            deleted_count = 0
            
            # В Redis можно использовать sets для тегов
            for tag in tags:
                tag_key = f"tag:{tag}"
                keys = await self.redis_client.smembers(tag_key)
                
                for key in keys:
                    await self.delete(key)
                    deleted_count += 1
            
            return deleted_count
        
        except Exception as e:
            logger.error(f"Ошибка инвалидации по тегам: {e}")
            return 0
    
    def _cleanup_memory_cache(self):
        """Очистка устаревших записей в памяти"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if entry["expires_at"] <= now
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # Если всё еще много, удаляем старые записи
        if len(self.memory_cache) > settings.CACHE_MAX_SIZE:
            # Сортируем по времени создания и удаляем самые старые
            sorted_keys = sorted(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k]["created_at"]
            )
            
            keys_to_remove = sorted_keys[:len(self.memory_cache) - settings.CACHE_MAX_SIZE // 2]
            for key in keys_to_remove:
                del self.memory_cache[key]
    
    async def clear(self):
        """Полная очистка кэша"""
        await self.initialize()
        
        try:
            # Очищаем Redis
            if self.redis_available and self.redis_client:
                await self.redis_client.flushdb()
            
            # Очищаем память
            self.memory_cache.clear()
            
            logger.info("Кэш полностью очищен")
            
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Получить статистику кэша"""
        await self.initialize()
        
        stats = {
            "enabled": settings.CACHE_ENABLED,
            "redis_available": self.redis_available,
            "memory_cache_size": len(self.memory_cache),
            "max_size": settings.CACHE_MAX_SIZE,
            "default_ttl": settings.CACHE_TTL_SECONDS
        }
        
        if self.redis_available and self.redis_client:
            try:
                info = await self.redis_client.info()
                stats["redis"] = {
                    "used_memory": info.get("used_memory_human", "N/A"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0)
                }
            except:
                stats["redis"] = {"error": "Не удалось получить статистику"}
        
        return stats


def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "cache",
    tags: Optional[List[str]] = None
):
    """
    Декоратор для кэширования результатов функций
    
    Args:
        ttl: Время жизни кэша в секундах
        key_prefix: Префикс для ключей кэша
        tags: Теги для групповой инвалидации
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Создаем экземпляр кэша
            cache = Cache()
            await cache.initialize()
            
            # Генерируем ключ
            key = cache._generate_key(
                f"{key_prefix}:{func.__module__}:{func.__name__}",
                *args,
                **kwargs
            )
            
            # Пробуем получить из кэша
            cached_result = await cache.get(key)
            if cached_result is not None:
                logger.debug(f"Кэш попадание для {func.__name__}")
                return cached_result
            
            # Выполняем функцию
            result = await func(*args, **kwargs)
            
            # Сохраняем в кэш
            await cache.set(
                key=key,
                value=result,
                ttl_seconds=ttl,
                tags=tags
            )
            
            return result
        
        return wrapper
    
    return decorator


# Глобальный экземпляр кэша
cache = Cache()