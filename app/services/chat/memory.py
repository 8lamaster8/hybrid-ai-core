"""
Менеджер памяти для управления сессиями и историей диалогов
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import asyncio

from app.core.config import settings
from app.core.logging import logger
from app.infrastructure.cache import Cache


class MemoryManager:
    """
    Менеджер памяти для хранения и управления сессиями диалогов
    с поддержкой контекста и истории
    """
    
    def __init__(self, session_ttl: int = 86400):  # 24 часа по умолчанию
        self.cache = Cache()
        self.session_ttl = session_ttl
        self.max_history_length = 50  # Максимальное количество сообщений в истории
        self.context_window = 10  # Количество сообщений для контекста
        
    async def initialize(self):
        """Инициализация менеджера памяти"""
        await self.cache.initialize()
        logger.info("Memory Manager initialized")
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Создание новой сессии
        
        Args:
            user_id: ID пользователя
            metadata: Дополнительные метаданные
        
        Returns:
            ID созданной сессии
        """
        import uuid
        session_id = str(uuid.uuid4())
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0,
            "metadata": metadata or {},
            "history": []
        }
        
        # Сохраняем сессию в кэш
        cache_key = f"session:{session_id}"
        await self.cache.set(
            key=cache_key,
            value=session_data,
            ttl_seconds=self.session_ttl
        )
        
        logger.info(f"Создана новая сессия: {session_id}", extra={
            "context": {"user_id": user_id, "metadata": metadata}
        })
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение данных сессии
        
        Args:
            session_id: ID сессии
        
        Returns:
            Данные сессии или None
        """
        cache_key = f"session:{session_id}"
        session_data = await self.cache.get(cache_key)
        
        if session_data:
            # Обновляем время последнего обращения
            session_data["updated_at"] = datetime.now().isoformat()
            await self.cache.set(
                key=cache_key,
                value=session_data,
                ttl_seconds=self.session_ttl
            )
        
        return session_data
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Добавление сообщения в историю сессии
        
        Args:
            session_id: ID сессии
            role: Роль (user/assistant/system)
            content: Содержимое сообщения
            metadata: Дополнительные метаданные
        
        Returns:
            Успешность операции
        """
        try:
            session_data = await self.get_session(session_id)
            if not session_data:
                # Создаем новую сессию если не найдена
                session_id = await self.create_session()
                session_data = await self.get_session(session_id)
            
            message = {
                "id": len(session_data["history"]) + 1,
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Добавляем сообщение в историю
            session_data["history"].append(message)
            session_data["message_count"] += 1
            session_data["updated_at"] = datetime.now().isoformat()
            
            # Ограничиваем длину истории
            if len(session_data["history"]) > self.max_history_length:
                session_data["history"] = session_data["history"][-self.max_history_length:]
            
            # Сохраняем обновленную сессию
            cache_key = f"session:{session_id}"
            await self.cache.set(
                key=cache_key,
                value=session_data,
                ttl_seconds=self.session_ttl
            )
            
            logger.debug(f"Добавлено сообщение в сессию {session_id}: {role}: {content[:50]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления сообщения: {e}")
            return False
    
    async def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        include_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Получение истории диалога
        
        Args:
            session_id: ID сессии
            limit: Ограничение количества сообщений
            include_context: Включать ли только контекстное окно
        
        Returns:
            История сообщений
        """
        session_data = await self.get_session(session_id)
        if not session_data:
            return []
        
        history = session_data["history"]
        
        if include_context and len(history) > self.context_window:
            # Возвращаем только последние сообщения для контекста
            history = history[-self.context_window:]
        
        if limit and len(history) > limit:
            history = history[-limit:]
        
        return history
    
    async def get_conversation_context(
        self,
        session_id: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Получение контекста диалога в текстовом формате
        
        Args:
            session_id: ID сессии
            max_tokens: Максимальная длина контекста
        
        Returns:
            Текстовый контекст диалога
        """
        history = await self.get_history(session_id, include_context=True)
        
        context_parts = []
        total_length = 0
        
        for msg in history:
            role_display = {
                "user": "Пользователь",
                "assistant": "Ассистент",
                "system": "Система"
            }.get(msg["role"], msg["role"])
            
            message_text = f"{role_display}: {msg['content']}"
            
            if total_length + len(message_text) > max_tokens:
                break
            
            context_parts.append(message_text)
            total_length += len(message_text)
        
        return "\n".join(context_parts)
    
    async def update_session_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Обновление метаданных сессии
        
        Args:
            session_id: ID сессии
            metadata: Новые метаданные
        
        Returns:
            Успешность операции
        """
        try:
            session_data = await self.get_session(session_id)
            if not session_data:
                return False
            
            # Обновляем метаданные
            session_data["metadata"].update(metadata)
            session_data["updated_at"] = datetime.now().isoformat()
            
            # Сохраняем
            cache_key = f"session:{session_id}"
            await self.cache.set(
                key=cache_key,
                value=session_data,
                ttl_seconds=self.session_ttl
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления метаданных: {e}")
            return False
    
    async def clear_session(self, session_id: str) -> bool:
        """
        Очистка сессии (удаление истории)
        
        Args:
            session_id: ID сессии
        
        Returns:
            Успешность операции
        """
        try:
            session_data = await self.get_session(session_id)
            if not session_data:
                return False
            
            # Очищаем историю
            session_data["history"] = []
            session_data["message_count"] = 0
            session_data["updated_at"] = datetime.now().isoformat()
            
            # Сохраняем
            cache_key = f"session:{session_id}"
            await self.cache.set(
                key=cache_key,
                value=session_data,
                ttl_seconds=self.session_ttl
            )
            
            logger.info(f"Сессия очищена: {session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка очистки сессии: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Полное удаление сессии
        
        Args:
            session_id: ID сессии
        
        Returns:
            Успешность операции
        """
        try:
            cache_key = f"session:{session_id}"
            await self.cache.delete(cache_key)
            
            logger.info(f"Сессия удалена: {session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления сессии: {e}")
            return False
    
    async def get_active_sessions(
        self,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Получение активных сессий (за последние N часов)
        
        Args:
            hours: Количество часов для поиска активных сессий
        
        Returns:
            Список активных сессий
        """
        # В Redis можно использовать keys или scan для поиска
        # Здесь упрощенная версия
        return []  # В реальной системе здесь поиск по паттерну
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Очистка просроченных сессий
        
        Returns:
            Количество удаленных сессий
        """
        # Redis автоматически удаляет ключи с TTL
        # Здесь можно добавить дополнительную логику
        return 0
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Получение статистики по сессиям
        
        Returns:
            Статистика
        """
        return {
            "session_ttl": self.session_ttl,
            "max_history_length": self.max_history_length,
            "context_window": self.context_window
        }
    
    async def health_check(self) -> bool:
        """Проверка здоровья менеджера памяти"""
        try:
            # Тестовая операция
            test_session = await self.create_session(metadata={"test": True})
            await self.add_message(test_session, "system", "Health check message")
            await self.delete_session(test_session)
            
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def close(self):
        """Корректное завершение работы"""
        pass


# Глобальный экземпляр
memory_manager = MemoryManager()