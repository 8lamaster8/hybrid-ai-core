"""
Исправленный файл database.py
"""
import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
import json
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, func, ForeignKey

from app.core.config import settings
from app.core.logging import logger

# Базовый класс для моделей
Base = declarative_base()


class DatabaseManager:
    """Менеджер базы данных с поддержкой async/await"""
    
    def __init__(self):
        self.engine = None
        self.async_session_factory = None
        self._initialized = False
    
    async def initialize(self):
        """Асинхронная инициализация базы данных"""
        if self._initialized:
            return
        
        try:
            # Создаем async engine
            database_url = settings.get_database_url(async_=True)
            
            self.engine = create_async_engine(
                database_url,
                echo=settings.is_development,  # Логирование SQL в разработке
                pool_size=20,
                max_overflow=40,
                pool_pre_ping=True,  # Проверка соединений перед использованием
                pool_recycle=3600,  # Пересоздание соединений каждый час
            )
            
            # Создаем фабрику сессий
            self.async_session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False
            )
            
            self._initialized = True
            
            logger.info("✅ База данных инициализирована")
            
            # Создаем таблицы при первом запуске
            await self.create_tables()
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации базы данных: {e}")
            raise
    
    async def create_tables(self):
        """Создание таблиц в базе данных"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ Таблицы базы данных созданы")
        except Exception as e:
            logger.error(f"❌ Ошибка создания таблиц: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Контекстный менеджер для получения сессии
        
        Usage:
            async with db.get_session() as session:
                # работа с сессией
        """
        if not self._initialized:
            await self.initialize()
        
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def health_check(self) -> bool:
        """Проверка здоровья базы данных"""
        try:
            async with self.get_session() as session:
                # Простой запрос для проверки
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def close(self):
        """Закрытие соединений с базой данных"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Соединения с базой данных закрыты")


# Модели данных
class Conversation(Base):
    """Модель диалога"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)  # Убрали unique, чтобы можно было несколько на сессию
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Статусы
    is_active = Column(Integer, default=1)
    message_count = Column(Integer, default=0)
    
    # Связи
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """Модель сообщения"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), index=True)
    role = Column(String)  # user, assistant, system
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Метаданные
    meta_data = Column(JSON, default=dict)
    
    # Связи
    conversation = relationship("Conversation", back_populates="messages")


class KnowledgeDocument(Base):
    """Модель документа знаний"""
    __tablename__ = "knowledge_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)
    source = Column(String)  # file, web, api, manual
    source_url = Column(String, nullable=True)
    
    # Векторное представление
    embedding = Column(JSON, nullable=True)
    embedding_model = Column(String, nullable=True)
    
    # Метаданные
    meta_data = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    
    # Статистика
    chunk_count = Column(Integer, default=1)
    word_count = Column(Integer, default=0)
    
    # Временные метки
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    indexed_at = Column(DateTime, nullable=True)


class Feedback(Base):
    """Модель обратной связи"""
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), index=True)  # ✅ ForeignKey добавлен
    message_id = Column(Integer, index=True)
    
    # Рейтинг
    rating = Column(Integer)  # 1-5
    helpful = Column(Integer, nullable=True)  # 0 или 1
    
    # Дополнительная информация
    comment = Column(Text, nullable=True)
    suggested_improvement = Column(Text, nullable=True)
    
    # Метаданные
    meta_data = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Связи
    conversation = relationship("Conversation", back_populates="feedbacks")


# Глобальный экземпляр менеджера БД
db_manager = DatabaseManager()

# Экспорт
__all__ = [
    "db_manager",
    "Base",
    "Conversation",
    "Message",
    "KnowledgeDocument",
    "Feedback",
    "DatabaseManager"
]