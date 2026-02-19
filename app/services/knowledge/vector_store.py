"""
Векторное хранилище на основе ChromaDB с оптимизациями для продакшена
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json  # Добавляем импорт

from app.core.config import settings
from app.core.logging import logger
from app.services.knowledge.base import KnowledgeBase, KnowledgeChunk, SearchType
from app.services.knowledge.embedding_service import EmbeddingService


class ChromaVectorStore(KnowledgeBase):
    """
    Векторное хранилище на ChromaDB с оптимизациями:
    
    1. Асинхронные операции
    2. Батчинг для больших объемов
    3. Кэширование эмбеддингов
    4. Гибридный поиск
    5. Мониторинг производительности
    """
    
    def __init__(
        self,
        collection_name: str = None,
        host: str = None,
        port: int = None,
        embedding_service: Optional[EmbeddingService] = None,
        persist_directory: str = "./chroma_data"  # ДОБАВЛЕНО: путь для сохранения
    ):
        self.collection_name = collection_name or settings.CHROMA_COLLECTION
        self.host = host or settings.CHROMA_HOST
        self.port = port or settings.CHROMA_PORT
        self.persist_directory = persist_directory
        
        # Сервис эмбеддингов
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Клиент ChromaDB
        self.client = None
        self.collection = None
        
        # Тред пул для блокирующих операций
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Кэш метаданных коллекции
        self._collection_info_cache = None
        self._cache_ttl = 60  # секунд
        self._last_cache_update = None
        
        # Статистика
        self.stats = {
            "searches": 0,
            "additions": 0,
            "deletions": 0,
            "errors": 0
        }
    
    async def initialize(self):
        """Асинхронная инициализация хранилища"""
        try:
            # Создаем клиент ChromaDB в памяти (для теста)
            # Если нужно постоянное хранилище, используем PersistentClient
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            # Получаем или создаем коллекцию
            await self._get_or_create_collection()
            
            # Инициализируем сервис эмбеддингов
            await self.embedding_service.initialize()
            
            logger.info(
                "✅ Векторное хранилище ChromaDB инициализировано (в памяти)",
                extra={
                    "context": {
                        "collection": self.collection_name,
                        "persist_directory": self.persist_directory,
                        "embedding_model": self.embedding_service.model_name
                    }
                }
            )
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации ChromaDB: {e}")
            
            # Пробуем fallback: in-memory клиент
            try:
                logger.info("Пробуем инициализировать in-memory ChromaDB...")
                self.client = chromadb.EphemeralClient()
                await self._get_or_create_collection()
                await self.embedding_service.initialize()
                
                logger.info("✅ In-memory ChromaDB инициализирован")
                return True
            except Exception as e2:
                logger.error(f"❌ Ошибка инициализации in-memory ChromaDB: {e2}")
                raise e
    
    async def _get_or_create_collection(self):
        """Получение или создание коллекции"""
        try:
            # Пробуем получить существующую коллекцию
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.debug(f"Коллекция {self.collection_name} найдена")
        
        except Exception as e:
            # Создаем новую коллекцию
            logger.info(f"Создаем новую коллекцию: {self.collection_name}")
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "База знаний AI ассистента",
                    "created_at": datetime.now().isoformat(),
                    "embedding_model": self.embedding_service.model_name,
                    "embedding_dimension": self.embedding_service.dimension
                }
            )
    
    async def add(self, chunks: List[KnowledgeChunk]) -> List[str]:
        """
        Добавление чанков в хранилище
        
        Args:
            chunks: Список чанков
        
        Returns:
            Список ID
        """
        if not chunks:
            return []
        
        try:
            # Подготавливаем данные
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for chunk in chunks:
                ids.append(chunk.id)
                documents.append(chunk.content)
                
                # Подготавливаем метаданные для ChromaDB
                metadata = self._prepare_metadata(chunk.metadata)
                metadata["chunk_id"] = chunk.id
                metadatas.append(metadata)
                
                # Используем существующий эмбеддинг или создаем новый
                if chunk.embedding:
                    embeddings.append(chunk.embedding)
                else:
                    # Создаем эмбеддинг асинхронно
                    embedding = await self.embedding_service.embed(chunk.content)
                    embeddings.append(embedding)
            
            # Добавляем в ChromaDB в отдельном треде
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.thread_pool,
                lambda: self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings if embeddings else None
                )
            )
            
            # Обновляем статистику
            self.stats["additions"] += len(chunks)
            self._collection_info_cache = None  # Инвалидируем кэш
            
            logger.debug(f"Добавлено {len(chunks)} чанков в коллекцию {self.collection_name}")
            
            return ids
        
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Ошибка добавления чанков: {e}")
            raise
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        search_type: SearchType = SearchType.SIMILARITY,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.1  # Уменьшаем порог для общих вопросов
    ) -> List[KnowledgeChunk]:
        """
        Поиск в векторном хранилище
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            search_type: Тип поиска
            filters: Фильтры
            min_score: Минимальный score
        
        Returns:
            Список чанков
        """
        try:
            # Создаем эмбеддинг запроса
            query_embedding = await self.embedding_service.embed(query)
            
            # Выполняем поиск в отдельном треде
            loop = asyncio.get_event_loop()
            
            if search_type == SearchType.KEYWORD:
                # Ключевой поиск (без эмбеддингов)
                results = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: self.collection.query(
                        query_texts=[query],
                        n_results=top_k,
                        where=filters
                    )
                )
            else:
                # Поиск по эмбеддингам
                results = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k,
                        where=filters
                    )
                )
            
            # Преобразуем результаты в KnowledgeChunk
            chunks = []
            
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    document = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    
                    # Преобразуем расстояние в score (1 - normalized distance)
                    score = 1.0 / (1.0 + distance)
                    
                    if score >= min_score:
                        chunk = KnowledgeChunk(
                            id=metadata.get("chunk_id", f"result_{i}"),
                            content=document,
                            metadata=metadata,
                            score=score
                        )
                        chunks.append(chunk)
            
            # Обновляем статистику
            self.stats["searches"] += 1
            
            logger.debug(
                f"Поиск '{query[:50]}...' вернул {len(chunks)} результатов",
                extra={
                    "context": {
                        "search_type": search_type.value,
                        "top_k": top_k,
                        "min_score": min_score,
                        "found": len(chunks)
                    }
                }
            )
            
            return chunks
        
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Ошибка поиска: {e}")
            return []
    
    async def delete(self, chunk_ids: List[str]) -> int:
        """Удаление чанков по ID"""
        try:
            loop = asyncio.get_event_loop()
            
            await loop.run_in_executor(
                self.thread_pool,
                lambda: self.collection.delete(ids=chunk_ids)
            )
            
            deleted_count = len(chunk_ids)
            self.stats["deletions"] += deleted_count
            self._collection_info_cache = None
            
            logger.info(f"Удалено {deleted_count} чанков")
            
            return deleted_count
        
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Ошибка удаления: {e}")
            return 0
    
    async def update(self, chunk: KnowledgeChunk) -> bool:
        """Обновление чанка"""
        try:
            # В ChromaDB обновление = удаление + добавление
            await self.delete([chunk.id])
            await self.add([chunk])
            
            logger.debug(f"Чанк {chunk.id} обновлен")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка обновления чанка: {e}")
            return False
    
    async def get_info(self) -> Dict[str, Any]:
        """Получение информации о хранилище"""
        try:
            # Используем кэш если он актуален
            if (self._collection_info_cache and 
                self._last_cache_update and
                (datetime.now() - self._last_cache_update).total_seconds() < self._cache_ttl):
                return self._collection_info_cache
            
            loop = asyncio.get_event_loop()
            
            # Получаем информацию о коллекции
            count = await loop.run_in_executor(
                self.thread_pool,
                self.collection.count
            )
            
            # Метаданные коллекции
            metadata = self.collection.metadata or {}
            
            info = {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "embedding_model": metadata.get("embedding_model", "unknown"),
                "embedding_dimension": metadata.get("embedding_dimension", 0),
                "created_at": metadata.get("created_at"),
                "stats": self.stats.copy(),
                "persist_directory": self.persist_directory
            }
            
            # Сохраняем в кэш
            self._collection_info_cache = info
            self._last_cache_update = datetime.now()
            
            return info
        
        except Exception as e:
            logger.error(f"Ошибка получения информации: {e}")
            return {
                "collection_name": self.collection_name,
                "error": str(e)
            }
    
    async def clear(self) -> bool:
        """Полная очистка коллекции"""
        try:
            loop = asyncio.get_event_loop()
            
            await loop.run_in_executor(
                self.thread_pool,
                self.client.delete_collection,
                self.collection_name
            )
            
            # Создаем новую пустую коллекцию
            await self._get_or_create_collection()
            
            # Сбрасываем статистику
            self.stats = {k: 0 for k in self.stats}
            self._collection_info_cache = None
            
            logger.info(f"Коллекция {self.collection_name} полностью очищена")
            
            return True
        
        except Exception as e:
            logger.error(f"Ошибка очистки коллекции: {e}")
            return False
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подготовка метаданных для ChromaDB
        
        Args:
            metadata: Исходные метаданные
        
        Returns:
            Подготовленные метаданные
        """
        # ChromaDB требует, чтобы значения были строками, числами или булевыми
        prepared = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            
            if isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif isinstance(value, (list, tuple)):
                # Преобразуем списки в строки JSON
                prepared[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                # Словари тоже в JSON
                prepared[key] = json.dumps(value, ensure_ascii=False)
            else:
                # Все остальное в строку
                prepared[key] = str(value)
        
        return prepared
    

        # В vector_store.py добавляем метод:
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья векторного хранилища"""
        try:
            count = await self.collection.count()
            return {
                "status": "healthy",
                "collection_count": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_service.model_name
            }
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
        }




    async def close(self):
        """Корректное закрытие ресурсов"""
        try:
            self.thread_pool.shutdown(wait=True)
            logger.info("Ресурсы векторного хранилища закрыты")
        except Exception as e:
            logger.error(f"Ошибка закрытия ресурсов: {e}")


# Глобальный экземпляр
vector_store = ChromaVectorStore()