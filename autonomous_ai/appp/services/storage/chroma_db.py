"""
💾 CHROMADB СЕРВИС - Хранилище эмбеддингов и семантический поиск
"""

import asyncio
import os
import json
import uuid
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import numpy as np

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Document, Embeddings

from appp.core.config import Config
from appp.core.logging import logger


class ChromaDBService:
    """
    Обёртка над ChromaDB для работы с эмбеддингами.
    Поддержка асинхронных операций через to_thread.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None,
        collection_name: str = "knowledge_embeddings",
        distance_metric: str = "cosine",
        max_collection_size: int = 1000000
    ):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.max_collection_size = max_collection_size
        
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        
        # Статистика
        self.stats = {
            'documents_added': 0,
            'embeddings_added': 0,
            'queries_performed': 0,
            'avg_query_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'collection_size': 0,
            'errors': 0,
            'last_compact': None
        }
        
        # Кэш запросов
        self.query_cache = {}
        self.cache_ttl = 3600  # 1 час
        
        os.makedirs(persist_directory, exist_ok=True)
        logger.info(f"💾 ChromaDBService создан (persist: {persist_directory})")
    
    async def initialize(self):
        """Инициализация клиента и коллекции"""
        logger.info("🔄 Инициализация ChromaDB...")
        
        try:
            loop = asyncio.get_event_loop()
            
            def create_client():
                import chromadb
                from chromadb.config import Settings
                
                # Современный API для PersistentClient
                return chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            
            self.client = await loop.run_in_executor(None, create_client)
            
            # Получаем или создаем коллекцию
            def get_or_create_collection():
                # Проверяем, существует ли коллекция
                try:
                    return self.client.get_collection(
                        name=self.collection_name,
                        embedding_function=self._wrap_embedding_function()
                    )
                except Exception:
                    # Создаем новую коллекцию
                    return self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self._wrap_embedding_function(),
                        metadata={"hnsw:space": self.distance_metric}
                    )
            
            self.collection = await loop.run_in_executor(None, get_or_create_collection)
            
            # Получаем размер коллекции
            def count():
                return self.collection.count()
            
            count_val = await loop.run_in_executor(None, count)
            self.stats['collection_size'] = count_val
            
            logger.info(f"✅ ChromaDB инициализирована, коллекция '{self.collection_name}', "
                       f"записей: {count_val}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации ChromaDB: {e}")
            return False
    
    def _wrap_embedding_function(self) -> Optional[EmbeddingFunction]:
        """Обертка функции эмбеддингов для ChromaDB"""
        if self.embedding_function is None:
            return None
        
        class CustomEmbeddingFunction(EmbeddingFunction):
            def __call__(self, texts: List[str]) -> Embeddings:
                return self.embedding_function(texts)
            
            def __init__(self, embedding_function):
                self.embedding_function = embedding_function
        
        return CustomEmbeddingFunction(self.embedding_function)
    
    async def add_document(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Добавление документа в коллекцию.
        
        Args:
            text: Текст документа
            metadata: Метаданные
            doc_id: ID документа (генерируется автоматически, если не указан)
            
        Returns:
            ID добавленного документа
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        
        # Добавляем метки времени
        metadata['added_at'] = datetime.now().isoformat()
        metadata['text_length'] = len(text)
        
        try:
            loop = asyncio.get_event_loop()
            
            def add():
                self.collection.add(
                    documents=[text],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
            
            await loop.run_in_executor(None, add)
            
            self.stats['documents_added'] += 1
            self.stats['collection_size'] = await self._update_count()
            
            logger.debug(f"✅ Документ добавлен: {doc_id} ({len(text)} символов)")
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления документа {doc_id}: {e}")
            self.stats['errors'] += 1
            raise
    
    async def add_embedding(
        self,
        embedding: List[float],
        text: str,
        metadata: Dict[str, Any] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Добавление готового эмбеддинга (если эмбеддинг уже вычислен).
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        
        metadata['added_at'] = datetime.now().isoformat()
        metadata['is_precomputed'] = True
        
        try:
            loop = asyncio.get_event_loop()
            
            def add():
                self.collection.add(
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
            
            await loop.run_in_executor(None, add)
            
            self.stats['embeddings_added'] += 1
            self.stats['collection_size'] = await self._update_count()
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Ошибка добавления эмбеддинга {doc_id}: {e}")
            self.stats['errors'] += 1
            raise
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Семантический поиск по запросу.
        
        Args:
            query: Текстовый запрос
            k: Количество результатов
            threshold: Минимальный порог схожести (0-1)
            filter_criteria: Фильтрация по метаданным
            
        Returns:
            Список результатов с полями: id, document, metadata, distance, score
        """
        start_time = datetime.now()
        
        # Проверка кэша
        cache_key = self._get_cache_key(query, k, threshold, filter_criteria)
        if cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).total_seconds() < self.cache_ttl:
                self.stats['cache_hits'] += 1
                return cache_entry['results']
        
        self.stats['cache_misses'] += 1
        
        try:
            loop = asyncio.get_event_loop()
            
            def search():
                return self.collection.query(
                    query_texts=[query],
                    n_results=k,
                    where=filter_criteria,
                    include=["documents", "metadatas", "distances"]
                )
            
            result = await loop.run_in_executor(None, search)
            
            # Форматирование результатов
            formatted_results = []
            if result['ids'] and result['ids'][0]:
                for i in range(len(result['ids'][0])):
                    # Chroma возвращает расстояние (меньше = ближе)
                    distance = result['distances'][0][i] if result['distances'] else 0
                    
                    # Конвертируем расстояние в score (для косинуса: 1 - расстояние)
                    score = 1 - distance if self.distance_metric == 'cosine' else distance
                    
                    if score >= threshold:
                        formatted_results.append({
                            'id': result['ids'][0][i],
                            'text': result['documents'][0][i] if result['documents'] else '',
                            'metadata': result['metadatas'][0][i] if result['metadatas'] else {},
                            'distance': distance,
                            'score': score
                        })
            
            # Сохраняем в кэш
            self.query_cache[cache_key] = {
                'results': formatted_results,
                'timestamp': datetime.now()
            }
            
            # Обновляем статистику
            self.stats['queries_performed'] += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            n = self.stats['queries_performed']
            self.stats['avg_query_time'] = (
                (self.stats['avg_query_time'] * (n - 1) + elapsed) / n
            )
            
            logger.debug(f"🔍 Поиск '{query[:50]}...' -> {len(formatted_results)} результатов")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            self.stats['errors'] += 1
            return []
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Получение документа по ID"""
        try:
            loop = asyncio.get_event_loop()
            
            def get():
                return self.collection.get(ids=[doc_id])
            
            result = await loop.run_in_executor(None, get)
            
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'text': result['documents'][0] if result['documents'] else '',
                    'metadata': result['metadatas'][0] if result['metadatas'] else {}
                }
            return None
            
        except Exception as e:
            logger.error(f"Ошибка получения документа {doc_id}: {e}")
            return None
    
    async def delete_document(self, doc_id: str) -> bool:
        """Удаление документа"""
        try:
            loop = asyncio.get_event_loop()
            
            def delete():
                self.collection.delete(ids=[doc_id])
            
            await loop.run_in_executor(None, delete)
            
            self.stats['collection_size'] = await self._update_count()
            logger.debug(f"🗑️ Документ удален: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления документа {doc_id}: {e}")
            return False
    
    async def update_document(
        self,
        doc_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Обновление документа"""
        try:
            loop = asyncio.get_event_loop()
            
            # Получаем текущие данные
            current = await self.get_document(doc_id)
            if not current:
                return False
            
            new_text = text if text is not None else current['text']
            new_metadata = {**current['metadata'], **(metadata or {})}
            new_metadata['updated_at'] = datetime.now().isoformat()
            
            def update():
                self.collection.update(
                    ids=[doc_id],
                    documents=[new_text] if text else None,
                    metadatas=[new_metadata]
                )
            
            await loop.run_in_executor(None, update)
            
            logger.debug(f"📝 Документ обновлен: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления документа {doc_id}: {e}")
            return False
    
    async def count(self) -> int:
        """Количество документов в коллекции"""
        return await self._update_count()
    
    async def _update_count(self) -> int:
        """Обновление и получение количества записей"""
        try:
            loop = asyncio.get_event_loop()
            
            def count():
                return self.collection.count()
            
            count_val = await loop.run_in_executor(None, count)
            self.stats['collection_size'] = count_val
            return count_val
            
        except Exception as e:
            logger.error(f"Ошибка получения количества: {e}")
            return self.stats['collection_size']
    
    async def optimize(self):
        """Оптимизация коллекции"""
        logger.info("🔄 Оптимизация ChromaDB...")
        
        try:
            if self.client is None:
                logger.warning("ChromaDB клиент не инициализирован")
                return
            
            loop = asyncio.get_event_loop()
            
            def optimize_task():
                # В PersistentClient нет прямого метода optimize, 
                # но можно вызвать heartbeat для проверки соединения
                self.client.heartbeat()
                # При необходимости можно выполнить другие действия
            
            await loop.run_in_executor(None, optimize_task)
            
            self.stats['last_compact'] = datetime.now().isoformat()
            logger.info("✅ Оптимизация ChromaDB завершена")
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации: {e}")
    
    async def clear(self):
        """Очистка коллекции"""
        try:
            loop = asyncio.get_event_loop()
            
            def delete_all():
                # Удаляем все документы
                all_ids = self.collection.get()['ids']
                if all_ids:
                    self.collection.delete(ids=all_ids)
            
            await loop.run_in_executor(None, delete_all)
            
            self.query_cache.clear()
            self.stats['collection_size'] = 0
            
            logger.info("🧹 Коллекция ChromaDB очищена")
            
        except Exception as e:
            logger.error(f"Ошибка очистки коллекции: {e}")
    
    def _get_cache_key(
        self,
        query: str,
        k: int,
        threshold: float,
        filter_criteria: Optional[Dict]
    ) -> str:
        """Генерация ключа кэша"""
        import hashlib
        key_parts = [
            query,
            str(k),
            str(threshold),
            json.dumps(filter_criteria, sort_keys=True) if filter_criteria else ''
        ]
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_detailed_stats(self) -> Dict[str, Any]:
        """Подробная статистика"""
        try:
            # Получаем размер базы данных
            db_size = 0
            if os.path.exists(self.persist_directory):
                for root, dirs, files in os.walk(self.persist_directory):
                    for file in files:
                        db_size += os.path.getsize(os.path.join(root, file))
            
            cache_size_mb = sum(
                len(json.dumps(v['results']).encode()) 
                for v in self.query_cache.values()
            ) / 1024 / 1024
            
            return {
                **self.stats,
                'database_size_mb': db_size / 1024 / 1024,
                'cache_size': len(self.query_cache),
                'cache_size_mb': cache_size_mb,
                'collection_name': self.collection_name,
                'distance_metric': self.distance_metric,
                'persist_directory': self.persist_directory,
                'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
                'avg_document_length': await self._avg_document_length()
            }
        except Exception as e:
            logger.error(f"Ошибка получения детальной статистики: {e}")
            return self.stats
    
    async def _avg_document_length(self) -> float:
        """Средняя длина документов"""
        try:
            loop = asyncio.get_event_loop()
            
            def get_all():
                return self.collection.get()
            
            result = await loop.run_in_executor(None, get_all)
            
            if result['documents']:
                total_len = sum(len(doc) for doc in result['documents'])
                return total_len / len(result['documents'])
            return 0.0
            
        except:
            return 0.0
    
    async def persist(self):
        """Принудительное сохранение"""
        await self.optimize()
    
    async def close(self):
        """Завершение работы"""
        logger.info("🛑 Завершение работы ChromaDBService...")
        await self.persist()
        logger.info("✅ ChromaDBService завершил работу")
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья"""
        try:
            # Проверяем доступность коллекции
            count = await self._update_count()
            return {
                'healthy': True,
                'collection': self.collection_name,
                'documents': count,
                'message': 'ChromaDB is operational',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Метрики для мониторинга"""
        return await self.get_detailed_stats()