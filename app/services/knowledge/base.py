"""
Базовый класс и интерфейсы для работы с векторными базами знаний
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from app.core.logging import logger


class SearchType(str, Enum):
    """Типы поиска"""
    SIMILARITY = "similarity"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"


@dataclass
class KnowledgeChunk:
    """Фрагмент знания"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None  # Релевантность при поиске
    
    def __post_init__(self):
        # Автоматически добавляем временные метки
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.utcnow().isoformat()
        
        # Вычисляем длину контента
        if "content_length" not in self.metadata:
            self.metadata["content_length"] = len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "score": self.score
        }


class KnowledgeBase(ABC):
    """
    Абстрактный класс базы знаний.
    Определяет интерфейс для работы с векторными базами данных.
    """
    
    @abstractmethod
    async def initialize(self):
        """Инициализация базы знаний"""
        pass
    
    @abstractmethod
    async def add(self, chunks: List[KnowledgeChunk]) -> List[str]:
        """
        Добавление фрагментов в базу знаний
        
        Args:
            chunks: Список фрагментов знаний
        
        Returns:
            Список ID добавленных фрагментов
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        search_type: SearchType = SearchType.SIMILARITY,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.3
    ) -> List[KnowledgeChunk]:
        """
        Поиск по базе знаний
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            search_type: Тип поиска
            filters: Фильтры по метаданным
            min_score: Минимальный порог релевантности
        
        Returns:
            Список найденных фрагментов
        """
        pass
    
    @abstractmethod
    async def delete(self, chunk_ids: List[str]) -> int:
        """
        Удаление фрагментов
        
        Args:
            chunk_ids: Список ID для удаления
        
        Returns:
            Количество удаленных фрагментов
        """
        pass
    
    @abstractmethod
    async def update(self, chunk: KnowledgeChunk) -> bool:
        """
        Обновление фрагмента
        
        Args:
            chunk: Фрагмент с обновленными данными
        
        Returns:
            Успешность операции
        """
        pass
    
    @abstractmethod
    async def get_info(self) -> Dict[str, Any]:
        """Получение информации о базе знаний"""
        pass
    
    async def batch_add(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Пакетное добавление текстов
        
        Args:
            texts: Список текстов
            metadatas: Список метаданных
            batch_size: Размер пакета
        
        Returns:
            Список ID добавленных фрагментов
        """
        from app.services.knowledge.embedding_service import EmbeddingService
        
        embedding_service = EmbeddingService()
        all_ids = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size] if metadatas else None
            
            # Создаем эмбеддинги
            embeddings = await embedding_service.embed_batch(batch_texts)
            
            # Создаем чанки
            chunks = []
            for j, text in enumerate(batch_texts):
                metadata = batch_metadatas[j] if batch_metadatas else {}
                chunk = KnowledgeChunk(
                    content=text,
                    embedding=embeddings[j] if j < len(embeddings) else None,
                    metadata=metadata
                )
                chunks.append(chunk)
            
            # Добавляем в базу
            ids = await self.add(chunks)
            all_ids.extend(ids)
            
            logger.info(f"Добавлено {len(ids)} чанков, всего: {len(all_ids)}")
        
        return all_ids
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[KnowledgeChunk]:
        """
        Семантический поиск с гибридным подходом
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            filters: Фильтры
        
        Returns:
            Результаты поиска
        """
        # Сначала семантический поиск
        semantic_results = await self.search(
            query=query,
            top_k=top_k * 2,  # Берем больше для последующей фильтрации
            search_type=SearchType.SEMANTIC,
            filters=filters
        )
        
        # Затем keyword поиск для recall
        keyword_results = await self.search(
            query=query,
            top_k=top_k,
            search_type=SearchType.KEYWORD,
            filters=filters
        )
        
        # Объединяем и дедуплицируем
        all_results = {}
        for result in semantic_results + keyword_results:
            if result.id not in all_results:
                all_results[result.id] = result
            else:
                # Объединяем скоринги
                existing = all_results[result.id]
                if result.score and existing.score:
                    existing.score = max(existing.score, result.score)
        
        # Сортируем по score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.score or 0,
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    async def clear(self) -> bool:
        """Полная очистка базы знаний"""
        try:
            # Получаем все ID
            info = await self.get_info()
            if "total_chunks" in info and info["total_chunks"] == 0:
                return True
            
            # В зависимости от реализации
            logger.warning("Метод clear() не реализован, требуется переопределение")
            return False
        
        except Exception as e:
            logger.error(f"Ошибка очистки базы знаний: {e}")
            return False