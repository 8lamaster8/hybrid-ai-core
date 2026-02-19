"""
Сервис для создания и управления эмбеддингами с оптимизациями
"""
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.core.logging import logger
from app.infrastructure.cache import cached


class EmbeddingService:
    """
    Сервис эмбеддингов с:
    1. Кэшированием результатов
    2. Батчингом для производительности
    3. Fallback стратегиями
    4. Мониторингом качества
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_enabled: bool = True
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        self.cache_enabled = cache_enabled
        
        # Модель для создания эмбеддингов
        self.model = None
        self.dimension = settings.EMBEDDING_DIMENSION
        
        # Тред пул для CPU операций
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Статистика
        self.stats = {
            "embeddings_created": 0,
            "cache_hits": 0,
            "batch_operations": 0,
            "errors": 0
        }
        
        logger.info(f"Инициализирован EmbeddingService с моделью {self.model_name}")
    
    async def initialize(self):
        """Асинхронная инициализация модели"""
        try:
            # Ленивая загрузка модели
            if self.model is None:
                loop = asyncio.get_event_loop()
                
                # Загружаем модель в отдельном треде
                self.model = await loop.run_in_executor(
                    self.thread_pool,
                    self._load_model
                )
                
                logger.info(
                    "✅ Модель эмбеддингов загружена",
                    extra={
                        "context": {
                            "model": self.model_name,
                            "device": self.device,
                            "dimension": self.dimension
                        }
                    }
                )
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели эмбеддингов: {e}")
            raise
    
    def _load_model(self):
        """Загрузка модели Sentence Transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.debug(f"Загружаем модель: {self.model_name} на устройство: {self.device}")
            
            model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Определяем реальную размерность
            test_embedding = model.encode(["тестовый текст"])
            self.dimension = test_embedding.shape[1]
            
            return model
        
        except ImportError:
            logger.error("Sentence Transformers не установлен!")
            raise
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    @cached(ttl=86400, key_prefix="embedding")  # Кэшируем на 24 часа
    async def embed(self, text: str) -> List[float]:
        """
        Создание эмбеддинга для текста с кэшированием
        
        Args:
            text: Текст для эмбеддинга
        
        Returns:
            Вектор эмбеддинга
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        try:
            # Убеждаемся, что модель загружена
            if self.model is None:
                await self.initialize()
            
            # Создаем эмбеддинг в отдельном треде
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.thread_pool,
                lambda: self.model.encode([text], convert_to_numpy=True)[0]
            )
            
            # Нормализуем вектор (L2 норма)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Обновляем статистику
            self.stats["embeddings_created"] += 1
            
            return embedding.tolist()
        
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Ошибка создания эмбеддинга: {e}")
            
            # Возвращаем нулевой вектор в случае ошибки
            return [0.0] * self.dimension
    
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Пакетное создание эмбеддингов
        
        Args:
            texts: Список текстов
            batch_size: Размер батча
            show_progress: Показывать прогресс
        
        Returns:
            Список эмбеддингов
        """
        if not texts:
            return []
        
        try:
            if self.model is None:
                await self.initialize()
            
            all_embeddings = []
            
            # Обрабатываем батчами
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Фильтруем пустые тексты
                valid_texts = [t for t in batch_texts if t and t.strip()]
                
                if not valid_texts:
                    # Добавляем нулевые векторы для пустых текстов
                    all_embeddings.extend([[0.0] * self.dimension] * len(batch_texts))
                    continue
                
                # Создаем эмбеддинги для батча
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: self.model.encode(
                        valid_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                )
                
                # Восстанавливаем исходный порядок с нулевыми векторами для пустых текстов
                batch_index = 0
                for text in batch_texts:
                    if text and text.strip():
                        embedding = batch_embeddings[batch_index].tolist()
                        batch_index += 1
                    else:
                        embedding = [0.0] * self.dimension
                    
                    all_embeddings.append(embedding)
                
                if show_progress and len(texts) > batch_size:
                    progress = min(i + batch_size, len(texts))
                    logger.debug(f"Эмбеддинги: {progress}/{len(texts)}")
            
            # Обновляем статистику
            self.stats["embeddings_created"] += len(texts)
            self.stats["batch_operations"] += 1
            
            return all_embeddings
        
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Ошибка пакетного создания эмбеддингов: {e}")
            
            # Возвращаем нулевые векторы в случае ошибки
            return [[0.0] * self.dimension for _ in range(len(texts))]
    
    async def similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Вычисление косинусного сходства между эмбеддингами
        
        Args:
            embedding1: Первый эмбеддинг
            embedding2: Второй эмбеддинг
        
        Returns:
            Косинусное сходство (0-1)
        """
        try:
            # Преобразуем в numpy массивы
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Вычисляем косинусное сходство
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            # Ограничиваем от -1 до 1
            similarity = max(-1.0, min(1.0, similarity))
            
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Ошибка вычисления сходства: {e}")
            return 0.0
    
    async def batch_similarity(
        self,
        query_embedding: List[float],
        document_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Пакетное вычисление сходства
        
        Args:
            query_embedding: Эмбеддинг запроса
            document_embeddings: Список эмбеддингов документов
        
        Returns:
            Список значений сходства
        """
        try:
            # Преобразуем в numpy массивы
            query_vec = np.array(query_embedding)
            doc_matrix = np.array(document_embeddings)
            
            # Вычисляем нормы
            query_norm = np.linalg.norm(query_vec)
            doc_norms = np.linalg.norm(doc_matrix, axis=1)
            
            # Вычисляем скалярные произведения
            similarities = np.dot(doc_matrix, query_vec) / (doc_norms * query_norm)
            
            # Заменяем NaN и бесконечности на 0
            similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
            
            return similarities.tolist()
        
        except Exception as e:
            logger.error(f"Ошибка пакетного вычисления сходства: {e}")
            return [0.0] * len(document_embeddings)
    
    async def get_stats(self) -> dict:
        """Получение статистики сервиса"""
        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "cache_enabled": self.cache_enabled,
            "stats": self.stats.copy()
        }
    
    async def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Валидация эмбеддинга
        
        Args:
            embedding: Эмбеддинг для проверки
        
        Returns:
            Валидность эмбеддинга
        """
        try:
            # Проверяем размерность
            if len(embedding) != self.dimension:
                return False
            
            # Проверяем, что не все значения нулевые
            if all(abs(x) < 1e-10 for x in embedding):
                return False
            
            # Проверяем NaN значения
            if any(np.isnan(x) for x in embedding):
                return False
            
            return True
        
        except Exception:
            return False
    
    async def close(self):
        """Корректное закрытие ресурсов"""
        try:
            self.thread_pool.shutdown(wait=True)
            
            # Очищаем кэш модели
            self.model = None
            
            logger.info("Ресурсы EmbeddingService закрыты")
        
        except Exception as e:
            logger.error(f"Ошибка закрытия EmbeddingService: {e}")


# Глобальный экземпляр
embedding_service = EmbeddingService()