"""
🧠 BGE-M3 Эмбеддинги - Семантическое представление текста
Модель от BAAI, локальное исполнение, поддержка кэширования
"""

import asyncio
import logging
import hashlib
import json
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import numpy as np

from sentence_transformers import SentenceTransformer  # <-- ИМПОРТ НА ВЕРХНЕМ УРОВНЕ

from appp.core.logging import logger

logger = logging.getLogger(__name__)


class BGE_M3_Embedder:
    """
    Обёртка над BAAI/bge-m3 для генерации эмбеддингов.
    Поддерживает батчинг, кэширование, нормализацию.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        model_path: Optional[str] = None,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        cache_dir: str = "./data/cache/embeddings",
        max_cache_size: int = 10000,
        batch_size: int = 32,
        embedding_dimension: int = 768
    ):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.device = device
        self.normalize = normalize_embeddings
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.batch_size = batch_size
        self.embedding_dimension = embedding_dimension

        self.model: Optional[SentenceTransformer] = None
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_metadata: Dict[str, Dict] = {}

        self.stats = {
            'embedding_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_processed': 0,
            'total_embeddings': 0,
            'avg_embedding_time': 0.0,
            'errors': 0
        }

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"🧠 BGE-M3 Embedder создан (device: {device})")

    async def initialize(self):
        """Загрузка модели и кэша"""
        logger.info("🔄 Загрузка модели эмбеддингов...")

        # Автоматическая загрузка модели, если её нет
        if not os.path.exists(self.model_path):
            logger.warning(f"⚠️ Модель не найдена по пути {self.model_path}, пробуем загрузить из хаба...")
            try:
                # Импорт уже есть вверху, используем SentenceTransformer
                model = SentenceTransformer(self.model_name)
                model.save(self.model_path)
                logger.info(f"✅ Модель загружена и сохранена в {self.model_path}")
            except Exception as e:
                logger.error(f"❌ Не удалось загрузить модель: {e}")
                self.model = None
                return False

        try:
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self.model_path, device=self.device)
            )
            logger.info(f"✅ Модель загружена: {self.model_name}")

            await self._load_cache()
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            self.model = None
            return False

    async def embed(
        self,
        text: Union[str, List[str]],
        use_cache: bool = True,
        normalize: Optional[bool] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Получение эмбеддингов для текста/списка текстов."""
        start_time = datetime.now()

        if normalize is None:
            normalize = self.normalize

        if isinstance(text, str):
            embedding = await self._embed_single(text, use_cache, normalize)
            self._update_stats(start_time)
            return embedding

        if not text:
            return []

        if use_cache:
            embeddings = []
            texts_to_embed = []
            indices = []

            for i, t in enumerate(text):
                cache_key = self._get_cache_key(t)
                if cache_key in self.cache:
                    self.stats['cache_hits'] += 1
                    embeddings.append(self.cache[cache_key])
                else:
                    self.stats['cache_misses'] += 1
                    texts_to_embed.append(t)
                    indices.append(i)
                    embeddings.append(None)

            if not texts_to_embed:
                self.stats['embedding_requests'] += len(text)
                self.stats['total_embeddings'] += len(text)
                self._update_stats(start_time)
                return embeddings

            new_embeddings = await self._embed_batch(texts_to_embed, normalize)

            for idx, emb in zip(indices, new_embeddings):
                cache_key = self._get_cache_key(text[idx])
                self.cache[cache_key] = emb
                self.cache_metadata[cache_key] = {
                    'created': datetime.now().isoformat(),
                    'length': len(text[idx])
                }
                embeddings[idx] = emb

            await self._prune_cache()

            self.stats['embedding_requests'] += len(text)
            self.stats['total_embeddings'] += len(text)
            self._update_stats(start_time)

            return embeddings
        else:
            embeddings = await self._embed_batch(text, normalize)
            self.stats['embedding_requests'] += len(text)
            self.stats['total_embeddings'] += len(text)
            self._update_stats(start_time)
            return embeddings

    async def _embed_single(
        self,
        text: str,
        use_cache: bool,
        normalize: bool
    ) -> np.ndarray:
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                return self.cache[cache_key]
            self.stats['cache_misses'] += 1

        embedding = await self._embed_batch([text], normalize)
        embedding = embedding[0]

        if use_cache:
            cache_key = self._get_cache_key(text)
            self.cache[cache_key] = embedding
            self.cache_metadata[cache_key] = {
                'created': datetime.now().isoformat(),
                'length': len(text)
            }
            await self._prune_cache()

        return embedding

    async def _embed_batch(self, texts: List[str], normalize: bool) -> List[np.ndarray]:
        loop = asyncio.get_event_loop()

        def encode():
            if self.model is None:
                raise RuntimeError("Модель не инициализирована")
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings

        embeddings = await loop.run_in_executor(None, encode)
        self.stats['batch_processed'] += 1

        return [emb for emb in embeddings]

    def _get_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    async def _prune_cache(self):
        if len(self.cache) <= self.max_cache_size:
            return

        sorted_items = sorted(
            self.cache_metadata.items(),
            key=lambda x: x[1].get('created', '')
        )

        to_remove = int(len(self.cache) * 0.2)
        for key, _ in sorted_items[:to_remove]:
            if key in self.cache:
                del self.cache[key]
            if key in self.cache_metadata:
                del self.cache_metadata[key]

        logger.debug(f"🧹 Кэш эмбеддингов обрезан: удалено {to_remove} записей")

    async def _load_cache(self):
        try:
            cache_file = os.path.join(self.cache_dir, 'embeddings_cache.npz')
            meta_file = os.path.join(self.cache_dir, 'cache_metadata.json')

            if os.path.exists(cache_file) and os.path.exists(meta_file):
                data = np.load(cache_file, allow_pickle=True)
                self.cache = {k: v for k, v in data.items()}
                with open(meta_file, 'r', encoding='utf-8') as f:
                    self.cache_metadata = json.load(f)
                logger.info(f"📦 Загружен кэш эмбеддингов: {len(self.cache)} записей")
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")

    async def save_cache(self):
        try:
            if not self.cache:
                return

            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, 'embeddings_cache.npz')
            np.savez_compressed(cache_file, **self.cache)
            meta_file = os.path.join(self.cache_dir, 'cache_metadata.json')
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Сохранен кэш эмбеддингов: {len(self.cache)} записей")
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    async def clear_cache(self):
        self.cache.clear()
        self.cache_metadata.clear()
        logger.info("🧹 Кэш эмбеддингов очищен")
        await self.save_cache()
        return len(self.cache)

    def _update_stats(self, start_time: datetime):
        elapsed = (datetime.now() - start_time).total_seconds()
        n = self.stats['total_embeddings']
        if n > 0:
            self.stats['avg_embedding_time'] = (
                (self.stats['avg_embedding_time'] * (n - 1) + elapsed) / n
            )

    async def get_metrics(self) -> Dict[str, Any]:
        return {
            'embedding_requests': self.stats['embedding_requests'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
            'cache_size': len(self.cache),
            'batch_processed': self.stats['batch_processed'],
            'total_embeddings': self.stats['total_embeddings'],
            'avg_embedding_time': self.stats['avg_embedding_time'],
            'errors': self.stats['errors'],
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dimension': self.embedding_dimension
        }

    def get_embedding_function(self):
        """
        Возвращает функцию для использования в ChromaDB.
        ChromaDB ожидает синхронную функцию.
        """
        def embed_function(texts: List[str]) -> List[List[float]]:
            if self.model is None:
                raise RuntimeError("Модель не инициализирована")
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        return embed_function

    async def health_check(self) -> Dict[str, Any]:
        try:
            test_text = "test"
            emb = await self.embed(test_text, use_cache=False)
            return {
                'healthy': True,
                'message': 'Embedder is operational',
                'model': self.model_name,
                'device': self.device,
                'embedding_dim': len(emb) if isinstance(emb, np.ndarray) else 0,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def close(self):
        await self.save_cache()
        logger.info("✅ BGE-M3 Embedder завершил работу")


# Глобальный экземпляр для импорта в других модулях
embedder = BGE_M3_Embedder()