"""
🌍 InternetSearcher — оркестратор поиска (использует Detective)
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

from appp.core.logging import logger
from appp.services.detective.detective import Detective
from appp.services.real_search import hybrid_searcher


class InternetSearcher:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        # Используем Detective, а не ProductionDetective
        detective_config = {
            'max_pages_per_topic': config.get('max_pages_per_topic', 15),
            'min_content_length': config.get('min_content_length', 1000),
            **config
        }
        self.detective = Detective(detective_config)
        self.search_engine = hybrid_searcher
        self.cache = {}
        self.cache_ttl = 3600
        self.stats = {
            'searches': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self._initialized = False

    async def initialize(self):
        if not self._initialized:
            await self.search_engine.initialize()
            await self.detective.initialize()
            self._initialized = True
            logger.info("✅ InternetSearcher готов")
        return True

    async def search_learn_and_respond(self, query: str) -> Dict:
        """Полный цикл поиска, загрузки, анализа"""
        start = datetime.now()
        self.stats['searches'] += 1

        # Проверка кэша
        cache_key = query.strip().lower()
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if (datetime.now() - entry['timestamp']).total_seconds() < self.cache_ttl:
                self.stats['cache_hits'] += 1
                return entry['response']

        self.stats['cache_misses'] += 1

        investigation = await self.detective.investigate_topic_advanced(query)
        if not investigation.get('success'):
            return {'success': False, 'error': 'Поиск не дал результатов'}

        chunks = investigation.get('content_chunks', [])
        if not chunks:
            return {'success': False, 'error': 'Нет контента'}

        # Простейший ответ — первый чанк
        answer = chunks[0].get('text', '')[:500] if chunks else ''
        sources = list({c.get('source_url') for c in chunks[:3]})
        processing_time = (datetime.now() - start).total_seconds()

        response = {
            'success': True,
            'query': query,
            'answer': answer,
            'sources': sources,
            'pages_downloaded': investigation.get('pages_processed', 0),
            'processing_time': processing_time
        }

        self.cache[cache_key] = {
            'response': response,
            'timestamp': datetime.now()
        }
        # Ограничим размер кэша
        if len(self.cache) > 500:
            # удаляем самую старую запись
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest]

        return response

    async def get_stats(self) -> Dict:
        detective_stats = await self.detective.get_stats()
        searcher_stats = await self.search_engine.get_stats()
        return {
            'internet_searcher': self.stats.copy(),
            'detective': detective_stats,
            'searcher': searcher_stats,
            'cache_size': len(self.cache)
        }

    async def clear_cache(self):
        self.cache.clear()
        await self.detective.clear_cache()
        await self.search_engine.clear_cache()
        logger.info("🧹 Кэш InternetSearcher очищен")

    async def health_check(self) -> Dict:
        return {
            'healthy': self._initialized,
            'message': 'InternetSearcher operational',
            'timestamp': datetime.now().isoformat()
        }

    async def close(self):
        await self.detective.cleanup()
        await self.search_engine.close()
        logger.info("✅ InternetSearcher закрыт")