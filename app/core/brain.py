"""
Продакшен AI Brain - центральный координатор системы
"""
from typing import Dict, Any, Optional
import asyncio
import uuid  # <-- ДОБАВЛЕНО

from app.core.logging import logger
from app.core.config import settings
from app.infrastructure.cache import Cache
from app.services.chat.orchestrator import ChatOrchestrator
from app.services.chat.memory import MemoryManager
from app.services.knowledge.vector_store import ChromaVectorStore


class Brain:
    """
    Центральный координатор системы с полным управлением:
    1. Векторное хранилище знаний
    2. Оркестратор чата
    3. Менеджер памяти сессий
    4. Кэширование
    5. Мониторинг и метрики
    """
    
    def __init__(self):
        self.cache = Cache()
        self.vector_store = None
        self.memory_manager = None
        self.orchestrator = None
        self.health_checker = None
        
        # Статистика
        self.stats = {
            "questions_processed": 0,
            "sessions_created": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_processing_time_ms": 0
        }
    
    async def initialize(self) -> None:
        """
        Полная инициализация системы с обработкой ошибок и fallback'ами
        """
        logger.info("🚀 Начинаем инициализацию AI Brain...")
        
        try:
            # 1. Инициализация кэша
            await self._initialize_cache()
            
            # 2. Инициализация векторного хранилища
            await self._initialize_vector_store()
            
            # 3. Инициализация менеджера памяти
            await self._initialize_memory_manager()
            
            # 4. Инициализация оркестратора
            await self._initialize_orchestrator()
            
            # 5. Инициализация HealthChecker (отложенная, чтобы избежать циклической зависимости)
            await self._initialize_health_checker()
            
            # 6. Запуск мониторинга
            await self._start_monitoring()
            
            logger.info("✅ AI Brain полностью инициализирован и готов к работе")
            
        except Exception as e:
            logger.critical(f"❌ Критическая ошибка инициализации AI Brain: {e}", exc_info=True)
            raise
    
    async def _initialize_cache(self) -> None:
        """Инициализация системы кэширования"""
        try:
            await self.cache.initialize()
            logger.info("✅ Кэш инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Кэш не удалось инициализировать: {e}")
            # Система продолжит работу без кэша
    
    async def _initialize_vector_store(self) -> None:
        """Инициализация векторного хранилища"""
        try:
            self.vector_store = ChromaVectorStore()
            await self.vector_store.initialize()
            logger.info("✅ Векторное хранилище инициализировано")
        except Exception as e:
            logger.error(f"❌ Векторное хранилище не удалось инициализировать: {e}")
            raise
    
    async def _initialize_memory_manager(self) -> None:
        """Инициализация менеджера памяти сессий"""
        try:
            self.memory_manager = MemoryManager(
                session_ttl=getattr(settings, 'SESSION_TTL', 86400)
            )
            await self.memory_manager.initialize()
            logger.info("✅ Менеджер памяти сессий инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Менеджер памяти не удалось инициализировать: {e}")
            # Система может работать без сохранения истории сессий
    
    async def _initialize_orchestrator(self) -> None:
        """Инициализация оркестратора чата"""
        try:
            self.orchestrator = ChatOrchestrator(
                knowledge_base=self.vector_store,
                memory_manager=self.memory_manager,
                use_cache=getattr(settings, 'CACHE_ENABLED', True),
                use_enhancements=getattr(settings, 'ENHANCEMENTS_ENABLED', True)
            )
            logger.info("✅ Оркестратор чата инициализирован")
        except Exception as e:
            logger.error(f"❌ Оркестратор чата не удалось инициализировать: {e}")
            raise
    
    async def _initialize_health_checker(self) -> None:
        """Инициализация HealthChecker (отложенная)"""
        try:
            from app.monitoring.health import HealthChecker
            self.health_checker = HealthChecker(brain_instance=self)
            logger.info("✅ HealthChecker инициализирован")
        except ImportError as e:
            logger.warning(f"⚠️ HealthChecker не найден: {e}")
        except Exception as e:
            logger.warning(f"⚠️ HealthChecker не удалось инициализировать: {e}")
    
    async def _start_monitoring(self) -> None:
        """Запуск системы мониторинга"""
        if getattr(settings, 'METRICS_ENABLED', False):
            try:
                # Запуск метрик
                from app.monitoring.metrics import start_metrics_collection
                await start_metrics_collection(port=getattr(settings, 'METRICS_PORT', 8001))
                logger.info("✅ Метрики запущены")
            except ImportError as e:
                logger.warning(f"⚠️ Модуль метрик не найден: {e}")
            except Exception as e:
                logger.warning(f"⚠️ Метрики не удалось запустить: {e}")
    
    async def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
        use_knowledge: bool = True
    ) -> Dict[str, Any]:
        """
        Основной метод для обработки вопросов пользователя
        
        Args:
            question: Вопрос пользователя
            session_id: ID сессии для контекста
            use_knowledge: Использовать ли базу знаний
        
        Returns:
            Ответ системы с метаданными
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self.orchestrator:
                raise RuntimeError("Система не инициализирована")
            
            # Генерация session_id если не передан (уже импортирован uuid наверху)
            if not session_id:
                session_id = str(uuid.uuid4())  # <-- ИСПРАВЛЕНО
                self.stats["sessions_created"] += 1
            
            # Обработка вопроса через оркестратор
            response = await self.orchestrator.process(
                question=question,
                session_id=session_id,
                use_knowledge=use_knowledge
            )
            
            # Обновление статистики
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self.stats["questions_processed"] += 1
            self.stats["total_processing_time_ms"] += processing_time
            
            if response.get("from_cache"):
                self.stats["cache_hits"] += 1
            
            # Добавляем session_id в ответ
            response["session_id"] = session_id
            response["processing_time_ms"] = processing_time
            
            return response
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Ошибка обработки вопроса: {e}", exc_info=True)
            
            return {
                "answer": "Произошла внутренняя ошибка при обработке вашего вопроса. Пожалуйста, попробуйте позже.",
                "confidence": 0.0,
                "sources": [],
                "metadata": {"error": "internal_error"},
                "followup_suggestions": [],
                "processing_time_ms": (asyncio.get_event_loop().time() - start_time) * 1000,
                "from_cache": False,
                "session_id": session_id or "error-session"
            }
    
    async def add_knowledge(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "api",
        tags: Optional[list] = None
    ) -> Dict[str, Any]:
        """Добавление знаний в систему с умным процессором"""
        try:
            if not self.vector_store:
                raise RuntimeError("Векторное хранилище не инициализировано")
            
            logger.info(f"🧠 Начинаем добавление знаний из источника: {source}")
            
            # Объединяем метаданные
            meta_data = metadata or {}
            meta_data.update({
                "source": source,
                "tags": tags or []
            })
            
            # ИСПРАВЛЕНО: Динамический импорт для избежания циклических зависимостей
            from app.services.knowledge.processor_factory import ProcessorFactory
            
            logger.info(f"🔍 Выбираем процессор для контента длиной {len(content)} символов")
            
            processor = ProcessorFactory.get_processor(content, meta_data.get('filename', 'unknown'), meta_data)
            
            logger.info("🔄 Обрабатываем контент...")
            
            # Обрабатываем контент
            chunks = processor.process_content(content, meta_data.get('filename', 'unknown'), meta_data)
            
            logger.info(f"📊 Получено {len(chunks)} чанков после обработки")
            
            if not chunks:
                logger.warning("⚠️ Процессор не смог создать чанки из контента")
                
                # Пробуем создать хотя бы один чанк
                if content and len(content.strip()) > 10:
                    from app.services.knowledge.base import KnowledgeChunk
                    from pathlib import Path
                    
                    filename = meta_data.get('filename', 'unknown')
                    chunk = KnowledgeChunk(
                        id=f"{Path(filename).stem}_fallback_{hash(content[:50])}",
                        content=content[:5000],
                        metadata={
                            "source": filename,
                            "file_name": filename,
                            "type": "fallback",
                            "content_type": "text",
                            **meta_data
                        }
                    )
                    chunks = [chunk]
                    logger.info("✅ Создан fallback чанк")
            
            if not chunks:
                return {
                    "success": False,
                    "error": "Не удалось создать чанки из контента",
                    "chunk_count": 0
                }
            
            # Добавляем чанки в векторное хранилище
            logger.info(f"📤 Добавляем {len(chunks)} чанков в векторное хранилище...")
            chunk_ids = await self.vector_store.add(chunks)
            
            logger.info(f"✅ Добавлено {len(chunk_ids)} чанков знаний в базу")
            
            return {
                "success": True,
                "chunk_count": len(chunk_ids),
                "chunk_ids": chunk_ids,
                "message": f"Добавлено {len(chunk_ids)} фрагментов знаний"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления знаний: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": "Не удалось добавить знания"
            }
            
    
    async def search_knowledge(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> list:
        """Поиск в базе знаний"""
        try:
            if not self.vector_store:
                return []
            
            results = await self.vector_store.search(
                query=query,
                top_k=top_k,
                filters=filters
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска знаний: {e}")
            return []
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Получение полной информации о системе"""
        try:
            info = {
                "version": "1.0.0",
                "status": "operational",
                "components": {
                    "cache": await self.cache.get_stats() if self.cache else {"status": "not_initialized"},
                    "vector_store": await self.vector_store.get_info() if self.vector_store else {"status": "not_initialized"},
                    "memory_manager": await self.memory_manager.get_session_stats() if self.memory_manager else {"status": "not_initialized"},
                    "orchestrator": {"status": "initialized" if self.orchestrator else "not_initialized"}
                },
                "stats": self.stats.copy(),
                "settings": {
                    "cache_enabled": getattr(settings, 'CACHE_ENABLED', True),
                    "chroma_mode": getattr(settings, 'CHROMA_MODE', 'persistent'),
                    "embedding_model": getattr(settings, 'EMBEDDING_MODEL', 'unknown')
                }
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о системе: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, bool]:
        """Комплексная проверка здоровья системы"""
        health_status = {}
        
        try:
            # Проверка кэша
            health_status["cache"] = self.cache is not None
            
            # Проверка векторного хранилища
            health_status["vector_store"] = self.vector_store is not None
            
            # Проверка менеджера памяти
            health_status["memory_manager"] = self.memory_manager is not None
            
            # Проверка оркестратора
            health_status["orchestrator"] = self.orchestrator is not None
            
            return health_status
            
        except Exception as e:
            logger.error(f"Ошибка проверки здоровья: {e}")
            return {component: False for component in ["cache", "vector_store", "memory_manager", "orchestrator"]}
    
    async def shutdown(self) -> None:
        """Корректное завершение работы системы"""
        logger.info("Завершение работы AI Brain...")
        
        try:
            if self.vector_store:
                await self.vector_store.close()
            
            if self.memory_manager:
                await self.memory_manager.close()
            
            logger.info("✅ AI Brain завершил работу")
            
        except Exception as e:
            logger.error(f"Ошибка при завершении работы: {e}")


# Глобальный экземпляр с lazy initialization
_brain_instance = None

def get_brain() -> Brain:
    """Фабрика для получения экземпляра Brain (паттерн Singleton)"""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = Brain()
    return _brain_instance

brain = get_brain()