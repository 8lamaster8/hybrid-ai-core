"""
🎯 СЕРВИСНЫЙ КООРДИНАТОР - Мозг системы
Координирует все компоненты, управляет очередями, балансирует нагрузку
"""
import os
import json
import asyncio
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
import heapq
import uuid
from urllib.parse import urlparse
import re
import time

from appp.core.config import Config
from appp.core.logging import logger
from appp.utils.response_templates import format_rich_response

class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskType(Enum):
    SIMPLE_QUESTION = "simple_question"
    DEEP_RESEARCH = "deep_research"
    SELF_LEARNING = "self_learning"
    TOPIC_EXPLORATION = "topic_exploration"
    KNOWLEDGE_UPDATE = "knowledge_update"
    SYSTEM_MAINTENANCE = "system_maintenance"
    CACHE_CLEANUP = "cache_cleanup"
    GRAPH_OPTIMIZATION = "graph_optimization"


class ServiceCoordinator:
    """
    Главный координатор всех сервисов системы.
    Управляет задачами, балансирует нагрузку, обеспечивает отказоустойчивость.
    """
    
    def __init__(self, config: Config, **services):
        self.config = config.coordinator
        self.services = services  # detective, committee, analyst, interviewer, chroma_db, graph_db, engram, embedder
        
        # Очереди задач с приоритетами
        self.priority_queue = []
        self.task_queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        
        # Хранилище задач
        self.tasks: Dict[str, Dict] = {}
        self.task_results: Dict[str, Any] = {}
        
        # Воркеры
        self.worker_tasks: List[asyncio.Task] = []
        self.num_workers = self.config.num_workers
        
        # Метрики
        self.metrics = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'avg_processing_time': 0,
            'queue_wait_time': 0,
            'worker_utilization': 0,
            'memory_usage_mb': 0,
            'errors': {}
        }
        
        # Статистика по типам задач
        self.task_stats = {task_type.value: {'processed': 0, 'failed': 0, 'avg_time': 0} 
                          for task_type in TaskType}
        
        # Семафоры
        self.processing_semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        # Подписки на события
        self.event_subscribers: Dict[str, List[Callable]] = {
            'task_started': [],
            'task_completed': [],
            'task_failed': [],
            'system_alert': []
        }
        
        # Расписание фоновых задач
        self.scheduled_tasks = []
        
        self.is_running = False
        self.is_shutting_down = False
        
        logger.info("🎯 ServiceCoordinator создан")
    
    async def initialize(self):
        """Инициализация координатора"""
        logger.info("🔄 Инициализация ServiceCoordinator...")
        
        try:
            await self._start_workers()
            await self._start_monitoring()
            await self._schedule_background_tasks()
            await self._check_services_health()
            
            self.is_running = True
            logger.info("✅ ServiceCoordinator инициализирован")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации координатора: {e}")
            return False
    
    async def _start_workers(self):
        for i in range(self.num_workers):
            worker_task = asyncio.create_task(
                self._worker_loop(f"worker-{i+1}"),
                name=f"coordinator_worker_{i}"
            )
            self.worker_tasks.append(worker_task)
    
    async def _worker_loop(self, worker_name: str):
        logger.debug(f"Воркер {worker_name} запущен")
        while not self.is_shutting_down:
            try:
                async with self.processing_semaphore:
                    task_data = await self._get_next_task()
                    if task_data is None:
                        await asyncio.sleep(0.1)
                        continue

                    task_id = task_data['task_id']
                    # Проверяем, не отменена ли задача
                    if self.tasks.get(task_id, {}).get('status') == TaskStatus.CANCELLED.value:
                        logger.info(f"⏭️ Задача {task_id} отменена, пропускаем")
                        continue
                    
                    result = await self._process_task(task_data)
                    self.task_results[task_id] = result
                    
                    if result.get('success', False):
                        await self._update_task_status(task_id, TaskStatus.COMPLETED)
                    else:
                        await self._update_task_status(task_id, TaskStatus.FAILED)
                        logger.error(f"❌ {worker_name} не выполнил задачу {task_id}: {result.get('error')}")
                    
                    await self._emit_event('task_completed', {
                        'task_id': task_id,
                        'worker': worker_name,
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    self._update_metrics(task_data, result)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в воркере {worker_name}: {e}")
                self.metrics['errors'][worker_name] = self.metrics['errors'].get(worker_name, 0) + 1
                await asyncio.sleep(1)
    
    async def _get_next_task(self) -> Optional[Dict]:
        try:
            if self.priority_queue:
                _, task_id = heapq.heappop(self.priority_queue)
                if task_id in self.tasks:
                    return self.tasks[task_id]
            if not self.task_queue.empty():
                return await self.task_queue.get()
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении задачи: {e}")
            return None
    
    async def _process_task(self, task_data: Dict) -> Dict:
        task_id = task_data['task_id']
        task_type = task_data['type']
        start_time = time.time()
        
        try:
            task_timeout = task_data.get('timeout', self.config.task_timeout)
            
            if task_type == TaskType.SIMPLE_QUESTION.value:
                result = await self._process_simple_question(task_data)
            elif task_type == TaskType.DEEP_RESEARCH.value:
                result = await self._process_deep_research(task_data)
            elif task_type == TaskType.SELF_LEARNING.value:
                result = await self._process_self_learning(task_data)
            elif task_type == TaskType.TOPIC_EXPLORATION.value:
                result = await self._process_topic_exploration(task_data)
            elif task_type == TaskType.KNOWLEDGE_UPDATE.value:
                result = await self._process_knowledge_update(task_data)
            elif task_type == TaskType.SYSTEM_MAINTENANCE.value:
                result = await self._process_system_maintenance(task_data)
            elif task_type == TaskType.CACHE_CLEANUP.value:
                result = await self._process_cache_cleanup(task_data)
            elif task_type == TaskType.GRAPH_OPTIMIZATION.value:
                result = await self._process_graph_optimization(task_data)
            else:
                result = {'success': False, 'error': f'Unknown task type: {task_type}'}
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['task_id'] = task_id
            
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Таймаут задачи {task_id}")
            return {'success': False, 'error': f'Task timeout after {task_timeout} seconds', 'task_id': task_id}
        except Exception as e:
            logger.error(f"Ошибка обработки задачи {task_id}: {e}")
            return {'success': False, 'error': str(e), 'task_id': task_id, 'traceback': traceback.format_exc()}



    async def _process_self_learning(self, task_data: Dict) -> Dict:
        """
        Самообучение: выбирает тему из существующих знаний или случайную,
        проводит исследование и сохраняет новые знания.
        """
        logger.info("🧠 Запуск самообучения")
        start_time = time.time()

        # 1. Получаем доступ к сервисам
        engram = self.services.get('engram')
        graph = self.services.get('graph_db')
        chroma = self.services.get('chroma_db')
        detective = self.services.get('detective')
        analyst = self.services.get('analyst')
        interviewer = self.services.get('interviewer')

        if not detective or not analyst or not interviewer:
            return {'success': False, 'error': 'Необходимые сервисы не доступны'}

        # 2. Пытаемся получить тему для изучения
        topic = None
        # Сначала пробуем взять из Engram запись с низкой уверенностью или давно не обновлявшуюся
        if engram:
            try:
                # Получаем список всех ключей (метод не реализован, но для примера)
                # В реальности нужно добавить метод get_all_keys() или аналогичный
                # Пока используем запасной вариант: случайная тема
                topics_pool = [
                    "Квантовая запутанность",
                    "Теорема Гёделя о неполноте",
                    "Искусственный интеллект",
                    "Нейронные сети",
                    "Алгоритмы сортировки",
                    "История Древнего Рима",
                    "Фотосинтез",
                    "Термодинамика",
                    "Экономические кризисы",
                    "Философия Канта"
                ]
                import random
                topic = random.choice(topics_pool)
            except Exception as e:
                logger.warning(f"Не удалось получить тему из Engram: {e}")
        
        if not topic:
            # Берём случайную тему из предопределённого списка
            import random
            topics_pool = [
                "Квантовая механика", "Теория относительности", "Машинное обучение",
                "Биология клетки", "История Византии", "Русская литература XIX века",
                "Функциональное программирование", "Блокчейн", "Нанотехнологии"
            ]
            topic = random.choice(topics_pool)

        logger.info(f"📚 Самообучение: выбрана тема '{topic}'")

        # 3. Генерируем исследовательские вопросы
        questions = await interviewer.generate_research_questions(
            topic, depth=2, num_questions=8
        )

        # 4. Проводим исследование через детектив
        investigation = await detective.investigate_topic_advanced(topic, questions)
        if not investigation.get('success'):
            return {'success': False, 'error': 'Ошибка исследования темы'}

        chunks = investigation.get('content_chunks', [])
        if not chunks:
            return {'success': False, 'error': 'Не удалось получить контент'}

        # 5. Анализируем
        analysis = await analyst.analyze(chunks, query=topic)

        # 6. Сохраняем знания
        await self._store_knowledge(topic, analysis)

        processing_time = time.time() - start_time
        logger.info(f"✅ Самообучение по теме '{topic}' завершено за {processing_time:.2f} сек")

        return {
            'success': True,
            'topic': topic,
            'pages_processed': investigation.get('pages_processed', 0),
            'chunks_analyzed': len(chunks),
            'key_points_count': len(analysis.get('key_points', [])),
            'confidence': analysis.get('confidence', 0),
            'processing_time': processing_time
        }


    async def _process_topic_exploration(self, task_data: Dict) -> Dict:
        """
        Исследование темы с циклами координат: генерирует вопросы, получает ответы,
        углубляется на основе полученной информации.
        """
        topic = task_data.get('topic')
        max_cycles = task_data.get('max_cycles', 3)
        if not topic:
            return {'success': False, 'error': 'Не указана тема'}

        logger.info(f"🌀 Цикл координат для темы '{topic}', макс. циклов: {max_cycles}")
        start_time = time.time()

        detective = self.services.get('detective')
        analyst = self.services.get('analyst')
        interviewer = self.services.get('interviewer')

        if not detective or not analyst or not interviewer:
            return {'success': False, 'error': 'Необходимые сервисы не доступны'}

        all_chunks = []
        all_key_points = []          # ← собираем все факты за все циклы
        cycle_results = []
        current_depth = 0
        current_questions = [f"Что такое {topic}?"]  # начальный вопрос

        for cycle in range(max_cycles):
            logger.info(f"🔄 Цикл {cycle+1}/{max_cycles}, глубина {current_depth}")

            # Поиск и загрузка контента по текущим вопросам
            investigation = await detective.investigate_topic_advanced(
                topic, 
                questions=current_questions[:3]
            )
            if not investigation.get('success'):
                logger.warning(f"Ошибка в цикле {cycle+1}: {investigation.get('error')}")
                break

            chunks = investigation.get('content_chunks', [])
            if not chunks:
                logger.warning(f"Нет новых чанков в цикле {cycle+1}")
                break

            all_chunks.extend(chunks)

            # Анализируем новые чанки
            analysis = await analyst.analyze(chunks, query=topic)
            cycle_key_points = analysis.get('key_points', [])
            
            # Добавляем найденные факты в общий список
            if cycle_key_points:
                all_key_points.extend(cycle_key_points)
                logger.info(f"   ✅ Добавлено {len(cycle_key_points)} фактов (всего {len(all_key_points)})")
            else:
                logger.warning(f"   ⚠️ Аналитик не вернул фактов в этом цикле")

            # Сохраняем результаты цикла
            cycle_results.append({
                'cycle': cycle + 1,
                'questions': current_questions,
                'pages': investigation.get('pages_processed', 0),
                'chunks': len(chunks),
                'key_points': cycle_key_points[:5],
                'confidence': analysis.get('confidence', 0)
            })

            # Генерируем углубляющие вопросы
            deepening = await interviewer.generate_deepening_questions(
                knowledge_chunks=chunks,
                current_depth=current_depth,
                max_questions=3
            )

            if deepening:
                current_questions = deepening
                current_depth += 1
            else:
                break

        # Финальный синтез
        if all_key_points:
            # Убираем дубликаты (по первым 100 символам)
            seen = set()
            unique_points = []
            for point in all_key_points:
                sig = point[:100].lower()
                if sig not in seen:
                    seen.add(sig)
                    unique_points.append(point)
            all_key_points = unique_points

            # Берём топ-15 фактов для вывода
            key_insights = all_key_points[:15]
            synthesis = "\n".join([f"• {p}" for p in key_insights[:10]])  # краткий синтез
            confidence = min(0.9, 0.5 + 0.1 * len(all_key_points))  # эвристика
        else:
            # Fallback: если фактов нет, берём первые предложения из лучших чанков
            logger.warning("⚠️ Факты не накоплены, используем fallback (первые предложения из чанков)")
            import re
            fallback_texts = []
            for chunk in all_chunks[:5]:
                text = chunk.get('text', '')
                if text:
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    for sent in sentences[:3]:
                        sent = sent.strip()
                        if 30 < len(sent) < 500 and not re.search(r'https?://|©|фото|купить', sent, re.I):
                            fallback_texts.append(sent)
            if fallback_texts:
                key_insights = fallback_texts[:10]
                synthesis = "\n".join([f"• {p}" for p in key_insights[:5]])
                confidence = 0.3
            else:
                key_insights = ["Не удалось извлечь конкретные факты."]
                synthesis = "Информация по теме не найдена."
                confidence = 0.0

        # Сохраняем итоговое знание в хранилища
        if all_key_points or fallback_texts:
            await self._store_knowledge(topic, {
                'summary': synthesis,
                'key_points': key_insights,
                'confidence': confidence,
                'cycles': len(cycle_results)
            })

        processing_time = time.time() - start_time

        return {
            'success': True,
            'topic': topic,
            'cycles_completed': len(cycle_results),
            'knowledge_chunks': len(all_chunks),
            'synthesis': synthesis,
            'key_insights': key_insights,
            'confidence': confidence,
            'cycle_details': cycle_results,
            'processing_time': processing_time
        }

    async def _process_system_maintenance(self, task_data: Dict) -> Dict:
        """
        Системное обслуживание: очистка старых логов, ротация, проверка диска.
        """
        logger.info("🛠️ Системное обслуживание")
        start_time = time.time()

        # Очистка старых временных файлов
        import shutil
        cache_dirs = [
            './data/cache/detective',
            './data/cache/embeddings',
        ]

        cleared = 0
        errors = []

        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    # Удаляем файлы старше 7 дней
                    now = time.time()
                    for filename in os.listdir(cache_dir):
                        filepath = os.path.join(cache_dir, filename)
                        if os.path.isfile(filepath):
                            if os.path.getmtime(filepath) < now - 7 * 86400:
                                os.remove(filepath)
                                cleared += 1
                except Exception as e:
                    errors.append(f"{cache_dir}: {e}")

        # Ротация логов (если настроено)
        log_file = self.config.get('log_file', './data/logs/autonomous_ai.log')
        max_size = self.config.get('max_log_size', 10 * 1024 * 1024)
        backup_count = self.config.get('backup_count', 5)

        if os.path.exists(log_file):
            try:
                size = os.path.getsize(log_file)
                if size > max_size:
                    # Простейшая ротация
                    for i in range(backup_count - 1, 0, -1):
                        old = f"{log_file}.{i}"
                        new = f"{log_file}.{i+1}"
                        if os.path.exists(old):
                            os.rename(old, new)
                    os.rename(log_file, f"{log_file}.1")
                    open(log_file, 'w').close()
                    logger.info("♻️ Лог-файл ротирован")
            except Exception as e:
                errors.append(f"ротация логов: {e}")

        processing_time = time.time() - start_time

        return {
            'success': True,
            'cleaned_files': cleared,
            'errors': errors,
            'processing_time': processing_time
        }


    async def _process_cache_cleanup(self, task_data: Dict) -> Dict:
        """
        Очистка всех кэшей.
        """
        logger.info("🧹 Очистка кэшей")
        start_time = time.time()

        detective = self.services.get('detective')
        embedder = self.services.get('embedder')
        searcher = self.services.get('internet_searcher')  # если есть

        results = {}

        if detective and hasattr(detective, 'clear_cache'):
            try:
                await detective.clear_cache()
                results['detective'] = 'кэш очищен'
            except Exception as e:
                results['detective'] = f'ошибка: {e}'

        if embedder and hasattr(embedder, 'clear_cache'):
            try:
                await embedder.clear_cache()
                results['embedder'] = 'кэш эмбеддингов очищен'
            except Exception as e:
                results['embedder'] = f'ошибка: {e}'

        if searcher and hasattr(searcher, 'clear_cache'):
            try:
                await searcher.clear_cache()
                results['internet_searcher'] = 'кэш поиска очищен'
            except Exception as e:
                results['internet_searcher'] = f'ошибка: {e}'

        processing_time = time.time() - start_time

        return {
            'success': True,
            'results': results,
            'processing_time': processing_time
        }

    async def _process_graph_optimization(self, task_data: Dict) -> Dict:
        """
        Оптимизация графа знаний.
        """
        logger.info("🕸️ Оптимизация графа знаний")
        start_time = time.time()

        graph = self.services.get('graph_db')
        if not graph:
            return {'success': False, 'error': 'GraphDB не доступен'}

        try:
            # Предполагаем, что у NetworkXGraphService есть метод optimize()
            if hasattr(graph, 'optimize'):
                result = await graph.optimize()
            else:
                # Реализуем простую оптимизацию на месте
                # Например, удаляем узлы без связей
                import networkx as nx
                isolated = list(nx.isolates(graph.graph))
                graph.graph.remove_nodes_from(isolated)
                result = {'nodes_removed': len(isolated), 'edges_remaining': graph.graph.number_of_edges()}
                await graph.save()
        except Exception as e:
            logger.error(f"Ошибка оптимизации графа: {e}")
            return {'success': False, 'error': str(e)}

        processing_time = time.time() - start_time

        return {
            'success': True,
            'optimization_result': result,
            'processing_time': processing_time
        }


    async def _process_knowledge_update(self, task_data: Dict) -> Dict:
        """
        Обновление знаний: переиндексация ChromaDB, синхронизация графа и т.п.
        """
        logger.info("🔄 Обновление знаний")
        start_time = time.time()

        chroma = self.services.get('chroma_db')
        graph = self.services.get('graph_db')
        engram = self.services.get('engram')

        results = {}

        if chroma:
            try:
                # В ChromaDBService нужно добавить метод rebuild_indexes()
                # Если его нет, пропускаем
                if hasattr(chroma, 'rebuild_indexes'):
                    await chroma.rebuild_indexes()
                    results['chroma'] = 'индексы перестроены'
                else:
                    results['chroma'] = 'метод rebuild_indexes не реализован'
            except Exception as e:
                results['chroma'] = f'ошибка: {e}'

        if graph:
            try:
                # Например, перестроить связи
                if hasattr(graph, 'optimize'):
                    await graph.optimize()
                    results['graph'] = 'граф оптимизирован'
                else:
                    results['graph'] = 'нет метода optimize'
            except Exception as e:
                results['graph'] = f'ошибка: {e}'

        if engram:
            try:
                # Очистка старых записей и т.п.
                if hasattr(engram, 'cleanup'):
                    await engram.cleanup()
                    results['engram'] = 'очистка выполнена'
                else:
                    results['engram'] = 'нет метода cleanup'
            except Exception as e:
                results['engram'] = f'ошибка: {e}'

        processing_time = time.time() - start_time

        return {
            'success': True,
            'results': results,
            'processing_time': processing_time
        }


    def _detect_question_type(self, question: str) -> str:
        """Определяет тип вопроса для выбора шаблона."""
        q = question.lower()
        if 'теорем' in q or 'пифагор' in q or 'формул' in q:
            return 'mathematical_theorem'
        elif 'войн' in q or 'истори' in q or 'событи' in q or 'год' in q:
            return 'historical_event'
        elif 'функци' in q or 'класс' in q or 'метод' in q or 'переменн' in q or 'язык' in q:
            return 'programming_concept'
        elif 'физик' in q or 'хими' in q or 'биолог' in q or 'квантов' in q:
            return 'scientific_concept'
        else:
            return 'default'  # общий случай
    
    # ---------- ОБРАБОТКА ПРОСТОГО ВОПРОСА (с Engram, ChromaDB, Детективом) ----------
        # ---------- ЗАМЕНИ ЭТОТ МЕТОД ПОЛНОСТЬЮ ----------
    async def _process_simple_question(self, task_data: Dict) -> Dict:
        question = task_data['question']
        start = time.time()

        # 1. Engram
        engram = self.services.get('engram')
        if engram:
            try:
                mem = await engram.retrieve(question, top_k=1, min_confidence=0.6)
                if mem:
                    return {
                        'success': True,
                        'source': 'engram',
                        'answer': mem[0]['content'],
                        'confidence': mem[0]['confidence'],
                        'sources': ['🧠 Engram'],
                        'profile': 'default',
                        'key_facts_metadata': [],
                        'query': question,
                        'processing_time': 0.1
                    }
            except Exception as e:
                logger.warning(f"Engram retrieve error: {e}")

        # 2. Детектив
        detective = self.services.get('detective')
        committee = self.services.get('committee')
        analyst = self.services.get('analyst')

        if not (detective and committee and analyst):
            return {'success': False, 'error': 'Сервисы не доступны'}

        try:
            # 2.1 Поиск
            search_result = await detective.search(question, num_results=15)
            if not search_result.get('success') or not search_result.get('results'):
                return {'success': False, 'error': 'Нет результатов поиска'}

            # 2.2 Фильтрация комитетом
            filtered = await committee.batch_evaluate(search_result['results'][:10])
            if not filtered:
                return {'success': False, 'error': 'Все результаты отбракованы'}

            # 2.3 Приоритеты доменов
            priority_domains = ['ru.wikipedia.org', 'habr.com', 'postnauka.ru', 'nplus1.ru', 'elementy.ru']
            trash_domains = [
                'otvet.mail.ru', 'answer.mail.ru', 'bolshoyvopros.ru',
                'dzen.ru', 'yandex.ru/q', 'traveler.ru', 'rtraveler.ru',
                'rambler.ru', 'mail.ru', 'ok.ru', 'vk.com',
                'reverso.net', 'translate.', 'wordhippo.com', 'academic.ru',
                '24smi.org', 'uznayvse.ru', 'socionika.info'
            ]

            candidates = []
            for doc in filtered:
                url = doc.get('url', '')
                domain = urlparse(url).netloc.lower()
                if any(bad in domain for bad in trash_domains):
                    continue
                candidates.append((url, domain, doc))

            priority_urls = []
            other_urls = []
            for url, domain, doc in candidates:
                if any(p in domain for p in priority_domains):
                    if 'wikipedia' in domain and not domain.startswith('ru.wikipedia'):
                        continue
                    priority_urls.append((url, doc))
                else:
                    other_urls.append((url, doc))

            urls_to_fetch = []
            for url, _ in priority_urls[:2]:
                urls_to_fetch.append(url)
            for url, _ in other_urls:
                if len(urls_to_fetch) >= 3:
                    break
                if url not in urls_to_fetch:
                    urls_to_fetch.append(url)

            logger.info(f"📥 Загружаем: {urls_to_fetch}")
            fetch_tasks = [detective.fetch_page_content(url, question) for url in urls_to_fetch]
            pages = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            valid_pages = [p for p in pages if isinstance(p, dict) and p.get('success')]
            logger.info(f"📊 Загружено: {len(valid_pages)}/{len(fetch_tasks)}")

            if valid_pages:
                analyst_docs = []
                for page in valid_pages:
                    analyst_docs.append({
                        'url': page['url'],
                        'title': page.get('title', ''),
                        'content': page.get('content', ''),
                    })

                # Анализ
                analysis = await analyst.analyze(analyst_docs, query=question)
                # analysis содержит: profile, key_facts_metadata, confidence, key_points, summary и т.д.

                # Сохраняем в Engram (если высокое качество)
                if analysis.get('confidence', 0) > 0.5 and analysis.get('key_points'):
                    if engram:
                        try:
                            content = '\n'.join(analysis['key_points'][:5])
                            await engram.store(
                                key=question,
                                content=content,
                                metadata={
                                    'source': 'detective',
                                    'confidence': analysis['confidence'],
                                    'sources': [page['url'] for page in valid_pages[:3]]
                                },
                                confidence=analysis['confidence']
                            )
                        except Exception as e:
                            logger.error(f"Engram store error: {e}")

                # --- СОХРАНЕНИЕ В ГРАФ ---
                if analysis.get('key_points') and analysis.get('confidence', 0) > 0.6:
                    clean_topic = question.strip().rstrip('?').strip()[:100]
                    if clean_topic:
                        analysis_for_graph = {
                            'summary': analysis.get('key_points', [''])[0],
                            'key_points': analysis['key_points'],
                            'confidence': analysis['confidence']
                        }
                        await self._store_knowledge(clean_topic, analysis_for_graph)
                        logger.info(f"📌 Результат вопроса сохранён в граф по теме '{clean_topic}'")
                # ------------------------------------

                return {
                    'success': True,
                    'source': 'detective',
                    # поле 'answer' не заполняем, оно будет сформировано через шаблон в autonomous_ai.py
                    'confidence': analysis.get('confidence', 0.5),
                    'sources': [page['url'] for page in valid_pages[:3]],
                    'profile': analysis.get('profile', 'default'),
                    'key_facts_metadata': analysis.get('key_facts_metadata', []),
                    'key_points': analysis.get('key_points', []),
                    'query': question,
                    'processing_time': time.time() - start
                }
            else:
                # Fallback — сниппеты
                fallback = self._synthesize_from_snippets(filtered[:3])
                return {
                    'success': True,
                    'source': 'detective_fallback',
                    'answer': fallback,
                    'confidence': 0.3,
                    'sources': [d.get('url', '') for d in filtered[:3]],
                    'profile': 'default',
                    'key_facts_metadata': [],
                    'key_points': [],
                    'query': question,
                    'processing_time': time.time() - start
                }

        except Exception as e:
            logger.error(f"Ошибка: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
 

    # ---------- ДОБАВЬ ЭТОТ МЕТОД (вспомогательный) ----------
    def _synthesize_from_snippets(self, docs: List[Dict]) -> str:
        """Склеивает заголовки и сниппеты для быстрого ответа"""
        parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.get('title', '')
            snippet = doc.get('snippet', '')
            if title:
                parts.append(f"{i}. {title}")
            if snippet:
                parts.append(f"   {snippet[:200]}")
        return '\n'.join(parts) if parts else "Информация не найдена."
    
    # ---------- СОХРАНЕНИЕ ЗНАНИЙ (Engram, ChromaDB, Graph) ----------
    async def _store_knowledge(self, topic: str, analysis: Dict):
        """Сохранение знаний во все хранилища."""
        storage_tasks = []

        # Engram память
        engram = self.services.get('engram')
        if engram:
            storage_tasks.append(
                engram.store(
                    key=topic,
                    content=analysis.get('summary', '') or '\n'.join(analysis.get('key_points', [])),
                    metadata={
                        'topic': topic,
                        'confidence': analysis.get('confidence', 0.5),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'coordinator',
                        'key_points_count': len(analysis.get('key_points', []))
                    },
                    confidence=analysis.get('confidence', 0.5)
                )
            )

        # ChromaDB (эмбеддинги)
        chroma = self.services.get('chroma_db')
        if chroma and 'key_points' in analysis:
            for i, point in enumerate(analysis['key_points'][:5]):  # ограничим 5
                if len(point) > 50:  # только осмысленные
                    storage_tasks.append(
                        chroma.add_document(
                            text=point,
                            metadata={
                                'topic': topic,
                                'type': 'key_point',
                                'source': 'analyst',
                                'confidence': analysis.get('confidence', 0.5),
                                'index': i
                            }
                        )
                    )

        # Граф знаний
        graph = self.services.get('graph_db')
        if graph:
            # Добавляем узел темы, если его нет
            storage_tasks.append(
                graph.add_knowledge_chunk(
                    topic=topic,
                    chunk={
                        'summary': analysis.get('summary', '')[:500],
                        'key_points': analysis.get('key_points', [])[:5],
                        'confidence': analysis.get('confidence', 0.5)
                    },
                    relations=[]  # можно добавить извлечение отношений позже
                )
            )

        # Параллельное сохранение (игнорируем ошибки)
        if storage_tasks:
            await asyncio.gather(*storage_tasks, return_exceptions=True)
            logger.info(f"💾 Знания по теме '{topic}' сохранены в {len(storage_tasks)} хранилищ")
    
    # ---------- ОСТАЛЬНЫЕ МЕТОДЫ (без изменений, но для полноты оставляем) ----------
    async def _process_deep_research(self, task_data: Dict) -> Dict:
        """
        Глубокое исследование темы с заданной глубиной.
        Генерирует вопросы, ищет информацию, анализирует, сохраняет.
        """
        topic = task_data.get('topic')
        depth = task_data.get('depth', 2)
        
        if not topic:
            return {'success': False, 'error': 'Не указана тема'}
        
        logger.info(f"🔍 Глубокое исследование: '{topic}' (глубина {depth})")
        start_time = time.time()
        
        # Получаем сервисы
        detective = self.services.get('detective')
        analyst = self.services.get('analyst')
        interviewer = self.services.get('interviewer')
        
        if not detective or not analyst or not interviewer:
            return {'success': False, 'error': 'Необходимые сервисы не доступны'}
        
        try:
            # 1. Генерируем вопросы в зависимости от глубины
            questions = await interviewer.generate_research_questions(
                topic,
                depth=depth,
                num_questions=5 + depth * 2  # 7 для depth=1, 9 для depth=2, 11 для depth=3
            )
            logger.info(f"   ✅ Сгенерировано {len(questions)} вопросов")
            
            # 2. Запускаем исследование через детектив
            investigation = await detective.investigate_topic_advanced(topic, questions)
            if not investigation.get('success'):
                return {'success': False, 'error': investigation.get('error', 'Ошибка исследования')}
            
            chunks = investigation.get('content_chunks', [])
            if not chunks:
                return {'success': False, 'error': 'Не удалось получить контент'}
            
            logger.info(f"   ✅ Получено {len(chunks)} чанков")
            
            # 3. Анализируем чанки
            analysis = await analyst.analyze(chunks, query=topic)
            
            # 4. Извлекаем ключевые факты
            key_points = analysis.get('key_points', [])
            
            # 5. Очищаем факты от мусора (используем quality.yaml если доступно)
            if key_points:
                # Если есть quality_config, используем его, иначе встроенные списки
                if hasattr(self, 'quality_config') and self.quality_config:
                    junk_phrases = self.quality_config.get('junk_phrases', [])
                else:
                    junk_phrases = [
                        'архивная копия', 'wayback machine', '↑', 'источник:',
                        'дата обращения:', 'архивировано', 'автор оригинала',
                        'лицензия creative commons', 'эта страница в последний раз',
                        'у этого термина существуют и другие значения', 'см. также',
                        'примечания', 'ссылки', 'литература', 'фото:', '©',
                        'getty images', 'reuters', 'ap'
                    ]
                
                # Функция очистки
                def clean_fact(text):
                    text = re.sub(r'\(значения\)|\[[^\]]+\]|\{\{[^\}]+\}\}', '', text)
                    text = re.sub(r'^см\.\s*|^также\s*', '', text, flags=re.I)
                    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
                    text = re.sub(r'^[,\s]+', '', text)
                    if len(text) < 30 or re.match(r'^[\s,.!?;:\-]+$', text):
                        return None
                    
                    # Проверка на мусорные фразы
                    text_lower = text.lower()
                    for junk in junk_phrases:
                        if junk in text_lower:
                            return None
                    return text.strip()
                
                cleaned_points = []
                seen = set()
                for point in key_points:
                    cleaned = clean_fact(point)
                    if cleaned and cleaned not in seen:
                        norm = cleaned.rstrip('.,!?;:').lower()
                        if norm not in seen:
                            seen.add(norm)
                            cleaned_points.append(cleaned)
                
                key_points = cleaned_points[:15]
                synthesis = " ".join(key_points[:3]) if key_points else ""
                if len(synthesis) > 300:
                    synthesis = synthesis[:300] + "..."
            else:
                # Fallback: первые предложения из чанков
                logger.warning("⚠️ Факты не найдены, используем fallback")
                fallback_texts = []
                import re
                for chunk in chunks[:3]:
                    text = chunk.get('text', '')
                    if text:
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                        for sent in sentences[:3]:
                            sent = sent.strip()
                            if 30 < len(sent) < 500 and not re.search(r'https?://|©|фото|купить', sent, re.I):
                                fallback_texts.append(sent)
                key_points = fallback_texts[:10]
                synthesis = " ".join(key_points[:3]) if key_points else "Информация не найдена."
            
            # 6. Сохраняем знания
            await self._store_knowledge(topic, {
                'summary': synthesis,
                'key_points': key_points,
                'confidence': analysis.get('confidence', 0.5),
                'depth': depth,
                'chunks_processed': len(chunks)
            })
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'topic': topic,
                'synthesis': synthesis,
                'key_findings': key_points[:10],
                'sources_used': investigation.get('pages_processed', 0),
                'chunks_processed': len(chunks),
                'depth_achieved': depth,
                'confidence': analysis.get('confidence', 0.5),
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Ошибка в _process_deep_research: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _synthesize_from_embeddings(self, results: List[Dict]) -> str:
        if not results:
            return ""
        # Простая сборка
        texts = [r['text'] for r in results if 'text' in r]
        return "\n\n".join(texts[:2])
    
    async def submit_task(self, task_data: Dict) -> str:
        task_id = str(uuid.uuid4())[:8]
        task = {
            'task_id': task_id,
            'created_at': datetime.now().isoformat(),
            'status': TaskStatus.PENDING.value,
            **task_data
        }
        if 'priority' not in task:
            task['priority'] = TaskPriority.MEDIUM.value

        self.tasks[task_id] = task

        try:
            if task['priority'] == TaskPriority.CRITICAL.value:
                heapq.heappush(self.priority_queue, (0, task_id))
            elif task['priority'] == TaskPriority.HIGH.value:
                heapq.heappush(self.priority_queue, (1, task_id))
            else:
                # Используем put_nowait, чтобы не блокироваться навсегда
                self.task_queue.put_nowait(task)
        except asyncio.QueueFull:
            logger.error(f"❌ Очередь задач переполнена, задача {task_id} отклонена")
            # Можно либо ждать, либо отменять
            raise RuntimeError("Task queue is full")

        await self._emit_event('task_started', {
            'task_id': task_id,
            'type': task['type'],
            'priority': task['priority'],
            'timestamp': task['created_at']
        })
        logger.info(f"📥 Задача добавлена: {task_id} ({task['type']})")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Dict:
        if task_id not in self.tasks:
            return {'error': 'Task not found'}
        task = self.tasks[task_id]
        result = {
            'task_id': task_id,
            'type': task.get('type'),
            'status': task.get('status'),
            'created_at': task.get('created_at'),
            'priority': task.get('priority'),
            'metadata': task.get('metadata', {})
        }
        if task_id in self.task_results:
            result['result'] = self.task_results[task_id]
        return result
    
    async def cancel_task(self, task_id: str) -> bool:
        if task_id not in self.tasks:
            return False
        task = self.tasks[task_id]
        if task['status'] not in [TaskStatus.PENDING.value, TaskStatus.PROCESSING.value]:
            return False

        # Помечаем как отменённую
        task['status'] = TaskStatus.CANCELLED.value
        task['cancelled_at'] = datetime.now().isoformat()

        # Удаляем из очередей, если ещё там (грубо, но эффективно)
        # Приоритетная очередь – нужно пересобрать
        self.priority_queue = [(p, tid) for p, tid in self.priority_queue if tid != task_id]
        heapq.heapify(self.priority_queue)

        # Для asyncio.Queue нет простого способа удалить элемент, но мы просто оставляем –
        # воркер при получении проверит статус и не будет обрабатывать.
        # Можно добавить проверку статуса в _get_next_task, но проще здесь:
        # Если задача в обычной очереди, она всё равно будет обработана, но мы её отменили,
        # поэтому воркер должен проверить статус перед обработкой.
        # Для этого добавим проверку в _worker_loop.

        logger.info(f"❌ Задача отменена: {task_id}")
        return True
    
    async def _start_monitoring(self):
        async def monitoring_loop():
            while not self.is_shutting_down:
                try:
                    await self._update_monitoring_metrics()
                    await asyncio.sleep(self.config.monitoring_interval)
                except Exception as e:
                    logger.error(f"Ошибка мониторинга: {e}")
                    await asyncio.sleep(5)
        self.monitoring_task = asyncio.create_task(monitoring_loop())
    
    async def _update_monitoring_metrics(self):
        queue_size = self.task_queue.qsize() + len(self.priority_queue)
        active_workers = len([t for t in self.worker_tasks if not t.done()])
        worker_utilization = active_workers / self.num_workers if self.num_workers > 0 else 0
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.metrics.update({
            'queue_size': queue_size,
            'worker_utilization': worker_utilization,
            'memory_usage_mb': memory_mb,
            'active_workers': active_workers,
            'pending_tasks': len([t for t in self.tasks.values() 
                                if t['status'] == TaskStatus.PENDING.value])
        })
    
    async def _schedule_background_tasks(self):
        async def hourly_maintenance():
            while not self.is_shutting_down:
                await asyncio.sleep(3600)
                await self.submit_task({'type': TaskType.SYSTEM_MAINTENANCE.value, 'priority': TaskPriority.LOW.value})
        async def daily_optimization():
            while not self.is_shutting_down:
                await asyncio.sleep(86400)
                await self.submit_task({'type': TaskType.GRAPH_OPTIMIZATION.value, 'priority': TaskPriority.LOW.value})
        async def cache_cleanup():
            while not self.is_shutting_down:
                await asyncio.sleep(21600)
                await self.submit_task({'type': TaskType.CACHE_CLEANUP.value, 'priority': TaskPriority.LOW.value})
        self.scheduled_tasks.extend([
            asyncio.create_task(hourly_maintenance()),
            asyncio.create_task(daily_optimization()),
            asyncio.create_task(cache_cleanup())
        ])
    
    async def _check_services_health(self, detailed: bool = False) -> Dict:
        health_results = {}
        for name, service in self.services.items():
            try:
                if hasattr(service, 'health_check'):
                    health = await service.health_check()
                    health_results[name] = {
                        'status': 'healthy' if health.get('healthy', False) else 'unhealthy',
                        'details': health if detailed else None
                    }
                else:
                    health_results[name] = {'status': 'unknown'}
            except Exception as e:
                health_results[name] = {'status': 'error', 'error': str(e)}
        return health_results
    
    async def get_system_metrics(self) -> Dict:
        all_metrics = {
            'coordinator': self.metrics.copy(),
            'task_stats': self.task_stats.copy(),
            'queue_info': {
                'priority_queue': len(self.priority_queue),
                'regular_queue': self.task_queue.qsize(),
                'total_tasks': len(self.tasks),
                'pending_tasks': len([t for t in self.tasks.values() 
                                    if t['status'] == TaskStatus.PENDING.value])
            }
        }
        for name, service in self.services.items():
            if hasattr(service, 'get_metrics'):
                try:
                    service_metrics = await service.get_metrics()
                    all_metrics[name] = service_metrics
                except Exception as e:
                    all_metrics[name] = {'error': str(e)}
        return all_metrics
    
    async def _emit_event(self, event_type: str, data: Dict):
        if event_type in self.event_subscribers:
            for callback in self.event_subscribers[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Ошибка в обработчике события {event_type}: {e}")
    
    def _update_metrics(self, task_data: Dict, result: Dict):
        task_type = task_data.get('type')
        if task_type in self.task_stats:
            stats = self.task_stats[task_type]
            stats['processed'] += 1
            if not result.get('success', False):
                stats['failed'] += 1
            processing_time = result.get('processing_time', 0)
            if stats['avg_time'] == 0:
                stats['avg_time'] = processing_time
            else:
                stats['avg_time'] = 0.9 * stats['avg_time'] + 0.1 * processing_time
        
        self.metrics['tasks_processed'] += 1
        if not result.get('success', False):
            self.metrics['tasks_failed'] += 1
    
    async def _update_task_status(self, task_id: str, status: TaskStatus):
        if task_id in self.tasks:
            self.tasks[task_id]['status'] = status.value
            self.tasks[task_id]['updated_at'] = datetime.now().isoformat()
    
    async def shutdown(self):
        if self.is_shutting_down:
            return
        logger.info("🛑 Завершение работы ServiceCoordinator...")
        self.is_shutting_down = True
        for task_id, task in self.tasks.items():
            if task['status'] == TaskStatus.PROCESSING.value:
                await self.cancel_task(task_id)
        for worker_task in self.worker_tasks:
            worker_task.cancel()
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
        for scheduled_task in self.scheduled_tasks:
            scheduled_task.cancel()
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        await self._save_state()
        logger.info("✅ ServiceCoordinator завершил работу")
    
    async def _save_state(self):
        try:
            state = {
                'metrics': self.metrics,
                'task_stats': self.task_stats,
                'tasks': {k: v for k, v in self.tasks.items()
                        if v['status'] in [TaskStatus.PENDING.value, TaskStatus.PROCESSING.value]},
                'saved_at': datetime.now().isoformat()
            }
            os.makedirs('./data/state', exist_ok=True)
            with open('./data/state/coordinator_state.json', 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.info("💾 Состояние координатора сохранено")
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")


# Глобальный экземпляр
service_coordinator: Optional[ServiceCoordinator] = None

def get_coordinator() -> ServiceCoordinator:
    global service_coordinator
    if service_coordinator is None:
        from appp.core.config import Config
        config = Config.get()
        service_coordinator = ServiceCoordinator(config)
    return service_coordinator