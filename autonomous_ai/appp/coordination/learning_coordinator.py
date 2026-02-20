"""
🔄 Координатор циклов самообучения для автономной AI-системы.
Управляет различными типами циклов обучения, адаптирует приоритеты,
использует компоненты системы.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse


import os
import logging

# Настройка отдельного лог-файла для обучения
learning_log_file = './data/logs/learning_ai.log'
os.makedirs(os.path.dirname(learning_log_file), exist_ok=True)

# Создаём handler для записи в файл
fh = logging.FileHandler(learning_log_file, encoding='utf-8', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Получаем логгер для этого модуля
logger = logging.getLogger(__name__)
logger.addHandler(fh)

# Если хочешь, чтобы логи обучения НЕ дублировались в основном файле autonomous_ai.log:
logger.propagate = False

# Если хочешь также видеть их в консоли (необязательно):
# ch = logging.StreamHandler()
# ch.setFormatter(formatter)
# logger.addHandler(ch)




logger = logging.getLogger(__name__)


class CycleType(Enum):
    DISCOVERY = "discovery"      # Открытие новых тем
    DEEPENING = "deepening"      # Углубление в существующие темы
    EXPANSION = "expansion"      # Расширение связей между темами
    META_ANALYSIS = "meta"       # Мета-анализ системы
    MAINTENANCE = "maintenance"  # Обслуживание (очистка, оптимизация)


class LearningCycle:
    """Базовый класс для всех циклов обучения."""
    
    def __init__(self, cycle_type: CycleType, services: Dict[str, Any]):
        self.cycle_type = cycle_type
        self.services = services  # { 'detective': ..., 'analyst': ..., 'interviewer': ..., 'committee': ..., 'engram': ..., 'graph': ..., 'chroma': ... }
        
        self.stats = {
            'executions': 0,
            'successful': 0,
            'failed': 0,
            'avg_duration': 0.0,
            'last_execution': None
        }
    
    async def execute(self) -> Dict[str, Any]:
        """Выполняет цикл – должен быть переопределён в наследниках."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'cycle_type': self.cycle_type.value,
            **self.stats
        }
    
    def _update_stats(self, success: bool, duration: float):
        self.stats['executions'] += 1
        if success:
            self.stats['successful'] += 1
        else:
            self.stats['failed'] += 1
        # скользящее среднее
        old_avg = self.stats['avg_duration']
        self.stats['avg_duration'] = old_avg + (duration - old_avg) / self.stats['executions']
        self.stats['last_execution'] = datetime.now().isoformat()


class DiscoveryCycle(LearningCycle):
    """
    Цикл открытия новых тем.
    Использует интервьюер для обнаружения новых тем, детектив для сбора информации,
    комитет для верификации, аналитик для обработки и сохраняет в хранилища.
    """
    
    def __init__(self, services: Dict[str, Any], config: Dict = None):
        super().__init__(CycleType.DISCOVERY, services)
        self.config = config or {}
        self.min_topics = self.config.get('min_topics', 1)
        self.max_topics = self.config.get('max_topics', 3)
    
    async def execute(self) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("🚀 Запуск цикла DISCOVERY")
        
        results = {
            'cycle_type': self.cycle_type.value,
            'start_time': datetime.now().isoformat(),
            'discovered_topics': [],
            'research_completed': [],
            'errors': []
        }
        
        try:
            interviewer = self.services.get('interviewer')
            detective = self.services.get('detective')
            analyst = self.services.get('analyst')
            committee = self.services.get('committee')
            engram = self.services.get('engram')
            graph = self.services.get('graph_db')
            chroma = self.services.get('chroma_db')
            
            if not all([interviewer, detective, analyst, committee]):
                raise RuntimeError("Не все необходимые сервисы доступны")
            
            # 1. Получаем потенциальные темы с приоритетами
            candidate_topics = await self._discover_potential_topics()
            results['discovered_topics'] = [{'name': t, 'priority': p} for t, p in candidate_topics[:self.max_topics]]
            
            if not candidate_topics:
                logger.info("Новых тем не обнаружено")
                self._update_stats(success=False, duration=time.time() - start_time)
                results['success'] = False
                return results
            
            # 2. Берём несколько тем с наивысшим приоритетом
            topics_to_research = [t for t, p in candidate_topics[:self.min_topics]]
            logger.info(f"   📋 Выбраны темы для изучения: {topics_to_research}")
            
            for topic in topics_to_research:
                try:
                    # Генерация исследовательских вопросов
                    questions = await interviewer.generate_research_questions(topic, depth=1, num_questions=5)
                    
                    # Берём первые 3 вопроса для поиска
                    search_queries = questions[:3] if questions else [topic]
                    
                    # --- Этапы, аналогичные simple_question ---
                    all_docs = []
                    for query in search_queries:
                        # Поиск
                        search_result = await detective.search(query, num_results=10)
                        if not search_result.get('success') or not search_result.get('results'):
                            continue
                        
                        # Фильтрация комитетом
                        filtered = await committee.batch_evaluate(search_result['results'][:10])
                        if not filtered:
                            continue
                        
                        # Отбор URL (приоритетные домены + не более 3 всего)
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
                        
                        if not urls_to_fetch:
                            continue
                        
                        # Загрузка страниц
                        fetch_tasks = [detective.fetch_page_content(url, query) for url in urls_to_fetch]
                        pages = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                        valid_pages = [p for p in pages if isinstance(p, dict) and p.get('success')]
                        
                        for page in valid_pages:
                            all_docs.append({
                                'url': page['url'],
                                'title': page.get('title', ''),
                                'content': page.get('content', ''),
                            })
                    
                    if not all_docs:
                        logger.warning(f"Не удалось загрузить ни одной страницы для темы {topic}")
                        if graph and hasattr(graph, 'increment_failed_attempts'):
                            await graph.increment_failed_attempts(topic)
                        continue
                    
                    # Анализ (с флагом is_discovery)
                    analysis = await analyst.analyze(all_docs, query=topic, is_discovery=True)
                    key_points = analysis.get('key_points', [])
                    confidence = analysis.get('confidence', 0.0)
                    
                    # Если фактов нет, увеличиваем счётчик неудач
                    if len(key_points) == 0:
                        if graph and hasattr(graph, 'increment_failed_attempts'):
                            await graph.increment_failed_attempts(topic)
                        logger.info(f"⚠️ Тема '{topic}' не дала фактов, счётчик неудач увеличен")
                        continue
                    
                    # Верификация комитетом (опционально, можно пропустить для ускорения)
                    #if key_points and committee:
                    #   committee_result = await committee.evaluate({
                    #       'url': all_docs[0].get('url', ''),
                    #       'title': topic,
                    #       'content': key_points[0],
                    #        'snippet': key_points[0]
                    #    })
                    #    if not committee_result.get('approved', False):
                    #        logger.info(f"Тема {topic} отклонена комитетом")
                    #        continue
                    
                    # Сохраняем знания
                    await self._store_knowledge(topic, {
                        'summary': ' '.join(key_points[:2]),
                        'key_points': key_points,
                        'confidence': confidence,
                        'source': 'discovery_cycle'
                    })
                    
                    results['research_completed'].append({
                        'topic': topic,
                        'pages': len(all_docs),
                        'key_points': len(key_points),
                        'confidence': confidence
                    })
                    
                    logger.info(f"✅ Тема '{topic}' изучена и сохранена (фактов: {len(key_points)})")
                    
                except Exception as e:
                    error_msg = f"Ошибка при обработке темы {topic}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            success = len(results['research_completed']) > 0
            duration = time.time() - start_time
            self._update_stats(success, duration)
            
            results['end_time'] = datetime.now().isoformat()
            results['duration_seconds'] = duration
            results['success'] = success
            
            return results
            
        except Exception as e:
            logger.error(f"Критическая ошибка в DiscoveryCycle: {e}", exc_info=True)
            results['errors'].append(str(e))
            results['success'] = False
            self._update_stats(False, time.time() - start_time)
            return results
    
    async def _discover_potential_topics(self) -> List[Tuple[str, float]]:
        """
        Получает потенциальные новые темы с приоритетами.
        Использует:
        - Слабые темы из графа (мало связей)
        - Старые темы из графа (давно не обновлялись)
        - Из Engram темы с низкой уверенностью
        """
        topics = []
        graph = self.services.get('graph_db')
        engram = self.services.get('engram')
        
        # 1. Слабые темы из графа (мало связей)
        if graph and hasattr(graph, 'get_weak_topics'):
            try:
                weak = await graph.get_weak_topics(limit=5)
                for t in weak:
                    topics.append((t, 0.8))
                    logger.debug(f"   Слабая тема из графа: {t}")
            except Exception as e:
                logger.warning(f"Не удалось получить слабые темы из графа: {e}")
        
        # 2. Старые темы из графа (давно не обновлялись)
        if graph and hasattr(graph, 'get_old_topics'):
            try:
                old = await graph.get_old_topics(days_threshold=7, limit=5)
                for t in old:
                    topics.append((t, 0.6))
                    logger.debug(f"   Старая тема из графа: {t}")
            except Exception as e:
                logger.warning(f"Не удалось получить старые темы из графа: {e}")
        
        # 3. Из Engram темы с низкой уверенностью (confidence < 0.6)
        if engram and hasattr(engram, 'get_all_keys') and hasattr(engram, 'cache'):
            try:
                keys = await engram.get_all_keys()
                for key in keys:
                    record = engram.cache.get(key)
                    if record:
                        conf = record.get('metadata', {}).get('confidence', 1.0)
                        if conf < 0.6:
                            topics.append((key, 0.5))
                            logger.debug(f"   Низкая уверенность в Engram: {key} (conf={conf})")
            except Exception as e:
                logger.warning(f"Не удалось получить темы из Engram: {e}")
        
        # Если ничего не найдено, возвращаем пустой список
        if not topics:
            return []
        
        # Убираем дубликаты, сортируем по убыванию приоритета
        unique = {}
        for t, p in topics:
            if t not in unique or p > unique[t]:
                unique[t] = p
        sorted_topics = sorted(unique.items(), key=lambda x: x[1], reverse=True)
        return sorted_topics
    
    async def _store_knowledge(self, topic: str, knowledge: Dict):
        """Сохраняет знания во все доступные хранилища."""
        tasks = []
        engram = self.services.get('engram')
        chroma = self.services.get('chroma_db')
        graph = self.services.get('graph_db')
        
        if engram:
            tasks.append(engram.store(
                key=topic,
                content=knowledge.get('summary', ''),
                metadata={
                    'topic': topic,
                    'confidence': knowledge.get('confidence', 0.5),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'discovery_cycle',
                    'key_points': knowledge.get('key_points', [])[:5]
                },
                confidence=knowledge.get('confidence', 0.5)
            ))
        
        if chroma and knowledge.get('key_points'):
            for i, point in enumerate(knowledge['key_points'][:5]):
                if len(point) > 50:
                    tasks.append(chroma.add_document(
                        text=point,
                        metadata={
                            'topic': topic,
                            'type': 'key_point',
                            'source': 'discovery_cycle',
                            'confidence': knowledge.get('confidence', 0.5),
                            'index': i
                        }
                    ))
        
        if graph and hasattr(graph, 'add_knowledge_chunk'):
            tasks.append(graph.add_knowledge_chunk(
                topic=topic,
                chunk={
                    'summary': knowledge.get('summary', '')[:500],
                    'key_points': knowledge.get('key_points', [])[:5],
                    'confidence': knowledge.get('confidence', 0.5)
                },
                relations=[]
            ))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"Знания по теме '{topic}' сохранены в {len(tasks)} хранилищ")


class DeepeningCycle(LearningCycle):
    """
    Цикл углубления в уже существующие темы.
    Выбирает тему с хорошим покрытием, генерирует уточняющие вопросы,
    ищет дополнительную информацию, обновляет хранилища.
    """
    
    def __init__(self, services: Dict[str, Any], config: Dict = None):
        super().__init__(CycleType.DEEPENING, services)
        self.config = config or {}
        self.depth = self.config.get('depth', 2)  # количество итераций углубления
    
    async def execute(self) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("🚀 Запуск цикла DEEPENING")
        
        results = {
            'cycle_type': self.cycle_type.value,
            'start_time': datetime.now().isoformat(),
            'deepened_topics': [],
            'errors': []
        }
        
        try:
            # Получаем список существующих тем из хранилищ
            existing_topics = await self._get_existing_topics()
            if not existing_topics:
                logger.warning("Нет существующих тем для углубления")
                self._update_stats(False, time.time() - start_time)
                results['success'] = False
                return results
            
            # Выбираем тему для углубления (случайную из топ-10)
            import random
            topic = random.choice(existing_topics[:10])
            logger.info(f"Выбрана тема для углубления: {topic}")
            
            interviewer = self.services.get('interviewer')
            detective = self.services.get('detective')
            analyst = self.services.get('analyst')
            committee = self.services.get('committee')
            
            all_key_points = []
            current_depth = 0
            while current_depth < self.depth:
                # Генерируем углубляющие вопросы
                if current_depth == 0:
                    questions = [f"Подробнее о {topic}"]
                else:
                    # Используем накопленные знания для генерации вопросов
                    questions = await interviewer.generate_deepening_questions(
                        knowledge_chunks=[{'text': ' '.join(all_key_points[-5:])}] if all_key_points else None,
                        current_depth=current_depth,
                        max_questions=3
                    )
                
                if not questions:
                    questions = [f"Какие существуют продвинутые аспекты темы {topic}?"]
                
                results.setdefault('deepened_topics', []).append({
                    'topic': topic,
                    'depth_level': current_depth,
                    'questions': questions
                })
                
                # Исследуем
                investigation = await detective.investigate_topic_advanced(topic, questions)
                if not investigation.get('success'):
                    raise RuntimeError(f"Ошибка исследования: {investigation.get('error')}")
                
                chunks = investigation.get('content_chunks', [])
                if chunks:
                    # Используем первый сгенерированный вопрос как запрос для анализа (более релевантно)
                    analysis_query = questions[0] if questions else topic
                    analysis = await analyst.analyze(chunks, query=analysis_query, is_discovery=True)
                    key_points = analysis.get('key_points', [])
                    confidence = analysis.get('confidence', 0.5)
                    
                    if key_points:
                        approved_count = 0
                        for sample in key_points[:3]:
                            committee_result = await committee.evaluate({
                                'url': investigation.get('metadata', [{}])[0].get('url', ''),
                                'title': topic,
                                'content': sample,
                                'snippet': sample
                            })
                            if committee_result.get('approved', False):
                                approved_count += 1
                        if approved_count < 2:  # меньше двух одобрено – отклоняем
                            logger.info(f"Тема {topic} отклонена комитетом (одобрено {approved_count}/3)")

                            logger.info(f"Новые данные по теме {topic} (глубина {current_depth}) отклонены комитетом")
                            continue
                        
                    all_key_points.extend(key_points)
                    logger.info(f"   ✅ Добавлено {len(key_points)} новых фактов (всего {len(all_key_points)})")
                current_depth += 1
            
            if all_key_points:
                # Сохраняем обновлённые знания
                await self._store_knowledge(topic, {
                    'summary': ' '.join(all_key_points[:2]),
                    'key_points': all_key_points,
                    'confidence': confidence if 'confidence' in locals() else 0.5,
                    'source': 'deepening_cycle',
                    'depth_achieved': self.depth
                })
                results['deepened_topics'][-1]['key_points_added'] = len(all_key_points)
                logger.info(f"✅ Тема '{topic}' углублена: добавлено {len(all_key_points)} новых фактов")
            
            success = len(all_key_points) > 0
            duration = time.time() - start_time
            self._update_stats(success, duration)
            
            results['end_time'] = datetime.now().isoformat()
            results['duration_seconds'] = duration
            results['success'] = success
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в DeepeningCycle: {e}", exc_info=True)
            results['errors'].append(str(e))
            results['success'] = False
            self._update_stats(False, time.time() - start_time)
            return results
    
    async def _get_existing_topics(self) -> List[str]:
        """Получает список существующих тем из хранилищ."""
        topics = set()
        graph = self.services.get('graph_db')
        if graph and hasattr(graph, 'get_all_topics'):
            try:
                topics.update(await graph.get_all_topics())
            except Exception as e:
                logger.warning(f"Ошибка получения тем из графа: {e}")
        
        engram = self.services.get('engram')
        if engram and hasattr(engram, 'get_all_keys'):
            try:
                topics.update(await engram.get_all_keys())
            except Exception as e:
                logger.warning(f"Ошибка получения тем из Engram: {e}")
        
        return list(topics)
    
    async def _store_knowledge(self, topic: str, knowledge: Dict):
        """Аналогично DiscoveryCycle."""
        tasks = []
        engram = self.services.get('engram')
        chroma = self.services.get('chroma_db')
        graph = self.services.get('graph_db')
        
        if engram:
            tasks.append(engram.store(
                key=topic,
                content=knowledge.get('summary', ''),
                metadata={
                    'topic': topic,
                    'confidence': knowledge.get('confidence', 0.5),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'deepening_cycle',
                    'key_points': knowledge.get('key_points', [])[:5],
                    'depth': knowledge.get('depth_achieved', 1)
                },
                confidence=knowledge.get('confidence', 0.5)
            ))
        
        if chroma and knowledge.get('key_points'):
            for i, point in enumerate(knowledge['key_points'][:5]):
                if len(point) > 50:
                    tasks.append(chroma.add_document(
                        text=point,
                        metadata={
                            'topic': topic,
                            'type': 'key_point',
                            'source': 'deepening_cycle',
                            'confidence': knowledge.get('confidence', 0.5),
                            'index': i
                        }
                    ))
        
        if graph and hasattr(graph, 'add_knowledge_chunk'):
            tasks.append(graph.add_knowledge_chunk(
                topic=topic,
                chunk={
                    'summary': knowledge.get('summary', '')[:500],
                    'key_points': knowledge.get('key_points', [])[:5],
                    'confidence': knowledge.get('confidence', 0.5)
                },
                relations=[]
            ))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class ExpansionCycle(LearningCycle):
    """
    Цикл расширения связей между темами.
    Выбирает две темы, ищет информацию об их взаимосвязи и сохраняет связи в граф.
    """
    
    def __init__(self, services: Dict[str, Any], config: Dict = None):
        super().__init__(CycleType.EXPANSION, services)
        self.config = config or {}
    
    async def execute(self) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("🚀 Запуск цикла EXPANSION")
        
        results = {
            'cycle_type': self.cycle_type.value,
            'start_time': datetime.now().isoformat(),
            'connections': [],
            'errors': []
        }
        
        try:
            topics = await self._get_existing_topics()
            if len(topics) < 2:
                logger.warning("Недостаточно тем для расширения связей")
                self._update_stats(False, time.time() - start_time)
                results['success'] = False
                return results
            
            import random
            topic1 = random.choice(topics)
            topic2 = random.choice([t for t in topics if t != topic1])
            
            logger.info(f"Исследование связи между '{topic1}' и '{topic2}'")
            
            detective = self.services.get('detective')
            analyst = self.services.get('analyst')
            graph = self.services.get('graph_db')
            
            # Формируем запрос о связи
            query = f"Связь между {topic1} и {topic2}"
            investigation = await detective.investigate_topic_advanced(query, [query])
            if not investigation.get('success'):
                raise RuntimeError("Не удалось исследовать связь")
            
            chunks = investigation.get('content_chunks', [])
            if chunks:
                analysis = await analyst.analyze(chunks, query=query, is_discovery=True)
                key_points = analysis.get('key_points', [])
                confidence = analysis.get('confidence', 0.5)
                
                if key_points and graph and hasattr(graph, 'add_relation'):
                    # Сохраняем связь в граф как отношение
                    await graph.add_relation(topic1, topic2, relation_type="связано_с", weight=confidence)
                    # Также можно сохранить пояснение
                    if hasattr(graph, 'add_knowledge_chunk'):
                        await graph.add_knowledge_chunk(
                            topic=f"{topic1}_{topic2}",
                            chunk={
                                'summary': ' '.join(key_points[:2]),
                                'key_points': key_points[:5],
                                'confidence': confidence
                            },
                            relations=[(topic1, topic2, "связано_с")]
                        )
                    
                    results['connections'].append({
                        'topic1': topic1,
                        'topic2': topic2,
                        'key_points': len(key_points),
                        'confidence': confidence
                    })
                    logger.info(f"   ✅ Создана связь между '{topic1}' и '{topic2}' (уверенность {confidence:.2f})")
            
            success = len(results['connections']) > 0
            duration = time.time() - start_time
            self._update_stats(success, duration)
            
            results['end_time'] = datetime.now().isoformat()
            results['duration_seconds'] = duration
            results['success'] = success
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в ExpansionCycle: {e}", exc_info=True)
            results['errors'].append(str(e))
            results['success'] = False
            self._update_stats(False, time.time() - start_time)
            return results
    
    async def _get_existing_topics(self) -> List[str]:
        """Получает список существующих тем из хранилищ."""
        topics = set()
        graph = self.services.get('graph_db')
        if graph and hasattr(graph, 'get_all_topics'):
            try:
                topics.update(await graph.get_all_topics())
            except Exception as e:
                logger.warning(f"Ошибка получения тем из графа: {e}")
        
        engram = self.services.get('engram')
        if engram and hasattr(engram, 'get_all_keys'):
            try:
                topics.update(await engram.get_all_keys())
            except Exception as e:
                logger.warning(f"Ошибка получения тем из Engram: {e}")
        
        return list(topics)


class MetaAnalysisCycle(LearningCycle):
    """
    Цикл мета-анализа системы.
    Собирает статистику со всех компонентов, анализирует эффективность,
    выдаёт рекомендации по настройке параметров.
    """
    
    def __init__(self, services: Dict[str, Any], config: Dict = None):
        super().__init__(CycleType.META_ANALYSIS, services)
        self.config = config or {}
    
    async def execute(self) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("🚀 Запуск цикла META_ANALYSIS")
        
        results = {
            'cycle_type': self.cycle_type.value,
            'start_time': datetime.now().isoformat(),
            'analysis': {},
            'adjustments': [],
            'errors': []
        }
        
        try:
            # Сбор метрик со всех сервисов
            stats = {}
            for name, service in self.services.items():
                if service and hasattr(service, 'get_metrics'):
                    try:
                        stats[name] = await service.get_metrics()
                    except Exception as e:
                        logger.warning(f"Не удалось получить метрики от {name}: {e}")
            
            # Анализ (пример)
            analysis = {
                'total_knowledge_chunks': stats.get('chroma_db', {}).get('collection_size', 0),
                'graph_nodes': stats.get('graph_db', {}).get('nodes', 0),
                'graph_edges': stats.get('graph_db', {}).get('edges', 0),
                'engram_entries': stats.get('engram', {}).get('total_records', 0),
                'detective_requests': stats.get('detective', {}).get('requests', 0),
                'analyst_confidence_avg': stats.get('analyst', {}).get('avg_confidence', 0),
            }
            
            # Выработка рекомендаций
            adjustments = []
            if analysis['total_knowledge_chunks'] < 100:
                adjustments.append("Увеличить приоритет цикла DISCOVERY")
            if analysis['graph_edges'] < analysis['graph_nodes'] * 0.5:
                adjustments.append("Увеличить приоритет цикла EXPANSION")
            if analysis['analyst_confidence_avg'] < 0.4:
                adjustments.append("Проверить качество источников, возможно, улучшить фильтрацию комитета")
            
            results['analysis'] = analysis
            results['adjustments'] = adjustments
            
            logger.info(f"📊 Мета-анализ: узлов={analysis['graph_nodes']}, связей={analysis['graph_edges']}, Engram={analysis['engram_entries']}")
            if adjustments:
                logger.info(f"💡 Рекомендации: {', '.join(adjustments)}")
            
            success = True
            duration = time.time() - start_time
            self._update_stats(success, duration)
            
            results['end_time'] = datetime.now().isoformat()
            results['duration_seconds'] = duration
            results['success'] = success
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в MetaAnalysisCycle: {e}", exc_info=True)
            results['errors'].append(str(e))
            results['success'] = False
            self._update_stats(False, time.time() - start_time)
            return results


class MaintenanceCycle(LearningCycle):
    """
    Цикл обслуживания системы.
    Очистка кэшей, оптимизация графа, перестроение индексов.
    """
    
    def __init__(self, services: Dict[str, Any], config: Dict = None):
        super().__init__(CycleType.MAINTENANCE, services)
        self.config = config or {}
    
    async def execute(self) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("🚀 Запуск цикла MAINTENANCE")
        
        results = {
            'cycle_type': self.cycle_type.value,
            'start_time': datetime.now().isoformat(),
            'operations': {},
            'errors': []
        }
        
        try:
            # Очистка кэша детектива
            if self.services.get('detective'):
                try:
                    await self.services['detective'].clear_cache()
                    results['operations']['detective_cache_cleared'] = True
                except Exception as e:
                    results['errors'].append(f"detective.clear_cache: {e}")
            
            # Оптимизация графа
            if self.services.get('graph_db') and hasattr(self.services['graph_db'], 'optimize'):
                try:
                    opt_result = await self.services['graph_db'].optimize()
                    results['operations']['graph_optimized'] = opt_result
                except Exception as e:
                    results['errors'].append(f"graph.optimize: {e}")
            
            # Очистка старых записей в Engram (если есть метод cleanup)
            if self.services.get('engram') and hasattr(self.services['engram'], 'cleanup'):
                try:
                    await self.services['engram'].cleanup()
                    results['operations']['engram_cleaned'] = True
                except Exception as e:
                    results['errors'].append(f"engram.cleanup: {e}")
            
            # Перестроение индексов Chroma (если есть метод optimize)
            if self.services.get('chroma_db') and hasattr(self.services['chroma_db'], 'optimize'):
                try:
                    await self.services['chroma_db'].optimize()
                    results['operations']['chroma_optimized'] = True
                except Exception as e:
                    results['errors'].append(f"chroma.optimize: {e}")
            
            success = True
            duration = time.time() - start_time
            self._update_stats(success, duration)
            
            results['end_time'] = datetime.now().isoformat()
            results['duration_seconds'] = duration
            results['success'] = success
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в MaintenanceCycle: {e}", exc_info=True)
            results['errors'].append(str(e))
            results['success'] = False
            self._update_stats(False, time.time() - start_time)
            return results


class LearningCycleCoordinator:
    """
    Координатор всех циклов самообучения.
    Управляет расписанием, выбирает циклы на основе приоритетов,
    запускает их в фоновом режиме.
    """
    
    def __init__(self, services: Dict[str, Any], config: Optional[Dict] = None):
        self.services = services
        self.config = config or {}
        
        # Флаг включения/выключения
        self.enabled = self.config.get('enabled', True)
        
        # Инициализация доступных циклов с их конфигурацией
        cycle_configs = self.config.get('cycles', {})
        self.cycles = {
            CycleType.DISCOVERY: DiscoveryCycle(services, cycle_configs.get('discovery', {})),
            CycleType.DEEPENING: DeepeningCycle(services, cycle_configs.get('deepening', {})),
            CycleType.EXPANSION: ExpansionCycle(services, cycle_configs.get('expansion', {})),
            CycleType.META_ANALYSIS: MetaAnalysisCycle(services, cycle_configs.get('meta', {})),
            CycleType.MAINTENANCE: MaintenanceCycle(services, cycle_configs.get('maintenance', {})),
        }
        
        # Приоритеты циклов (сумма должна быть 1, но не обязательно)
        self.cycle_priorities = self.config.get('priorities', {
            CycleType.DISCOVERY.value: 0.40,
            CycleType.DEEPENING.value: 0.30,
            CycleType.EXPANSION.value: 0.20,
            CycleType.META_ANALYSIS.value: 0.07,
            CycleType.MAINTENANCE.value: 0.03,
        })
        
        # Интервалы выполнения (минимальное время между запусками одного цикла)
        self.schedule_intervals = {
            CycleType.DISCOVERY: timedelta(seconds=self.config.get('intervals', {}).get('discovery', 3600)),
            CycleType.DEEPENING: timedelta(seconds=self.config.get('intervals', {}).get('deepening', 7200)),
            CycleType.EXPANSION: timedelta(seconds=self.config.get('intervals', {}).get('expansion', 14400)),
            CycleType.META_ANALYSIS: timedelta(seconds=self.config.get('intervals', {}).get('meta_analysis', 86400)),
            CycleType.MAINTENANCE: timedelta(seconds=self.config.get('intervals', {}).get('maintenance', 43200)),
        }
        
        self.last_execution = {ctype: None for ctype in CycleType}
        self.execution_history = []
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info("LearningCycleCoordinator инициализирован. enabled=%s", self.enabled)
    
    async def start(self):
        logger.info("🔥 LearningCycleCoordinator.start() вызван")
        logger.info(f"   enabled={self.enabled}, is_running={self.is_running}")
        """Запускает основной цикл координации."""
        if not self.enabled:
            logger.info("Самообучение отключено в конфигурации")
            return
        
        if self.is_running:
            logger.warning("Координатор уже запущен")
            return
        
        self.is_running = True
        logger.info("Запуск LearningCycleCoordinator")
        
        while self.is_running:
            try:
                cycle_type = self._select_cycle_to_run()
                if cycle_type:
                    logger.info(f"Выбран цикл: {cycle_type.value}")
                    cycle = self.cycles[cycle_type]
                    result = await cycle.execute()
                    
                    self.last_execution[cycle_type] = datetime.now()
                    self.execution_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'cycle': cycle_type.value,
                        'result': result,
                        'duration': result.get('duration_seconds', 0)
                    })
                    
                    # Адаптация приоритетов на основе результата
                    self._adapt_priorities(cycle_type, result)
                    
                    # Ограничим историю
                    if len(self.execution_history) > 1000:
                        self.execution_history = self.execution_history[-500:]
                
                # Пауза перед следующей проверкой (можно настраивать)
                await asyncio.sleep(self.config.get('check_interval', 60))
                
            except asyncio.CancelledError:
                logger.info("Координатор остановлен по запросу")
                break
            except Exception as e:
                logger.error(f"Ошибка в цикле координатора: {e}", exc_info=True)
                await asyncio.sleep(300)  # при ошибке ждём 5 минут
    
    def stop(self):
        """Останавливает координатор."""
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("LearningCycleCoordinator остановлен")
    
    async def run_cycle(self, cycle_type: CycleType) -> Dict[str, Any]:
        """Запускает конкретный цикл по требованию (синхронно-асинхронно)."""
        if cycle_type not in self.cycles:
            return {'error': f'Неизвестный тип цикла: {cycle_type}'}
        
        logger.info(f"Запуск цикла {cycle_type.value} по требованию")
        cycle = self.cycles[cycle_type]
        result = await cycle.execute()
        self.last_execution[cycle_type] = datetime.now()
        return {
            'cycle': cycle_type.value,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    def _select_cycle_to_run(self) -> Optional[CycleType]:
        """Выбирает цикл для выполнения на основе приоритетов и времени последнего запуска."""
        now = datetime.now()
        candidates = []
        
        for cycle_type, interval in self.schedule_intervals.items():
            last = self.last_execution[cycle_type]
            if last is None or (now - last) >= interval:
                # Цикл готов к запуску
                priority = self.cycle_priorities.get(cycle_type.value, 0)
                candidates.append((cycle_type, priority))
        
        if not candidates:
            return None
        
        # Нормализация вероятностей
        total = sum(p for _, p in candidates)
        if total == 0:
            return None
        
        # Выбор случайного с учётом весов
        r = random.random()
        cumulative = 0.0
        for cycle, prob in candidates:
            cumulative += prob / total
            if r <= cumulative:
                return cycle
        
        return candidates[-1][0]  # запасной вариант
    
    def _adapt_priorities(self, cycle_type: CycleType, result: Dict[str, Any]):
        """Адаптирует приоритеты на основе успешности выполнения."""
        success = result.get('success', False)
        if success:
            # Увеличиваем приоритет успешного цикла
            self.cycle_priorities[cycle_type.value] *= 1.1
        else:
            # Уменьшаем приоритет неудачного
            self.cycle_priorities[cycle_type.value] *= 0.9
        
        # Нормализация (чтобы сумма не уходила в бесконечность)
        total = sum(self.cycle_priorities.values())
        for ctype in self.cycle_priorities:
            self.cycle_priorities[ctype] /= total
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику координатора и всех циклов."""
        cycle_stats = {ctype.value: self.cycles[ctype].get_stats() for ctype in self.cycles}
        return {
            'enabled': self.enabled,
            'is_running': self.is_running,
            'cycle_priorities': self.cycle_priorities.copy(),
            'last_executions': {k.value: v.isoformat() if v else None for k, v in self.last_execution.items()},
            'cycle_stats': cycle_stats,
            'total_executions': len(self.execution_history)
        }


# Для удобного использования можно создать глобальный экземпляр
_learning_coordinator: Optional[LearningCycleCoordinator] = None

def get_learning_coordinator(services: Optional[Dict[str, Any]] = None, config: Optional[Dict] = None) -> LearningCycleCoordinator:
    """Возвращает глобальный экземпляр координатора (создаёт при необходимости)."""
    global _learning_coordinator
    if _learning_coordinator is None:
        if services is None:
            raise RuntimeError("Необходимо передать словарь сервисов при первом вызове")
        _learning_coordinator = LearningCycleCoordinator(services, config)
    return _learning_coordinator