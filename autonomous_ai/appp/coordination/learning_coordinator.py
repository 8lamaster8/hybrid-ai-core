"""
🔄 Координатор циклов самообучения для автономной AI-системы.
Управляет различными типами циклов обучения, адаптирует приоритеты,
использует компоненты системы (детектив, аналитик, интервьюер, комитет, хранилища).
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

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
    
    def __init__(self, services: Dict[str, Any]):
        super().__init__(CycleType.DISCOVERY, services)
        self.min_topics = 3
        self.max_topics = 10
    
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
            
            # 1. Получаем новые темы от интервьюера (например, из анализа неохваченных областей)
            # В реальности здесь может быть более сложная логика; пока используем заглушку.
            new_topics = await self._discover_potential_topics()
            results['discovered_topics'] = [{'name': t, 'priority': 0.5} for t in new_topics[:self.max_topics]]
            
            if not new_topics:
                logger.info("Новых тем не обнаружено")
                self._update_stats(success=False, duration=time.time() - start_time)
                results['success'] = False
                return results
            
            # 2. Берём несколько тем с наивысшим приоритетом
            topics_to_research = new_topics[:self.min_topics]
            
            for topic in topics_to_research:
                try:
                    # Генерация исследовательских вопросов
                    questions = await interviewer.generate_research_questions(topic, depth=1, num_questions=5)
                    
                    # Исследование через детектив
                    investigation = await detective.investigate_topic_advanced(topic, questions[:3])
                    if not investigation.get('success'):
                        logger.warning(f"Не удалось исследовать тему {topic}")
                        continue
                    
                    chunks = investigation.get('content_chunks', [])
                    if not chunks:
                        continue
                    
                    # Анализ
                    analysis = await analyst.analyze(chunks, query=topic)
                    key_points = analysis.get('key_points', [])
                    confidence = analysis.get('confidence', 0.0)
                    
                    # Верификация комитетом (на основе текста)
                    if key_points:
                        # Берём первый ключевой пункт как образец для оценки
                        committee_result = await committee.evaluate_data({
                            'topic': topic,
                            'text': key_points[0],
                            'url': investigation.get('metadata', [{}])[0].get('url', '')
                        })
                        if committee_result.get('final_decision', {}).get('decision') != 'approve':
                            logger.info(f"Тема {topic} отклонена комитетом")
                            continue
                    
                    # Сохраняем знания
                    await self._store_knowledge(topic, {
                        'summary': ' '.join(key_points[:2]),
                        'key_points': key_points,
                        'confidence': confidence,
                        'source': 'discovery_cycle'
                    })
                    
                    results['research_completed'].append({
                        'topic': topic,
                        'chunks': len(chunks),
                        'key_points': len(key_points),
                        'confidence': confidence
                    })
                    
                    logger.info(f"✅ Тема '{topic}' изучена и сохранена")
                    
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
    
    async def _discover_potential_topics(self) -> List[str]:
        """
        Получает потенциальные новые темы.
        В реальности можно использовать:
        - Анализ непокрытых областей из графа знаний.
        - Случайные темы из внешнего источника (например, популярные запросы).
        - Из Engram темы с низкой уверенностью.
        """
        # Пока возвращаем список общих тем
        return [
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
        
        if graph:
            # В граф можно сохранить как узел с отношениями (пока без отношений)
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
            
            # Выбираем тему для углубления (например, случайную из топ-10)
            import random
            topic = random.choice(existing_topics[:10])
            logger.info(f"Выбрана тема для углубления: {topic}")
            
            interviewer = self.services.get('interviewer')
            detective = self.services.get('detective')
            analyst = self.services.get('analyst')
            committee = self.services.get('committee')
            
            # Генерируем углубляющие вопросы (более специфичные)
            questions = await interviewer.generate_deepening_questions(
                knowledge_chunks=None,  # можно передать предыдущие знания
                current_depth=1,
                max_questions=3
            )
            if not questions:
                questions = [f"Какие существуют продвинутые аспекты темы {topic}?"]
            
            results['deepened_topics'].append({
                'topic': topic,
                'questions': questions
            })
            
            # Исследуем
            investigation = await detective.investigate_topic_advanced(topic, questions)
            if not investigation.get('success'):
                raise RuntimeError(f"Ошибка исследования: {investigation.get('error')}")
            
            chunks = investigation.get('content_chunks', [])
            if chunks:
                analysis = await analyst.analyze(chunks, query=topic)
                key_points = analysis.get('key_points', [])
                confidence = analysis.get('confidence', 0.5)
                
                if key_points:
                    # Верификация (опционально)
                    if committee:
                        sample_text = key_points[0]
                        committee_result = await committee.evaluate_data({
                            'topic': topic,
                            'text': sample_text,
                            'url': investigation.get('metadata', [{}])[0].get('url', '')
                        })
                        if committee_result.get('final_decision', {}).get('decision') != 'approve':
                            logger.info(f"Новые данные по теме {topic} отклонены комитетом")
                        else:
                            # Сохраняем
                            await self._store_knowledge(topic, {
                                'summary': ' '.join(key_points[:2]),
                                'key_points': key_points,
                                'confidence': confidence,
                                'source': 'deepening_cycle'
                            })
                            results['deepened_topics'][0]['key_points_added'] = len(key_points)
                    else:
                        # без комитета сохраняем
                        await self._store_knowledge(topic, {
                            'summary': ' '.join(key_points[:2]),
                            'key_points': key_points,
                            'confidence': confidence,
                            'source': 'deepening_cycle'
                        })
                        results['deepened_topics'][0]['key_points_added'] = len(key_points)
            
            success = True
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
            topics.update(await graph.get_all_topics())
        engram = self.services.get('engram')
        if engram and hasattr(engram, 'get_all_keys'):
            topics.update(await engram.get_all_keys())
        return list(topics)
    
    async def _store_knowledge(self, topic: str, knowledge: Dict):
        """Аналогично DiscoveryCycle._store_knowledge."""
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
                            'source': 'deepening_cycle',
                            'confidence': knowledge.get('confidence', 0.5),
                            'index': i
                        }
                    ))
        
        if graph:
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
                analysis = await analyst.analyze(chunks, query=query)
                key_points = analysis.get('key_points', [])
                confidence = analysis.get('confidence', 0.5)
                
                if key_points and graph:
                    # Сохраняем связь в граф как отношение
                    await graph.add_relation(topic1, topic2, relation_type="связано_с", weight=confidence)
                    # Также можно сохранить пояснение
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
        """Получает список существующих тем из графа или энграма."""
        topics = set()
        graph = self.services.get('graph_db')
        if graph and hasattr(graph, 'get_all_topics'):
            topics.update(await graph.get_all_topics())
        engram = self.services.get('engram')
        if engram and hasattr(engram, 'get_all_keys'):
            topics.update(await engram.get_all_keys())
        return list(topics)


class MetaAnalysisCycle(LearningCycle):
    """
    Цикл мета-анализа системы.
    Собирает статистику со всех компонентов, анализирует эффективность,
    выдаёт рекомендации по настройке параметров.
    """
    
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
                'total_knowledge_chunks': stats.get('chroma_db', {}).get('total_documents', 0),
                'graph_nodes': stats.get('graph_db', {}).get('nodes', 0),
                'graph_edges': stats.get('graph_db', {}).get('edges', 0),
                'engram_entries': stats.get('engram', {}).get('entries', 0),
                'detective_requests': stats.get('detective', {}).get('requests_processed', 0),
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


class LearningCycleCoordinator:
    """
    Координатор всех циклов самообучения.
    Управляет расписанием, выбирает циклы на основе приоритетов,
    запускает их в фоновом режиме.
    """
    
    def __init__(self, services: Dict[str, Any], config: Optional[Dict] = None):
        self.services = services
        self.config = config or {}
        
        # Инициализация доступных циклов
        self.cycles = {
            CycleType.DISCOVERY: DiscoveryCycle(services),
            CycleType.DEEPENING: DeepeningCycle(services),
            CycleType.EXPANSION: ExpansionCycle(services),
            CycleType.META_ANALYSIS: MetaAnalysisCycle(services),
            # MAINTENANCE можно добавить позже, если будет реализация
        }
        
        # Приоритеты циклов (сумма = 1)
        self.cycle_priorities = {
            CycleType.DISCOVERY: 0.40,
            CycleType.DEEPENING: 0.30,
            CycleType.EXPANSION: 0.20,
            CycleType.META_ANALYSIS: 0.10,
        }
        
        # Интервалы выполнения (если не используется адаптивное расписание)
        self.schedule_intervals = {
            CycleType.DISCOVERY: timedelta(hours=1),
            CycleType.DEEPENING: timedelta(hours=2),
            CycleType.EXPANSION: timedelta(hours=4),
            CycleType.META_ANALYSIS: timedelta(days=1),
        }
        
        self.last_execution = {ctype: None for ctype in CycleType}
        self.execution_history = []
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info("LearningCycleCoordinator инициализирован")
    
    async def start(self):
        """Запускает основной цикл координации."""
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
                await asyncio.sleep(60)  # проверка каждую минуту
                
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
                priority = self.cycle_priorities.get(cycle_type, 0)
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
            self.cycle_priorities[cycle_type] *= 1.1
        else:
            # Уменьшаем приоритет неудачного
            self.cycle_priorities[cycle_type] *= 0.9
        
        # Нормализация
        total = sum(self.cycle_priorities.values())
        for ctype in self.cycle_priorities:
            self.cycle_priorities[ctype] /= total
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику координатора и всех циклов."""
        cycle_stats = {ctype.value: self.cycles[ctype].get_stats() for ctype in self.cycles}
        return {
            'is_running': self.is_running,
            'cycle_priorities': {k.value: v for k, v in self.cycle_priorities.items()},
            'last_executions': {k.value: v.isoformat() if v else None for k, v in self.last_execution.items()},
            'cycle_stats': cycle_stats,
            'total_executions': len(self.execution_history)
        }


# Для удобного использования можно создать глобальный экземпляр
_learning_coordinator: Optional[LearningCycleCoordinator] = None

def get_learning_coordinator(services: Optional[Dict[str, Any]] = None) -> LearningCycleCoordinator:
    """Возвращает глобальный экземпляр координатора (создаёт при необходимости)."""
    global _learning_coordinator
    if _learning_coordinator is None:
        if services is None:
            raise RuntimeError("Необходимо передать словарь сервисов при первом вызове")
        _learning_coordinator = LearningCycleCoordinator(services)
    return _learning_coordinator