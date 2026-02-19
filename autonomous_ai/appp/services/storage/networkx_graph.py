"""
🕸️ NETWORKX ГРАФ ЗНАНИЙ
Хранение знаний в виде графа, сохранение в SQLite
"""

import asyncio
import json
import os
import sqlite3
import pickle
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import networkx as nx

from appp.core.logging import logger

from enum import Enum

class NodeType(Enum):
    TOPIC = "topic"
    CHUNK = "chunk"
    FACT = "fact"
    ENTITY = "entity"
    CONCEPT = "concept"
    QUESTION = "question"
    ANSWER = "answer"
    SOURCE = "source"


class RelationType(Enum):
    # Иерархические
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    
    # Семантические
    DEFINED_AS = "defined_as"
    EXAMPLE_OF = "example_of"
    CONTRASTS_WITH = "contrasts_with"
    SIMILAR_TO = "similar_to"
    
    # Причинно-следственные
    CAUSES = "causes"
    LEADS_TO = "leads_to"
    PREVENTS = "prevents"
    ENABLES = "enables"
    
    # Временные
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    
    # Атрибутивные
    HAS_PROPERTY = "has_property"
    HAS_VALUE = "has_value"
    LOCATED_IN = "located_in"
    CREATED_BY = "created_by"
    
    # Документальные
    MENTIONED_IN = "mentioned_in"
    SUPPORTED_BY = "supported_by"
    CONTRADICTED_BY = "contradicted_by"
    
    # Пространственные
    NEAR = "near"
    CONTAINS = "contains"
    BELONGS_TO = "belongs_to"

class NetworkXGraphService:
    """
    Граф знаний на основе NetworkX.
    - Хранение узлов (сущности, темы, чанки)
    - Связи (отношения)
    - Персистентность: сериализация в SQLite
    - Анализ: слабые места, смежные темы, старые знания
    """
    
    def __init__(
        self,
        db_path: str = "./data/graphs/knowledge_graph.db",
        auto_save: bool = True,
        save_interval: int = 60,
        enable_compression: bool = True
    ):
        self.db_path = db_path
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.enable_compression = enable_compression
        
        self.graph: nx.Graph = nx.Graph()
        self.node_metadata: Dict[str, Dict] = {}
        
        # Для автосохранения
        self.dirty = False
        self.save_task: Optional[asyncio.Task] = None
        
        # Статистика
        self.stats = {
            'nodes_added': 0,
            'edges_added': 0,
            'nodes_removed': 0,
            'edges_removed': 0,
            'queries': 0,
            'last_save_time': None,
            'last_save_size': 0,
            'errors': 0
        }
        
        logger.info(f"🕸️ NetworkXGraphService создан (db: {db_path})")
    
    async def initialize(self):
        """Инициализация: создание директории, загрузка данных"""
        logger.info("🔄 Инициализация графа знаний...")
        
        try:
            # Создаем директорию
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Загружаем данные из SQLite
            await self._load_from_db()
            
            # Запускаем задачу автосохранения
            if self.auto_save:
                self.save_task = asyncio.create_task(self._auto_save_loop())
            
            logger.info(f"✅ Граф знаний загружен: {self.graph.number_of_nodes()} узлов, "
                       f"{self.graph.number_of_edges()} связей")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации графа: {e}")
            return False


    def calculate_edge_weight(self, 
                            confidence: float,           # уверенность в факте (0-1)
                            source_importance: float,    # важность источника
                            temporal_decay: bool = True, # учёт времени
                            usage_count: int = 0) -> float:  # частота использования
        
        weight = confidence
        
        # Бонус за приоритетный домен
        if source_domain in self.priority_domains:
            weight += 0.2
        
        # Штраф за низкодоверенный домен
        if source_domain in self.low_trust_domains:
            weight -= 0.3
        
        # Бонус за частоту использования
        weight += min(0.5, usage_count * 0.1)
        
        # Временной распад (старые связи весят меньше)
        if temporal_decay and 'created_at' in data:
            days_old = (datetime.now() - created_date).days
            decay = max(0.5, 1.0 - days_old * 0.01)  # не меньше 0.5
            weight *= decay
        
        return max(0.1, min(2.0, weight))  # ограничиваем от 0.1 до 2.0
    
    def _node_type_to_str(self, node_type: NodeType) -> str:
        """Конвертирует NodeType в строку для хранения в графе"""
        return node_type.value

    def _str_to_node_type(self, type_str: str) -> NodeType:
        """Конвертирует строку в NodeType (с запасом на старые данные)"""
        try:
            return NodeType(type_str)
        except ValueError:
            # Если это старый тип, маппим на ближайший
            mapping = {
                'key_point': NodeType.FACT,
                'update': NodeType.CHUNK,
                'insight': NodeType.FACT,
                'topic': NodeType.TOPIC,
                'chunk': NodeType.CHUNK,
                'entity': NodeType.ENTITY
            }
            return mapping.get(type_str, NodeType.CONCEPT)

    def _relation_type_to_str(self, rel_type: RelationType) -> str:
        """Конвертирует RelationType в строку для хранения в графе"""
        return rel_type.value

    def _str_to_relation_type(self, type_str: str) -> RelationType:
        """Конвертирует строку в RelationType"""
        try:
            return RelationType(type_str)
        except ValueError:
            # Маппинг старых типов
            mapping = {
                'contains': RelationType.CONTAINS,
                'has_point': RelationType.HAS_PART,
                'has_insight': RelationType.HAS_PART,
                'updated': RelationType.SUPPORTED_BY,
                'contains_entity': RelationType.CONTAINS,
                'related': RelationType.SIMILAR_TO
            }
            return mapping.get(type_str, RelationType.SIMILAR_TO)


    async def add_knowledge_chunk(
        self,
        topic: str,
        chunk: Dict,
        relations: List[Dict] = None
    ):
        """
        Добавление чанка знаний в граф с умными весами.
        
        Args:
            topic: Тема (корневой узел)
            chunk: Чанк с текстом и метаданными
            relations: Дополнительные отношения
        """
        chunk_id = chunk.get('chunk_id', f"chunk_{datetime.now().timestamp()}")
        
        # Добавляем узел чанка
        self.graph.add_node(chunk_id, type='chunk', topic=topic, **chunk)
        self.node_metadata[chunk_id] = {
            'created_at': datetime.now().isoformat(),
            'type': 'chunk',
            'topic': topic
        }
        self.stats['nodes_added'] += 1
        
        # Добавляем узел темы (если нет)
        topic_id = f"topic_{topic}"
        if not self.graph.has_node(topic_id):
            self.graph.add_node(topic_id, type='topic', name=topic)
            self.node_metadata[topic_id] = {
                'created_at': datetime.now().isoformat(),
                'type': 'topic',
                'name': topic
            }
            self.stats['nodes_added'] += 1
        
        # ========== ВЫЧИСЛЯЕМ УМНЫЙ ВЕС ==========
        
        # Берём confidence из чанка (если есть)
        confidence = chunk.get('confidence', 0.5)
        
        # Оцениваем качество по длине текста
        text_length = len(chunk.get('text', ''))
        if text_length > 2000:
            quality_bonus = 0.3
        elif text_length > 1000:
            quality_bonus = 0.2
        elif text_length > 500:
            quality_bonus = 0.1
        else:
            quality_bonus = 0.0
        
        # Базовая уверенность
        weight = confidence + quality_bonus
        
        # Бонус за наличие ключевых точек (если чанк уже проанализирован)
        if chunk.get('key_points'):
            weight += 0.2
        
        # Ограничиваем вес
        weight = max(0.3, min(1.5, weight))
        
        # ========== ДОБАВЛЯЕМ СВЯЗЬ С ВЕСОМ ==========
        
        self.graph.add_edge(
            topic_id, 
            chunk_id, 
            relation='contains', 
            weight=weight,
            confidence=confidence,
            quality_bonus=quality_bonus,
            created_at=datetime.now().isoformat(),
            text_length=text_length
        )
        self.stats['edges_added'] += 1
        
        logger.info(f"   🔗 Связь {topic} -> чанк, вес={weight:.2f} (confidence={confidence}, бонус={quality_bonus})")
        
        # Добавляем дополнительные отношения
        if relations:
            for rel in relations:
                await self._add_relation(rel, chunk_id)
        
        self.dirty = True


    async def update_edge_weight(self, source_id: str, target_id: str, relation_type: str = 'contains'):
        """
        Увеличивает вес ребра при использовании (для часто используемых связей).
        """
        for u, v, data in self.graph.edges(data=True):
            # Проверяем совпадение в обоих направлениях (граф неориентированный)
            if (u == source_id and v == target_id) or (u == target_id and v == source_id):
                if data.get('relation') == relation_type:
                    current_weight = data.get('weight', 1.0)
                    usage_count = data.get('usage_count', 0) + 1
                    
                    # Увеличиваем вес, но не больше 2.0
                    new_weight = min(2.0, current_weight + 0.1)
                    
                    # Обновляем
                    data['weight'] = new_weight
                    data['usage_count'] = usage_count
                    data['last_used'] = datetime.now().isoformat()
                    
                    logger.debug(f"📈 Вес связи увеличен: {new_weight:.2f} (использований: {usage_count})")
                    self.dirty = True
                    return True
        return False


    
    async def add_topic_knowledge(
        self,
        topic: str,
        knowledge: Dict,
        depth: int = 1
    ):
        """
        Добавление структурированного знания по теме.
        
        Args:
            topic: Тема
            knowledge: Синтезированное знание (ключевые точки, обзор)
            depth: Глубина исследования
        """
        topic_id = f"topic_{topic}"
        
        # Создаем или обновляем узел темы
        node_data = {
            'type': 'topic',
            'name': topic,
            'depth': depth,
            'overview': knowledge.get('overview', ''),
            'updated_at': datetime.now().isoformat()
        }
        
        if self.graph.has_node(topic_id):
            # Обновляем атрибуты
            self.graph.nodes[topic_id].update(node_data)
        else:
            self.graph.add_node(topic_id, **node_data)
            self.node_metadata[topic_id] = {
                'created_at': datetime.now().isoformat(),
                'type': 'topic',
                'name': topic
            }
            self.stats['nodes_added'] += 1
        
        # Добавляем ключевые точки как отдельные узлы
        key_points = knowledge.get('key_points', [])
        for i, point in enumerate(key_points):
            point_id = f"point_{topic}_{i}_{hash(point.get('content', '')) % 10000}"
            self.graph.add_node(
                point_id,
                type='key_point',
                content=point.get('content', ''),
                confidence=point.get('confidence', 0),
                importance=point.get('importance', 0.5),
                topic=topic
            )
            self.graph.add_edge(topic_id, point_id, relation='has_point')
            self.stats['nodes_added'] += 1
            self.stats['edges_added'] += 1
        
        self.dirty = True
    
    async def add_exploration_result(
        self,
        topic: str,
        exploration: Dict,
        depth_achieved: int
    ):
        """
        Добавление результата исследования с циклами координат.
        """
        topic_id = f"topic_{topic}"
        
        # Обновляем атрибуты темы
        if self.graph.has_node(topic_id):
            self.graph.nodes[topic_id]['exploration_depth'] = depth_achieved
            self.graph.nodes[topic_id]['last_explored'] = datetime.now().isoformat()
            self.graph.nodes[topic_id]['comprehensive_summary'] = exploration.get('comprehensive_summary', '')
        
        # Добавляем инсайты
        insights = exploration.get('key_insights', [])
        for i, insight in enumerate(insights):
            insight_id = f"insight_{topic}_{i}_{datetime.now().timestamp()}"
            self.graph.add_node(
                insight_id,
                type='insight',
                content=insight,
                topic=topic,
                depth=depth_achieved
            )
            self.graph.add_edge(topic_id, insight_id, relation='has_insight')
            self.stats['nodes_added'] += 1
            self.stats['edges_added'] += 1
        
        self.dirty = True
    
    async def update_topic_knowledge(
        self,
        topic: str,
        new_information: Dict,
        update_type: str = 'refresh'
    ):
        """
        Обновление существующего знания.
        """
        topic_id = f"topic_{topic}"
        
        if not self.graph.has_node(topic_id):
            logger.warning(f"Тема {topic} не найдена, создаем новую")
            await self.add_topic_knowledge(topic, new_information, depth=1)
            return
        
        # Обновляем метаданные
        self.graph.nodes[topic_id]['updated_at'] = datetime.now().isoformat()
        self.graph.nodes[topic_id]['update_count'] = self.graph.nodes[topic_id].get('update_count', 0) + 1
        
        # Добавляем новую информацию как отдельный чанк
        chunk_id = f"update_{topic}_{datetime.now().timestamp()}"
        self.graph.add_node(
            chunk_id,
            type='update',
            content=new_information.get('answer', ''),
            confidence=new_information.get('confidence', 0),
            timestamp=datetime.now().isoformat()
        )
        self.graph.add_edge(topic_id, chunk_id, relation='updated')
        
        self.stats['nodes_added'] += 1
        self.stats['edges_added'] += 1
        self.dirty = True
    
    # ========== НОВЫЕ МЕТОДЫ ДЛЯ RELATED-ТЕМ И САМООБУЧЕНИЯ ==========
    
    async def get_related_nodes(self, topic: str, max_nodes: int = 3) -> List[str]:
        """
        Получение узлов, связанных с данной темой.
        Используется для генерации сравнительных вопросов.
        
        Args:
            topic: Тема для поиска связей
            max_nodes: Максимальное количество возвращаемых узлов
        
        Returns:
            Список названий связанных тем
        """
        try:
            # Ищем узел темы
            topic_id = None
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get('type') == 'topic' and attrs.get('name', '').lower() == topic.lower():
                    topic_id = node
                    break
            
            if not topic_id:
                logger.debug(f"Тема '{topic}' не найдена в графе")
                return []
            
            # Собираем связанные узлы
            related = set()
            
            # Прямые соседи по графу
            for neighbor in self.graph.neighbors(topic_id):
                node_data = self.graph.nodes[neighbor]
                node_type = node_data.get('type', '')
                node_name = node_data.get('name', '')
                
                # Если сосед - тема, добавляем сразу
                if node_type == 'topic' and node_name:
                    related.add(node_name)
                
                # Если сосед - чанк или сущность, ищем через него другие темы
                elif node_type in ['chunk', 'entity', 'key_point']:
                    for n2 in self.graph.neighbors(neighbor):
                        if n2 != topic_id:
                            n2_data = self.graph.nodes[n2]
                            if n2_data.get('type') == 'topic':
                                n2_name = n2_data.get('name', '')
                                if n2_name:
                                    related.add(n2_name)
            
            # Если ничего не нашли, возвращаем общие темы для fallback
            if not related:
                fallback_map = {
                    'искусственный интеллект': ['нейросети', 'машинное обучение', 'глубокое обучение'],
                    'ии': ['нейросети', 'машинное обучение', 'глубокое обучение'],
                    'нейросети': ['искусственный интеллект', 'машинное обучение', 'обратное распространение'],
                    'машинное обучение': ['искусственный интеллект', 'нейросети', 'обучение с учителем'],
                }
                
                topic_lower = topic.lower()
                for key, values in fallback_map.items():
                    if key in topic_lower or topic_lower in key:
                        return values[:max_nodes]
            
            return list(related)[:max_nodes]
            
        except Exception as e:
            logger.error(f"Ошибка в get_related_nodes: {e}")
            return []

    async def get_weak_topics(self, min_connections: int = 2, limit: int = 5) -> List[str]:
        """
        Поиск слабо изученных тем (мало связей).
        Для самообучения.
        
        Args:
            min_connections: Минимальное количество связей для сильной темы
            limit: Максимальное количество возвращаемых тем
        
        Returns:
            Список названий слабых тем
        """
        try:
            weak = []
            
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get('type') == 'topic':
                    # Считаем степень узла (количество связей)
                    degree = self.graph.degree(node)
                    
                    # Получаем название темы
                    name = attrs.get('name', '')
                    if not name:
                        # Пробуем извлечь из node_id
                        if node.startswith('topic_'):
                            name = node[6:]
                        else:
                            continue
                    
                    if degree < min_connections:
                        weak.append({
                            'name': name,
                            'connections': degree,
                            'node': node
                        })
            
            # Сортируем по возрастанию связей (самые слабые первые)
            weak.sort(key=lambda x: x['connections'])
            
            return [w['name'] for w in weak[:limit]]
            
        except Exception as e:
            logger.error(f"Ошибка в get_weak_topics: {e}")
            return []

    async def get_old_topics(self, days_threshold: int = 7, limit: int = 5) -> List[str]:
        """
        Поиск давно не обновлявшихся тем.
        Для самообучения.
        
        Args:
            days_threshold: Количество дней без обновления
            limit: Максимальное количество возвращаемых тем
        
        Returns:
            Список названий старых тем
        """
        try:
            old = []
            cutoff = datetime.now() - timedelta(days=days_threshold)
            
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get('type') == 'topic':
                    # Получаем название темы
                    name = attrs.get('name', '')
                    if not name:
                        if node.startswith('topic_'):
                            name = node[6:]
                        else:
                            continue
                    
                    # Проверяем дату последнего обновления
                    updated = attrs.get('updated_at', attrs.get('created_at', '2000-01-01'))
                    try:
                        if isinstance(updated, str):
                            updated_date = datetime.fromisoformat(updated)
                        else:
                            updated_date = datetime(2000, 1, 1)
                        
                        if updated_date < cutoff:
                            old.append({
                                'name': name,
                                'last_updated': updated_date,
                                'days_old': (datetime.now() - updated_date).days
                            })
                    except:
                        # Если не можем распарсить дату, считаем старой
                        old.append({
                            'name': name,
                            'last_updated': datetime(2000, 1, 1),
                            'days_old': 999
                        })
            
            # Сортируем по убыванию старости
            old.sort(key=lambda x: x['days_old'], reverse=True)
            
            return [o['name'] for o in old[:limit]]
            
        except Exception as e:
            logger.error(f"Ошибка в get_old_topics: {e}")
            return []
    
    # ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========
    
    async def get_weak_knowledge_areas(self, limit: int = 5) -> List[str]:
        """
        Поиск слабо изученных областей (тем с малым количеством связей).
        """
        weak_areas = []
        
        # Узлы-темы
        topic_nodes = [
            n for n, attrs in self.graph.nodes(data=True)
            if attrs.get('type') == 'topic'
        ]
        
        for node in topic_nodes:
            # Количество исходящих связей
            out_degree = self.graph.degree(node)
            
            # Дата последнего обновления
            last_updated = self.graph.nodes[node].get('updated_at', '2000-01-01')
            last_date = datetime.fromisoformat(last_updated) if isinstance(last_updated, str) else datetime(2000,1,1)
            days_old = (datetime.now() - last_date).days
            
            # Вес слабости: маленькая степень + большая давность
            weakness_score = (1 / (out_degree + 1)) * 10 + days_old * 0.1
            
            weak_areas.append((weakness_score, self.graph.nodes[node].get('name', node)))
        
        # Сортируем по убыванию слабости
        weak_areas.sort(key=lambda x: x[0], reverse=True)
        
        return [name for score, name in weak_areas[:limit]]
    
    async def get_related_topics(self, seed_topics: List[str], limit: int = 3) -> List[str]:
        """
        Поиск смежных тем через общие чанки/сущности.
        """
        related = set()
        
        for seed in seed_topics:
            seed_id = f"topic_{seed}"
            if not self.graph.has_node(seed_id):
                continue
            
            # Находим все чанки, связанные с seed
            neighbors = list(self.graph.neighbors(seed_id))
            
            for n in neighbors:
                # Для каждого чанка смотрим, с какими еще темами он связан
                if self.graph.nodes[n].get('type') == 'chunk':
                    chunk_neighbors = list(self.graph.neighbors(n))
                    for cn in chunk_neighbors:
                        if cn != seed_id and self.graph.nodes[cn].get('type') == 'topic':
                            topic_name = self.graph.nodes[cn].get('name', cn)
                            related.add(topic_name)
        
        return list(related)[:limit]
    
    async def get_old_knowledge(self, days_old: int = 7, limit: int = 10) -> List[Dict]:
        """
        Получение тем, которые не обновлялись более N дней.
        """
        old_topics = []
        cutoff = datetime.now() - timedelta(days=days_old)
        
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == 'topic':
                updated = attrs.get('updated_at', attrs.get('created_at', '2000-01-01'))
                try:
                    updated_date = datetime.fromisoformat(updated)
                    if updated_date < cutoff:
                        old_topics.append({
                            'node': node,
                            'topic': attrs.get('name', node),
                            'last_updated': updated,
                            'days_old': (datetime.now() - updated_date).days
                        })
                except:
                    pass
        
        # Сортируем по старости
        old_topics.sort(key=lambda x: x['days_old'], reverse=True)
        
        return old_topics[:limit]
    
    async def optimize(self) -> Dict:
        """
        Оптимизация графа (сжатие, удаление изолированных узлов).
        """
        logger.info("🔄 Оптимизация графа знаний...")
        
        removed_nodes = 0
        removed_edges = 0
        
        # Удаляем изолированные узлы (без связей)
        isolated = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated)
        removed_nodes += len(isolated)
        
        # Удаляем дублирующиеся ребра
        loops = list(nx.selfloop_edges(self.graph))
        self.graph.remove_edges_from(loops)
        removed_edges += len(loops)
        
        self.stats['nodes_removed'] += removed_nodes
        self.stats['edges_removed'] += removed_edges
        
        # Сохраняем после оптимизации
        await self.save()
        
        return {
            'removed_isolated_nodes': removed_nodes,
            'removed_self_loops': removed_edges,
            'nodes_after': self.graph.number_of_nodes(),
            'edges_after': self.graph.number_of_edges()
        }
    
    async def analyze_structure(self) -> Dict:
        """
        Анализ структуры графа.
        """
        analysis = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
            'node_types': defaultdict(int),
            'relation_types': defaultdict(int)
        }
        
        # Типы узлов
        for _, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            analysis['node_types'][node_type] += 1
        
        # Типы связей
        for _, _, attrs in self.graph.edges(data=True):
            rel_type = attrs.get('relation', 'unknown')
            analysis['relation_types'][rel_type] += 1
        
        # Самые важные узлы (центральность)
        try:
            if self.graph.number_of_nodes() > 1:
                centrality = nx.degree_centrality(self.graph)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                analysis['top_central_nodes'] = [
                    {'node': n, 'centrality': c} for n, c in top_nodes
                ]
        except:
            pass
        
        return analysis
    
    async def _add_relation(self, rel: Dict, source_chunk_id: str):
        """Добавление отношения в граф"""
        source = rel.get('source', '')
        target = rel.get('target', '')
        rel_type = rel.get('type', 'related')
        
        if not source or not target:
            return
        
        # Создаем узлы для сущностей, если их нет
        for entity in [source, target]:
            if not self.graph.has_node(entity):
                self.graph.add_node(entity, type='entity')
                self.node_metadata[entity] = {
                    'created_at': datetime.now().isoformat(),
                    'type': 'entity'
                }
                self.stats['nodes_added'] += 1
        
        # Добавляем ребро
        self.graph.add_edge(source, target, relation=rel_type, weight=0.8, source_chunk=source_chunk_id)
        self.stats['edges_added'] += 1
        
        # Связываем чанк с сущностями
        self.graph.add_edge(source_chunk_id, source, relation='contains_entity')
        self.graph.add_edge(source_chunk_id, target, relation='contains_entity')
        self.stats['edges_added'] += 2
    
    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики графа"""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            **self.stats,
            'node_metadata_count': len(self.node_metadata),
            'graph_ready': True
        }
    
    async def save(self):
        """Сохранение графа в SQLite"""
        try:
            start_time = datetime.now()
            
            # Сериализуем граф
            graph_data = pickle.dumps(self.graph)
            metadata_data = pickle.dumps(self.node_metadata)
            
            if self.enable_compression:
                import zlib
                graph_data = zlib.compress(graph_data)
                metadata_data = zlib.compress(metadata_data)
            
            # Сохраняем в SQLite
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Создаем таблицу, если нет
            c.execute('''
                CREATE TABLE IF NOT EXISTS graph_store (
                    id INTEGER PRIMARY KEY,
                    graph BLOB,
                    metadata BLOB,
                    timestamp TEXT,
                    version TEXT
                )
            ''')
            
            c.execute('''
                INSERT INTO graph_store (graph, metadata, timestamp, version)
                VALUES (?, ?, ?, ?)
            ''', (
                graph_data,
                metadata_data,
                datetime.now().isoformat(),
                '1.0'
            ))
            
            conn.commit()
            conn.close()
            
            self.dirty = False
            self.stats['last_save_time'] = datetime.now().isoformat()
            self.stats['last_save_size'] = len(graph_data) + len(metadata_data)
            
            logger.debug(f"💾 Граф сохранен ({self.graph.number_of_nodes()} узлов) за "
                        f"{(datetime.now() - start_time).total_seconds():.2f} сек")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения графа: {e}")
            self.stats['errors'] += 1
    
    async def _load_from_db(self):
        """Загрузка графа из SQLite"""
        if not os.path.exists(self.db_path):
            logger.info("Файл БД графа не найден, создаем новый граф")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Проверяем наличие таблицы
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_store'")
            if not c.fetchone():
                conn.close()
                return
            
            # Берем последнюю запись
            c.execute('''
                SELECT graph, metadata FROM graph_store
                ORDER BY id DESC LIMIT 1
            ''')
            
            row = c.fetchone()
            conn.close()
            
            if row:
                graph_data, metadata_data = row
                
                if self.enable_compression:
                    import zlib
                    graph_data = zlib.decompress(graph_data)
                    metadata_data = zlib.decompress(metadata_data)
                
                self.graph = pickle.loads(graph_data)
                self.node_metadata = pickle.loads(metadata_data)
                
                logger.info(f"📦 Граф загружен: {self.graph.number_of_nodes()} узлов")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки графа: {e}")
            # Создаем новый граф
            self.graph = nx.Graph()
            self.node_metadata = {}
    
    async def _auto_save_loop(self):
        """Автоматическое сохранение графа"""
        while True:
            await asyncio.sleep(self.save_interval)
            if self.dirty:
                await self.save()
    
    async def get_weak_topics(self, min_weight: float = 0.7, limit: int = 5) -> List[str]:
        """
        Находит темы, у которых средний вес связей ниже порога.
        Для самообучения - выбирать слабо изученные темы.
        
        Args:
            min_weight: Минимальный средний вес (ниже этого - тема слабая)
            limit: Максимальное количество тем
        
        Returns:
            Список названий слабых тем
        """
        try:
            topic_weights = []
            
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get('type') == 'topic':
                    # Получаем название темы
                    topic_name = attrs.get('name', '')
                    if not topic_name and node.startswith('topic_'):
                        topic_name = node[6:]
                    
                    if not topic_name:
                        continue
                    
                    # Собираем веса всех связей этой темы
                    weights = []
                    for u, v, data in self.graph.edges(data=True):
                        if (u == node or v == node) and data.get('relation') == 'contains':
                            weights.append(data.get('weight', 1.0))
                    
                    if weights:
                        avg_weight = sum(weights) / len(weights)
                        if avg_weight < min_weight:
                            topic_weights.append((avg_weight, topic_name))
                            logger.debug(f"   Слабая тема: {topic_name}, средний вес={avg_weight:.2f}")
            
            # Сортируем по возрастанию веса (самые слабые первые)
            topic_weights.sort(key=lambda x: x[0])
            
            result = [name for weight, name in topic_weights[:limit]]
            logger.info(f"📊 Найдено слабых тем: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в get_weak_topics: {e}")
            return []


    async def close(self):
        """Завершение работы"""
        logger.info("🛑 Завершение работы NetworkXGraphService...")
        
        if self.save_task:
            self.save_task.cancel()
            try:
                await self.save_task
            except asyncio.CancelledError:
                pass
        
        # Сохраняем перед закрытием
        if self.dirty:
            await self.save()
        
        logger.info("✅ NetworkXGraphService завершил работу")
    
    async def health_check(self) -> Dict:
        """Проверка здоровья"""
        try:
            # Простая проверка - можем получить количество узлов
            nodes = self.graph.number_of_nodes()
            return {
                'healthy': True,
                'nodes': nodes,
                'edges': self.graph.number_of_edges(),
                'message': 'Graph service is operational',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_metrics(self) -> Dict:
        """Метрики для мониторинга"""
        stats = await self.get_stats()
        analysis = await self.analyze_structure()
        return {
            **stats,
            'density': analysis.get('density', 0),
            'connected_components': analysis.get('connected_components', 0),
            'avg_clustering': analysis.get('avg_clustering', 0),
            'node_type_distribution': dict(analysis.get('node_types', {}))
        }