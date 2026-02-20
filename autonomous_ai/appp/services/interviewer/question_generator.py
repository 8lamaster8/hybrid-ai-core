"""
🎤 ИНТЕРВЬЮЕР - Генерация исследовательских вопросов
Создаёт вопросы для углубления знаний, циклов координат, самообучения
"""

import random
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from appp.core.logging import logger


class QuestionGenerator:
    """
    Интервьюер: генерирует вопросы для исследования тем.
    Поддерживает разные типы вопросов и уровни глубины.
    """
    
    def __init__(self, config: Dict, graph_db: Optional[Any] = None):
        self.config = config

        self.graph_db = graph_db
        
        self.max_questions_per_topic = config.get('max_questions_per_topic', 15)
        self.question_depth_levels = config.get('question_depth_levels', 3)
        self.enable_followup_questions = config.get('enable_followup_questions', True)
        self.question_types = config.get('question_types', 
                                         ['factual', 'comparative', 'causal', 'procedural'])
        self.min_question_quality = config.get('min_question_quality', 0.6)
        self.language = config.get('language', 'ru')
        
        # Шаблоны вопросов для русского языка
        self.templates_ru = {
            'factual': [
                "Что такое {topic}?",
                "Кто создал {topic}?",
                "Когда появился {topic}?",
                "Где используется {topic}?",
                "Каковы основные характеристики {topic}?",
                "Из каких компонентов состоит {topic}?",
                "Какие существуют виды {topic}?",
                "Как определяется {topic}?",
                "Какие примеры {topic} вы знаете?",
                "В чем суть {topic}?"
            ],
            'comparative': [
                "Чем {topic} отличается от {related}?",
                "Какие преимущества у {topic} перед {related}?",
                "Что общего у {topic} и {related}?",
                "Что лучше: {topic} или {related}?",
                "Сравните {topic} и {related}",
                "В каких случаях предпочтительнее использовать {topic} вместо {related}?"
            ],
            'causal': [
                "Почему {topic} важен?",
                "Какие факторы влияют на {topic}?",
                "Каковы причины возникновения {topic}?",
                "К каким последствиям приводит {topic}?",
                "Зачем нужно изучать {topic}?",
                "Как {topic} влияет на {related}?"
            ],
            'procedural': [
                "Как работает {topic}?",
                "Каким образом реализовать {topic}?",
                "Как использовать {topic} на практике?",
                "Какие шаги необходимы для {topic}?",
                "Как научиться {topic}?",
                "Какие инструменты нужны для работы с {topic}?"
            ],
            'historical': [
                "Какова история развития {topic}?",
                "Кто внес наибольший вклад в развитие {topic}?",
                "Как эволюционировал {topic} с течением времени?",
                "Какие этапы можно выделить в развитии {topic}?"
            ],
            'future': [
                "Какое будущее у {topic}?",
                "Какие тенденции развития {topic}?",
                "Что изменится в {topic} через 5 лет?",
                "Какие инновации ожидают {topic}?"
            ],
            'problem': [
                "С какими проблемами сталкивается {topic}?",
                "Каковы ограничения {topic}?",
                "Какие недостатки у {topic}?",
                "Какие сложности возникают при работе с {topic}?"
            ]
        }
        
        # Для английского языка (полные рабочие шаблоны)
        self.templates_en = {
            'factual': [
                "What is {topic}?",
                "Who created {topic}?",
                "When did {topic} appear?",
                "Where is {topic} used?",
                "What are the main characteristics of {topic}?",
                "What components does {topic} consist of?",
                "What types of {topic} exist?",
                "How is {topic} defined?",
                "What examples of {topic} do you know?",
                "What is the essence of {topic}?"
            ],
            'comparative': [
                "How is {topic} different from {related}?",
                "What are the advantages of {topic} over {related}?",
                "What do {topic} and {related} have in common?",
                "Which is better: {topic} or {related}?",
                "Compare {topic} and {related}",
                "In what cases is it preferable to use {topic} instead of {related}?"
            ],
            'causal': [
                "Why is {topic} important?",
                "What factors influence {topic}?",
                "What are the causes of {topic}?",
                "What are the consequences of {topic}?",
                "Why should we study {topic}?",
                "How does {topic} affect {related}?"
            ],
            'procedural': [
                "How does {topic} work?",
                "How to implement {topic}?",
                "How to use {topic} in practice?",
                "What steps are needed for {topic}?",
                "How to learn {topic}?",
                "What tools are needed to work with {topic}?"
            ],
            'historical': [
                "What is the history of {topic}?",
                "Who contributed most to the development of {topic}?",
                "How has {topic} evolved over time?",
                "What stages can be identified in the development of {topic}?"
            ],
            'future': [
                "What is the future of {topic}?",
                "What are the development trends of {topic}?",
                "What will change in {topic} in 5 years?",
                "What innovations are expected in {topic}?"
            ],
            'problem': [
                "What problems does {topic} face?",
                "What are the limitations of {topic}?",
                "What are the disadvantages of {topic}?",
                "What difficulties arise when working with {topic}?"
            ]
        }
        
        # Выбираем язык
        if self.language == 'ru':
            self.templates = self.templates_ru
        else:
            self.templates = self.templates_en
        
        # Стоп-слова для фильтрации вопросов
        self.stop_phrases = [
            'не найдено', 'неизвестно', 'нет информации',
            'не удалось', 'ошибка', 'информация отсутствует'
        ]
        
        self.stats = {
            'questions_generated': 0,
            'research_cycles_supported': 0,
            'avg_questions_per_topic': 0,
            'errors': 0
        }
        
        logger.info("🎤 QuestionGenerator создан")
    
    async def initialize(self):
        """Инициализация"""
        logger.info("🔄 Инициализация QuestionGenerator...")
        return True
    
    async def generate_research_questions(
        self,
        topic: str,
        depth: int = 2,
        num_questions: int = 10,
        question_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Генерация вопросов для исследования темы.
        """
        logger.info(f"🎯 Генерация вопросов для темы: '{topic}' (глубина {depth})")
        
        # Очищаем тему
        topic_clean = self._clean_topic(topic)
        
        # Получаем связанные темы для сравнительных вопросов
        related_topics = await self._get_related_topics(topic_clean)
        logger.info(f"   🔗 Связанные темы: {related_topics}")
        
        # Выбираем типы вопросов
        if question_types is None:
            if depth == 1:
                types = ['factual']
            elif depth == 2:
                types = ['factual', 'procedural', 'comparative']
            else:
                types = ['factual', 'comparative', 'causal', 'historical', 'future', 'problem']
        else:
            types = question_types
        
        all_questions = []
        
        for q_type in types:
            if q_type not in self.templates:
                continue
                
            templates = self.templates[q_type]
            num_from_type = max(1, num_questions // len(types))
            selected = random.sample(templates, min(num_from_type, len(templates)))
            
            for template in selected:
                # Проверяем, есть ли в шаблоне {related}
                if '{related}' in template:
                    # Если есть related-темы, создаём по вопросу для каждой
                    if related_topics:
                        for related in related_topics:
                            try:
                                question = template.format(topic=topic_clean, related=related)
                                all_questions.append(question)
                            except KeyError:
                                # Если что-то пошло не так, используем упрощённый вариант
                                simple = template.replace('{related}', 'другими подходами')
                                question = simple.format(topic=topic_clean)
                                all_questions.append(question)
                    else:
                        # Если related-тем нет, заменяем {related} на что-то общее
                        simple_template = template.replace('от {related}', 'от аналогов')
                        simple_template = simple_template.replace('с {related}', 'с альтернативами')
                        simple_template = simple_template.replace('{related}', 'другими подходами')
                        question = simple_template.format(topic=topic_clean)
                        all_questions.append(question)
                else:
                    # Обычный шаблон без {related}
                    question = template.format(topic=topic_clean)
                    all_questions.append(question)
        
        # Добавляем вариации
        all_questions = self._add_variations(all_questions, topic_clean)
        
        # Удаляем дубликаты
        unique_questions = list(dict.fromkeys(all_questions))
        
        # Фильтруем по качеству
        filtered_questions = self._filter_questions(unique_questions)
        
        # Ограничиваем количество
        result = filtered_questions[:num_questions]
        
        logger.info(f"✅ Сгенерировано {len(result)} вопросов по теме '{topic}'")
        
        return result
    
    async def _get_related_topics(self, topic: str, max_topics: int = 3) -> List[str]:
        try:
            graph = self.graph_db
            if graph is None:
                return []
            
            # Получаем связанные темы с весами
            related = []
            for node, attrs in graph.graph.nodes(data=True):
                if attrs.get('type') == 'topic' and node != f"topic_{topic}":
                    # Ищем путь до темы через общие чанки
                    for edge_u, edge_v, edge_data in graph.graph.edges(data=True):
                        if edge_data.get('relation') == 'contains':
                            chunk = edge_v if edge_u.startswith('topic') else edge_u
                            if chunk.startswith('chunk'):
                                # Проверяем, связан ли чанк с нашей темой
                                if graph.graph.has_edge(f"topic_{topic}", chunk):
                                    weight = edge_data.get('weight', 1.0)
                                    related.append((node, weight))
            
            # Сортируем по весу и берём лучшие
            related.sort(key=lambda x: x[1], reverse=True)
            return [r[0].replace('topic_', '') for r in related[:max_topics]]
            
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            return []


    async def generate_deepening_questions(
        self,
        knowledge_chunks: List[Dict],
        current_depth: int,
        max_questions: int = 5
    ) -> List[str]:
        """
        Генерация углубляющих вопросов на основе уже полученных знаний.
        Используется в циклах координат.
        
        Args:
            knowledge_chunks: Результаты предыдущего цикла (ответы на вопросы)
            current_depth: Текущая глубина
            max_questions: Максимальное количество вопросов
            
        Returns:
            Список новых вопросов для углубления
        """
        if not knowledge_chunks:
            return []
        
        # Анализируем ответы, ищем ключевые термины
        all_text = ' '.join([
            chunk.get('answer', '') or chunk.get('text', '') 
            for chunk in knowledge_chunks
        ])
        
        # Извлекаем потенциальные темы для углубления
        # (простейшая эвристика: ищем слова с большой буквы, длинные термины)
        words = all_text.split()
        candidates = []
        
        for word in words:
            # Существительные с большой буквы (кроме начала предложения)
            if word[0].isupper() and len(word) > 3:
                if word not in candidates and not self._is_stop_word(word):
                    candidates.append(word)
        
        # Если кандидатов мало, берем любые длинные слова
        if len(candidates) < 3:
            long_words = [w for w in words if len(w) > 6 and w.isalpha()]
            candidates.extend(long_words)
        
        # Удаляем дубликаты и ограничиваем
        candidates = list(dict.fromkeys(candidates))[:5]
        
        # Генерируем вопросы для каждого кандидата
        deepening_questions = []
        
        for candidate in candidates:
            # В зависимости от глубины, задаем разные типы вопросов
            if current_depth == 0:
                q = f"Что такое {candidate}?"
            elif current_depth == 1:
                q = f"Как работает {candidate}?"
            elif current_depth == 2:
                q = f"Какие существуют разновидности {candidate}?"
            else:
                q = f"Каковы перспективы развития {candidate}?"
            
            deepening_questions.append(q)
            
            if len(deepening_questions) >= max_questions:
                break
        
        return deepening_questions[:max_questions]
    
    async def generate_followup_questions(
        self,
        question: str,
        answer: str,
        max_questions: int = 3
    ) -> List[str]:
        """
        Генерация уточняющих вопросов на основе ответа.
        """
        if not self.enable_followup_questions:
            return []
        
        followups = []
        
        # Простые эвристики
        if 'потому что' in answer or 'так как' in answer:
            followups.append(f"Почему это важно для {self._extract_topic(question)}?")
        
        if 'например' in answer or 'к примеру' in answer:
            followups.append(f"Какие еще примеры {self._extract_topic(question)} существуют?")
        
        if 'является' in answer and 'это' in answer:
            followups.append(f"Каковы основные характеристики {self._extract_topic(question)}?")
        
        if 'используется' in answer:
            followups.append(f"Где еще можно применить {self._extract_topic(question)}?")
        
        # Ограничиваем количество
        return followups[:max_questions]
    
    def _clean_topic(self, topic: str) -> str:
        """Очистка темы от лишнего"""
        topic = topic.strip().rstrip('?')
        
        # Убираем вопросительные слова в начале
        prefixes = ['что такое ', 'кто такой ', 'кто такая ', 'как ', 'почему ', 'зачем ',
                   'what is ', 'who is ', 'how to ', 'why ']
        
        topic_lower = topic.lower()
        for prefix in prefixes:
            if topic_lower.startswith(prefix):
                topic = topic[len(prefix):].strip()
                break
        
        return topic
    
    def _extract_topic(self, question: str) -> str:
        """Извлечение темы из вопроса"""
        return self._clean_topic(question)
    
    def _add_variations(self, questions: List[str], topic: str) -> List[str]:
        """Добавление вариаций формулировок"""
        variations = []
        
        # Для некоторых вопросов меняем порядок слов
        for q in questions:
            variations.append(q)
            
            # Добавляем вопрос "Расскажи о ..."
            if 'Что такое' in q or 'Кто такой' in q:
                variations.append(f"Расскажи о {topic}")
                variations.append(f"Информация о {topic}")
                variations.append(f"Подробнее о {topic}")
        
        return variations
    
    def _filter_questions(self, questions: List[str]) -> List[str]:
        """Фильтрация вопросов по качеству"""
        filtered = []
        
        for q in questions:
            # Длина вопроса
            if len(q) < 10 or len(q) > 200:
                continue
            
            # Проверка на стоп-фразы
            q_lower = q.lower()
            if any(stop in q_lower for stop in self.stop_phrases):
                continue
            
            # Убираем вопросы без темы
            topic_placeholder = '{topic}'
            if topic_placeholder in q:
                continue  # шаблон не заполнен
            
            filtered.append(q)
        
        return filtered
    
    def _is_stop_word(self, word: str) -> bool:
        """Проверка на стоп-слова"""
        stop_words = {'Это', 'Что', 'Как', 'Где', 'Когда', 'Почему', 'Зачем',
                     'Кто', 'Какой', 'Какая', 'Какое', 'Какие', 'Чей',
                     'The', 'A', 'An', 'What', 'Who', 'Where', 'When', 'Why'}
        return word in stop_words
    
    async def get_stats(self) -> Dict:
        """Статистика"""
        return {
            'questions_generated': self.stats['questions_generated'],
            'research_cycles_supported': self.stats['research_cycles_supported'],
            'avg_questions_per_topic': self.stats['avg_questions_per_topic'],
            'errors': self.stats['errors']
        }
    
    async def health_check(self) -> Dict:
        """Проверка здоровья"""
        return {
            'healthy': True,
            'message': 'QuestionGenerator is operational',
            'timestamp': datetime.now().isoformat()
        }