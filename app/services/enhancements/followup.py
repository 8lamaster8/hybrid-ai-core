"""
Генератор follow-up вопросов на основе контекста
"""
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import random

from app.core.logging import logger
from app.services.knowledge.vector_store import ChromaVectorStore  # ИЗМЕНЕНО: KnowledgeVectorStore → ChromaVectorStore


class FollowupGenerator:
    """Генератор уточняющих и связанных вопросов"""
    
    def __init__(self):
        self.vector_store = None
        self.patterns = {
            "factual": [
                "Что еще вы хотели бы узнать о {topic}?",
                "Интересно ли вам узнать больше о {aspect}?",
                "Хотите углубиться в тему {topic}?"
            ],
            "how_to": [
                "Нужна ли дополнительная помощь по {topic}?",
                "Хотите пошаговую инструкцию по {task}?",
                "Интересны ли другие способы решения {problem}?"
            ],
            "comparison": [
                "Сравнить {topic} с чем-то еще?",
                "Интересны ли аналоги {topic}?",
                "Хотите узнать преимущества и недостатки?"
            ],
            "general": [
                "Что еще вас интересует?",
                "Есть ли еще вопросы?",
                "Могу ли я помочь с чем-то еще?"
            ]
        }
        
        # Ключевые слова для извлечения тем
        self.topic_keywords = ["о", "про", "как", "что", "где", "когда", "почему", "зачем"]
        
    async def initialize(self):
        """Инициализация генератора"""
        self.vector_store = ChromaVectorStore()  # ИЗМЕНЕНО
        await self.vector_store.initialize()
        logger.info("Followup Generator initialized")
    
    def _extract_topic(self, question: str) -> Optional[str]:
        """
        Извлечение основной темы из вопроса
        
        Args:
            question: Вопрос пользователя
        
        Returns:
            Извлеченная тема или None
        """
        # Убираем вопросительные слова
        words = question.lower().split()
        
        # Ищем существительные после вопросительных слов
        topic_words = []
        for i, word in enumerate(words):
            if word in self.topic_keywords and i + 1 < len(words):
                # Берем следующее слово как потенциальную тему
                next_word = words[i + 1]
                if len(next_word) > 2:  # Игнорируем короткие слова
                    topic_words.append(next_word)
        
        if topic_words:
            return " ".join(topic_words[-2:])  # Последние 1-2 слова как тему
        
        # Альтернатива: первые существительные
        for word in words:
            if len(word) > 3 and not word.endswith(('?', '!', '.')):
                return word
        
        return None
    
    def _detect_question_type(self, question: str) -> str:
        """
        Определение типа вопроса
        
        Args:
            question: Вопрос пользователя
        
        Returns:
            Тип вопроса
        """
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["как", "способ", "метод", "инструкция"]):
            return "how_to"
        elif any(word in question_lower for word in ["сравнить", "лучше", "хуже", "чем", "против"]):
            return "comparison"
        elif "?" in question:
            return "factual"
        else:
            return "general"
    
    async def _get_related_topics(self, topic: str, limit: int = 3) -> List[str]:
        """
        Получение связанных тем из базы знаний
        
        Args:
            topic: Основная тема
            limit: Количество связанных тем
        
        Returns:
            Список связанных тем
        """
        try:
            if not self.vector_store:
                return []
            
            # Ищем похожие документы
            results = await self.vector_store.search(
                query=topic,
                top_k=limit * 2
            )
            
            # Извлекаем ключевые слова из результатов
            related = set()
            for result in results:
                content = result.content.lower()
                
                # Простая эвристика для извлечения тем
                words = re.findall(r'\b[a-zа-я]{4,}\b', content)
                
                # Ищем слова, связанные с темой
                for word in words[:10]:  # Берем первые слова
                    if (len(word) > 3 and 
                        word not in topic and 
                        not any(stop in word for stop in ["этот", "такой", "очень"])):
                        related.add(word)
                        
                        if len(related) >= limit:
                            break
                
                if len(related) >= limit:
                    break
            
            return list(related)[:limit]
            
        except Exception as e:
            logger.error(f"Ошибка получения связанных тем: {e}")
            return []
    
    def _generate_from_pattern(
        self, 
        pattern_type: str, 
        topic: Optional[str] = None,
        aspect: Optional[str] = None
    ) -> str:
        """
        Генерация вопроса из шаблона
        
        Args:
            pattern_type: Тип шаблона
            topic: Тема
            aspect: Аспект
        
        Returns:
            Сгенерированный вопрос
        """
        patterns = self.patterns.get(pattern_type, self.patterns["general"])
        pattern = random.choice(patterns)
        
        # Заполняем плейсхолдеры
        if topic and "{topic}" in pattern:
            pattern = pattern.replace("{topic}", topic)
        
        if aspect and "{aspect}" in pattern:
            pattern = pattern.replace("{aspect}", aspect)
        elif "{aspect}" in pattern and topic:
            # Если аспект не указан, используем тему
            pattern = pattern.replace("{aspect}", topic)
        
        # Заменяем оставшиеся плейсхолдеры
        if "{task}" in pattern and topic:
            pattern = pattern.replace("{task}", topic)
        
        if "{problem}" in pattern and topic:
            pattern = pattern.replace("{problem}", topic)
        
        return pattern.capitalize()
    
    async def generate(
        self,
        question: str,
        answer: str,
        context: Any,
        count: int = 3
    ) -> List[str]:
        """
        Генерация follow-up вопросов
        
        Args:
            question: Исходный вопрос
            answer: Ответ системы
            context: Контекст обработки
            count: Количество вопросов
        
        Returns:
            Список follow-up вопросов
        """
        try:
            followups = []
            
            # Извлекаем тему
            topic = self._extract_topic(question)
            
            # Определяем тип вопроса
            question_type = self._detect_question_type(question)
            
            # 1. Основной follow-up на основе темы
            if topic:
                main_followup = self._generate_from_pattern(question_type, topic)
                if main_followup not in followups:
                    followups.append(main_followup)
            
            # 2. Получаем связанные темы
            if topic and len(followups) < count:
                related_topics = await self._get_related_topics(topic, limit=2)
                
                for related in related_topics:
                    if len(followups) >= count:
                        break
                    
                    followup = self._generate_from_pattern("factual", topic=related)
                    if followup not in followups:
                        followups.append(followup)
            
            # 3. Дополняем общими вопросами если нужно
            while len(followups) < count:
                general = self._generate_from_pattern("general")
                if general not in followups:
                    followups.append(general)
            
            # Лимитируем количество
            followups = followups[:count]
            
            # Логирование
            logger.debug(
                f"Сгенерировано {len(followups)} follow-up вопросов",
                extra={
                    "context": {
                        "original_question": question[:50],
                        "topic": topic,
                        "question_type": question_type,
                        "followups": followups
                    }
                }
            )
            
            return followups
            
        except Exception as e:
            logger.error(f"Ошибка генерации follow-up вопросов: {e}")
            return ["Что еще вас интересует?", "Есть ли еще вопросы?"]
    
    async def generate_from_answer(
        self,
        answer: str,
        count: int = 2
    ) -> List[str]:
        """
        Генерация вопросов на основе ответа
        
        Args:
            answer: Ответ системы
            count: Количество вопросов
        
        Returns:
            Список вопросов
        """
        try:
            # Простая эвристика: ищем утверждения в ответе
            sentences = re.split(r'[.!?]', answer)
            questions = []
            
            for sentence in sentences:
                if len(sentence.strip()) > 20 and len(questions) < count:
                    # Преобразуем утверждение в вопрос
                    words = sentence.strip().split()
                    if len(words) > 3:
                        # Просто добавляем "Правда ли, что ...?"
                        question = f"Правда ли, что {sentence.strip().lower()}?"
                        questions.append(question.capitalize())
            
            return questions[:count]
            
        except Exception as e:
            logger.error(f"Ошибка генерации вопросов из ответа: {e}")
            return []
    
    async def get_suggestions_for_topic(
        self,
        topic: str,
        limit: int = 5
    ) -> List[str]:
        """
        Получение популярных вопросов по теме
        
        Args:
            topic: Тема
            limit: Максимальное количество
        
        Returns:
            Список популярных вопросов
        """
        try:
            # В реальной системе здесь запрос к аналитике
            # Пока заглушка с предопределенными вопросами
            
            topic_questions = {
                "python": [
                    "Как установить Python?",
                    "Какие фреймворки есть для веб-разработки?",
                    "Чем отличается Python 2 от Python 3?",
                    "Как работать с виртуальными окружениями?"
                ],
                "машинное обучение": [
                    "С чего начать изучение машинного обучения?",
                    "Какие библиотеки используются для ML?",
                    "В чем разница между supervised и unsupervised learning?",
                    "Как подготовить данные для обучения модели?"
                ],
                "базы данных": [
                    "Как выбрать базу данных для проекта?",
                    "В чем разница между SQL и NoSQL?",
                    "Как оптимизировать запросы к базе данных?",
                    "Что такое индексы и зачем они нужны?"
                ]
            }
            
            # Ищем совпадения по теме
            for key, questions in topic_questions.items():
                if key in topic.lower():
                    return questions[:limit]
            
            # Возвращаем общие вопросы если тема не найдена
            return [
                f"Что такое {topic}?",
                f"Для чего используется {topic}?",
                f"Какие есть альтернативы {topic}?"
            ][:limit]
            
        except Exception as e:
            logger.error(f"Ошибка получения вопросов по теме: {e}")
            return []
    
    async def close(self):
        """Корректное завершение работы"""
        pass


# Глобальный экземпляр
followup_generator = FollowupGenerator()