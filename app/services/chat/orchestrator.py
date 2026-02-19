from typing import Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime

from app.core.logging import logger
from app.services.knowledge.base import KnowledgeBase
from app.services.chat.cache import QuestionCache
from app.services.chat.memory import MemoryManager  # ДОБАВЛЯЕМ
from app.services.enhancements.ab_testing import ABTestingService
from app.services.enhancements.rl_agent import RLAgent
from app.services.enhancements.followup import FollowupGenerator

@dataclass
class ProcessingContext:
    question: str
    session_id: str
    timestamp: datetime
    metadata: Dict[str, Any]

class ChatOrchestrator:
    """Продакшен-оркестратор чата с управлением памятью"""
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        memory_manager: Optional[MemoryManager] = None,  # ДОБАВЛЯЕМ параметр
        use_cache: bool = True,
        use_enhancements: bool = True
    ):
        self.knowledge_base = knowledge_base
        self.memory_manager = memory_manager
        self.use_cache = use_cache
        self.use_enhancements = use_enhancements
        
        # Инициализация компонентов
        if use_cache:
            self.cache = QuestionCache()
        
        if use_enhancements:
            self.ab_testing = ABTestingService()
            self.rl_agent = RLAgent()
            self.followup_generator = FollowupGenerator()
    
    async def process(
        self,
        question: str,
        session_id: str,
        use_knowledge: bool = True
    ) -> Dict[str, Any]:
        """Основной метод обработки вопроса с сохранением контекста"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing question: '{question}' for session {session_id}")
            
            # Сохраняем вопрос пользователя в историю сессии
            if self.memory_manager:
                await self.memory_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=question
                )
            
            # Получаем контекст диалога
            context_text = ""
            if self.memory_manager:
                context_text = await self.memory_manager.get_conversation_context(session_id)
            
            # Формируем контекстный запрос
            enriched_question = self._enrich_with_context(question, context_text)
            
            # ДЕБАГ: логируем обогащенный вопрос
            if enriched_question != question:
                logger.debug(f"Enriched question with context: {enriched_question}")
            
            # Проверка кэша
            if self.use_cache:
                cached = self.cache.get(enriched_question)
                if cached:
                    logger.debug(f"Cache hit for question: {question[:50]}")
                    cached["from_cache"] = True
                    
                    # Сохраняем ответ в историю
                    if self.memory_manager and "answer" in cached:
                        await self.memory_manager.add_message(
                            session_id=session_id,
                            role="assistant",
                            content=cached["answer"]
                        )
                    
                    return cached
            
            # Остальная логика обработки...
            context = ProcessingContext(
                question=question,
                session_id=session_id,
                timestamp=datetime.now(),
                metadata={
                    "has_memory_manager": self.memory_manager is not None,
                    "context_length": len(context_text),
                    "enriched_question": enriched_question != question  # Флаг, что вопрос был обогащен
                }
            )
            
            # Анализ вопроса
            question_analysis = await self._analyze_question(question)
            context.metadata["analysis"] = question_analysis
            
            # Поиск в базе знаний - ИСПОЛЬЗУЕМ ОБОГАЩЕННЫЙ ВОПРОС!
            if use_knowledge:
                try:
                    # Используем enriched_question для поиска, чтобы учитывать контекст
                    knowledge_results = await self.knowledge_base.search(
                        query=enriched_question,  # <-- ЗДЕСЬ ИЗМЕНЕНИЕ!
                        top_k=5
                    )
                    logger.info(f"Found {len(knowledge_results)} knowledge results")
                    context.metadata["knowledge_results"] = len(knowledge_results)
                    
                    if knowledge_results:
                        logger.debug(f"Top result: {knowledge_results[0].content[:100]}...")
                        logger.debug(f"Top result score: {knowledge_results[0].score}")
                except Exception as e:
                    logger.error(f"Error searching knowledge base: {e}")
                    knowledge_results = []
            else:
                knowledge_results = []
            
            # RL агент
            confidence = 0.8
            if self.use_enhancements:
                confidence = await self.rl_agent.adjust_confidence(
                    question_analysis=question_analysis,
                    knowledge_results=knowledge_results
                )
            
            # Генерация ответа
            answer = await self._generate_answer(
                question=question,
                knowledge_results=knowledge_results,
                confidence=confidence
            )
            
            # A/B тестирование
            if self.use_enhancements:
                template = await self.ab_testing.get_template(
                    question_type=question_analysis.get("type", "general")
                )
                answer = template.format(answer=answer)
            
            # Follow-up вопросы
            followup_suggestions = []
            if self.use_enhancements:
                followup_suggestions = await self.followup_generator.generate(
                    question=question,
                    answer=answer,
                    context=context
                )
            
            # Сохраняем ответ в историю
            if self.memory_manager:
                await self.memory_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=answer,
                    metadata={"confidence": confidence}
                )
            
            # Формируем результат
            result = {
                "answer": answer,
                "confidence": confidence,
                "sources": [r.id for r in knowledge_results],
                "metadata": context.metadata,
                "followup_suggestions": followup_suggestions,
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "from_cache": False
            }
            
            # Сохранение в кэш
            if self.use_cache:
                self.cache.set(enriched_question, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}", exc_info=True)
            return {
                "answer": "Произошла ошибка при обработке вашего вопроса.",
                "confidence": 0.0,
                "sources": [],
                "metadata": {"error": str(e)},
                "followup_suggestions": [],
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "from_cache": False
            }
    
    def _enrich_with_context(self, question: str, context: str) -> str:
        """Обогащение вопроса контекстом диалога"""
        if not context or len(context.strip()) < 20:
            return question
        
        # Ограничиваем длину контекста (макс 2000 символов)
        context = context[-2000:] if len(context) > 2000 else context
        
        # Для семантического поиска контекст может улучшить результаты
        return f"Предыдущий диалог: {context}\n\nТекущий вопрос: {question}"
    
    async def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Анализ вопроса"""
        return {
            "type": "factual" if "?" in question else "conversational",
            "length": len(question),
            "language": "ru" if any(cyr in question for cyr in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя") else "en",
            "requires_knowledge": len(question.split()) > 3,
            "has_context": "..." in question  # Простая эвристика
        }
    
    async def _generate_answer(
        self,
        question: str,
        knowledge_results: list,
        confidence: float
    ) -> str:
        """Генерация ответа на основе знаний"""
        # Проверяем, есть ли результаты
        if not knowledge_results or len(knowledge_results) == 0:
            # Если результатов нет, возвращаем общий ответ
            general_responses = [
                "Привет! Я AI ассистент. Чем могу помочь?",
                "Здравствуйте! Задайте мне вопрос, и я постараюсь помочь.",
                "Приветствую! Я здесь, чтобы помочь вам с вопросами.",
                "Здравствуйте! Как я могу вам помочь сегодня?"
            ]
            import random
            return random.choice(general_responses)
        
        # Проверяем, есть ли content у первого результата
        best_result = knowledge_results[0]
        if not hasattr(best_result, 'content') or not best_result.content:
            return "Информация по вашему вопросу есть, но я не могу её обработать."
        
        # Форматируем ответ
        content = best_result.content
        
        if confidence > 0.7:
            return f"На основе имеющейся информации: {content}"
        elif confidence > 0.4:
            return f"Возможно, это может помочь: {content}"
        else:
            return f"Не уверен, но возможно: {content}"
    
    async def get_session_history(self, session_id: str, limit: int = 20):
        """Получение истории сессии"""
        if self.memory_manager:
            return await self.memory_manager.get_history(session_id, limit)
        return []
    
    async def clear_session(self, session_id: str) -> bool:
        """Очистка сессии"""
        if self.memory_manager:
            return await self.memory_manager.clear_session(session_id)
        return False