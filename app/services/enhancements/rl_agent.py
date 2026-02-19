"""
Исправленный RL Agent
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import asyncio

from app.core.config import settings
from app.core.logging import logger
from app.infrastructure.database import db_manager, Feedback
from app.infrastructure.cache import cached


class RLAgent:
    """
    RL агент для адаптивной настройки уверенности и стратегий ответов
    на основе обратной связи пользователей
    """
    
    def __init__(self):
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        
        # Состояния: тип вопроса + наличие знаний
        self.states = [
            "factual_with_knowledge",
            "factual_no_knowledge", 
            "conversational_with_knowledge",
            "conversational_no_knowledge",
            "ambiguous"
        ]
        
        # Действия: уровень уверенности
        self.actions = ["high", "medium", "low", "cautious"]
        
        # Q-таблица
        self.q_table = self._initialize_q_table()
        
        # Статистика
        self.stats = {
            "updates": 0,
            "rewards_received": 0,
            "explorations": 0,
            "exploitations": 0
        }
    
    async def initialize(self):
        """Инициализация агента"""
        # Загрузка Q-таблицы из кэша или БД
        await self._load_q_table()
        logger.info("RL Agent initialized")
    
    def _initialize_q_table(self) -> Dict[str, Dict[str, float]]:
        """Инициализация Q-таблицы"""
        q_table = {}
        for state in self.states:
            q_table[state] = {}
            for action in self.actions:
                # Начальные значения
                if action == "medium":
                    q_table[state][action] = 0.5  # Нейтральное значение
                else:
                    q_table[state][action] = 0.3
        return q_table
    
    async def _load_q_table(self):
        """Загрузка Q-таблицы из кэша"""
        try:
            # В реальной системе здесь загрузка из БД
            # Пока используем инициализированную таблицу
            pass
        except Exception as e:
            logger.warning(f"Не удалось загрузить Q-таблицу: {e}")
    
    async def _save_q_table(self):
        """Сохранение Q-таблицы в кэш"""
        try:
            # В реальной системе здесь сохранение в БД
            pass
        except Exception as e:
            logger.error(f"Не удалось сохранить Q-таблицу: {e}")
    
    def _get_state(
        self,
        question_analysis: Dict[str, Any],
        has_knowledge: bool
    ) -> str:
        """
        Определение состояния на основе анализа вопроса
        
        Args:
            question_analysis: Результат анализа вопроса
            has_knowledge: Есть ли релевантные знания
        
        Returns:
            Состояние для RL
        """
        question_type = question_analysis.get("type", "general")
        
        if question_type == "factual":
            return f"factual_{'with' if has_knowledge else 'no'}_knowledge"
        elif question_type == "conversational":
            return f"conversational_{'with' if has_knowledge else 'no'}_knowledge"
        else:
            return "ambiguous"
    
    async def choose_action(
        self,
        question_analysis: Dict[str, Any],
        has_knowledge: bool,
        force_exploit: bool = False
    ) -> Tuple[str, bool]:
        """
        Выбор действия (уровня уверенности) на основе политики ε-жадности
        
        Args:
            question_analysis: Анализ вопроса
            has_knowledge: Наличие знаний
            force_exploit: Принудительная эксплуатация (без исследования)
        
        Returns:
            (действие, было_ли_исследование)
        """
        state = self._get_state(question_analysis, has_knowledge)
        
        # ε-жадная стратегия
        if not force_exploit and np.random.random() < self.exploration_rate:
            # Исследование: случайное действие
            action = np.random.choice(self.actions)
            self.stats["explorations"] += 1
            return action, True
        else:
            # Эксплуатация: лучшее действие из Q-таблицы
            actions = self.q_table[state]
            best_action = max(actions, key=actions.get)
            self.stats["exploitations"] += 1
            return best_action, False
    
    def _action_to_confidence(self, action: str) -> float:
        """Преобразование действия в числовое значение уверенности"""
        confidence_map = {
            "high": 0.9,
            "medium": 0.7,
            "low": 0.4,
            "cautious": 0.2
        }
        return confidence_map.get(action, 0.5)
    
    async def adjust_confidence(
        self,
        question_analysis: Dict[str, Any],
        knowledge_results: List[Any]
    ) -> float:
        """
        Корректировка уверенности на основе RL
        
        Args:
            question_analysis: Анализ вопроса
            knowledge_results: Результаты поиска знаний
        
        Returns:
            Скорректированная уверенность
        """
        has_knowledge = len(knowledge_results) > 0
        
        # Выбираем действие
        action, was_exploration = await self.choose_action(
            question_analysis=question_analysis,
            has_knowledge=has_knowledge
        )
        
        # Преобразуем в уверенность
        base_confidence = self._action_to_confidence(action)
        
        # Корректировка на основе качества знаний
        if has_knowledge:
            # Учитываем score лучшего результата
            best_score = knowledge_results[0].score if hasattr(knowledge_results[0], 'score') else 0.5
            adjusted = base_confidence * (0.5 + 0.5 * best_score)
        else:
            adjusted = base_confidence * 0.5
        
        # Логирование
        logger.debug(
            f"RL confidence adjustment: {base_confidence:.2f} -> {adjusted:.2f}",
            extra={
                "context": {
                    "action": action,
                    "was_exploration": was_exploration,
                    "has_knowledge": has_knowledge,
                    "knowledge_count": len(knowledge_results)
                }
            }
        )
        
        return min(max(adjusted, 0.0), 1.0)  # Ограничение 0-1
    
    async def receive_feedback(
        self,
        conversation_id: int,  # ИСПРАВЛЕНО: conversation_id вместо session_id
        message_id: int,
        rating: int,
        helpful: Optional[bool] = None,
        meta_data: Optional[Dict[str, Any]] = None  # ИСПРАВЛЕНО: meta_data вместо metadata
    ):
        """
        Получение обратной связи и обновление Q-таблицы
        
        Args:
            conversation_id: ID диалога (из таблицы Conversation)
            message_id: ID сообщения
            rating: Рейтинг 1-5
            helpful: Полезность
            meta_data: Дополнительные метаданные
        """
        try:
            # Сохраняем обратную связь в БД
            async with db_manager.get_session() as session:
                feedback = Feedback(
                    conversation_id=conversation_id,  # ИСПРАВЛЕНО
                    message_id=message_id,
                    rating=rating,
                    helpful=helpful,
                    meta_data=meta_data or {}  # ИСПРАВЛЕНО
                )
                session.add(feedback)
                await session.commit()
            
            # Вычисляем награду
            reward = self._calculate_reward(rating, helpful)
            
            # Обновляем Q-таблицу (упрощенное обновление)
            await self._update_q_table(reward)
            
            self.stats["rewards_received"] += 1
            
            logger.info(
                f"Получена обратная связь: rating={rating}, reward={reward:.2f}",
                extra={"context": {"conversation_id": conversation_id, "message_id": message_id}}
            )
            
        except Exception as e:
            logger.error(f"Ошибка обработки обратной связи: {e}")
    
    def _calculate_reward(self, rating: int, helpful: Optional[bool]) -> float:
        """Вычисление награды на основе обратной связи"""
        if helpful is False:
            return -1.0
        elif helpful is True:
            return 1.0
        
        # На основе рейтинга
        rating_map = {
            1: -0.8,
            2: -0.4,
            3: 0.0,
            4: 0.6,
            5: 1.0
        }
        return rating_map.get(rating, 0.0)
    
    async def _update_q_table(self, reward: float):
        """Обновление Q-таблицы (упрощенное)"""
        # В реальной системе здесь full RL update с учетом состояния и действия
        self.stats["updates"] += 1
        
        # Периодическое сохранение
        if self.stats["updates"] % 100 == 0:
            await self._save_q_table()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики агента"""
        return {
            "q_table_size": len(self.q_table),
            "stats": self.stats.copy(),
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate
        }
    
    async def train_on_history(self, days: int = 30):
        """
        Обучение на исторических данных
        
        Args:
            days: Количество дней для обучения
        """
        try:
            logger.info(f"Начинаем обучение на исторических данных за {days} дней")
            
            # В реальной системе здесь обучение на обратной связи из БД
            # Пока заглушка
            await asyncio.sleep(1)
            
            logger.info("Обучение завершено")
            
        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
    
    async def close(self):
        """Корректное завершение работы"""
        await self._save_q_table()
        logger.info("RL Agent закрыт")


# Глобальный экземпляр
rl_agent = RLAgent()