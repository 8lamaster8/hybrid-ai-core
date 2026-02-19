"""
A/B Testing для экспериментов с шаблонами ответов
"""
import random
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import hashlib

from app.core.config import settings
from app.core.logging import logger
from app.infrastructure.cache import Cache


class ABTestingService:
    """Сервис A/B тестирования для экспериментов с ответами"""
    
    def __init__(self):
        self.cache = Cache()
        self.experiments = {
            "response_template": {
                "variants": [
                    {"id": "A", "template": "{answer}", "weight": 50},
                    {"id": "B", "template": "📚 {answer}", "weight": 25},
                    {"id": "C", "template": "🔍 Нашел информацию: {answer}", "weight": 15},
                    {"id": "D", "template": "💡 Вот что я узнал: {answer}", "weight": 10}
                ]
            },
            "confidence_display": {
                "variants": [
                    {"id": "A", "show": False, "weight": 60},
                    {"id": "B", "show": True, "template": "(Уверенность: {confidence:.0%})", "weight": 40}
                ]
            }
        }
    
    async def initialize(self):
        """Инициализация сервиса"""
        await self.cache.initialize()
        logger.info("A/B Testing Service initialized")
    
    def _get_user_variant(
        self,
        user_id: Optional[str],
        experiment_name: str,
        num_variants: int
    ) -> int:
        """
        Детерминированное распределение пользователей по вариантам
        
        Args:
            user_id: ID пользователя (если нет - случайный выбор)
            experiment_name: Название эксперимента
            num_variants: Количество вариантов
        
        Returns:
            Номер варианта (0-based)
        """
        if user_id:
            # Детерминированный выбор на основе user_id
            hash_str = f"{user_id}:{experiment_name}"
            hash_int = int(hashlib.md5(hash_str.encode()).hexdigest()[:8], 16)
            return hash_int % num_variants
        else:
            # Случайный выбор для анонимных пользователей
            return random.randint(0, num_variants - 1)
    
    async def get_template(
        self,
        question_type: str = "general",
        user_id: Optional[str] = None
    ) -> str:
        """
        Получение шаблона ответа на основе A/B теста
        
        Args:
            question_type: Тип вопроса
            user_id: ID пользователя
        
        Returns:
            Шаблон для форматирования ответа
        """
        experiment = self.experiments.get("response_template")
        if not experiment:
            return "{answer}"
        
        # Выбираем вариант на основе весов
        variants = experiment["variants"]
        total_weight = sum(v["weight"] for v in variants)
        
        if user_id:
            # Детерминированный выбор для залогиненного пользователя
            variant_idx = self._get_user_variant(user_id, "response_template", len(variants))
            variant = variants[variant_idx]
        else:
            # Взвешенный случайный выбор для анонимных
            rand = random.uniform(0, total_weight)
            cumulative = 0
            variant = variants[0]
            
            for v in variants:
                cumulative += v["weight"]
                if rand <= cumulative:
                    variant = v
                    break
        
        # Логируем выбор для аналитики
        await self._log_experiment_event(
            experiment_name="response_template",
            variant_id=variant["id"],
            user_id=user_id,
            metadata={"question_type": question_type}
        )
        
        return variant["template"]
    
    async def should_show_confidence(
        self,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Определение, показывать ли уверенность в ответе
        
        Returns:
            Словарь с настройками отображения уверенности
        """
        experiment = self.experiments.get("confidence_display")
        if not experiment:
            return {"show": False}
        
        variants = experiment["variants"]
        
        if user_id:
            variant_idx = self._get_user_variant(user_id, "confidence_display", len(variants))
            variant = variants[variant_idx]
        else:
            variant = random.choice(variants)
        
        await self._log_experiment_event(
            experiment_name="confidence_display",
            variant_id=variant["id"],
            user_id=user_id
        )
        
        return variant
    
    async def create_experiment(
        self,
        name: str,
        variants: List[Dict[str, Any]],
        description: str = ""
    ) -> bool:
        """
        Создание нового эксперимента
        
        Args:
            name: Название эксперимента
            variants: Список вариантов с весами
            description: Описание эксперимента
        
        Returns:
            Успешность создания
        """
        try:
            # Валидация вариантов
            total_weight = sum(v.get("weight", 0) for v in variants)
            if total_weight <= 0:
                raise ValueError("Сумма весов должна быть положительной")
            
            self.experiments[name] = {
                "variants": variants,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "total_weight": total_weight
            }
            
            logger.info(f"Создан эксперимент: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка создания эксперимента: {e}")
            return False
    
    async def get_experiment_results(
        self,
        experiment_name: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Получение результатов эксперимента
        
        Args:
            experiment_name: Название эксперимента
            days: За сколько дней
        
        Returns:
            Статистика по эксперименту
        """
        try:
            # В реальной системе здесь запрос к БД
            # Здесь упрощенная версия
            return {
                "experiment": experiment_name,
                "total_participants": random.randint(100, 1000),
                "variants": [
                    {"id": "A", "conversion_rate": 0.42, "participants": 450},
                    {"id": "B", "conversion_rate": 0.38, "participants": 550}
                ],
                "confidence_level": 0.95,
                "is_significant": True
            }
        except Exception as e:
            logger.error(f"Ошибка получения результатов: {e}")
            return {}
    
    async def _log_experiment_event(
        self,
        experiment_name: str,
        variant_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Логирование события эксперимента"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "experiment": experiment_name,
            "variant": variant_id,
            "user_id": user_id or "anonymous",
            "metadata": metadata or {}
        }
        
        # В реальной системе здесь отправка в аналитику
        logger.debug(f"Experiment event: {json.dumps(event, ensure_ascii=False)}")
    
    async def close(self):
        """Корректное завершение работы"""
        pass


# Глобальный экземпляр
ab_testing_service = ABTestingService()