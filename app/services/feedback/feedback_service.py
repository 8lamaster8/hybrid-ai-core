"""
Профессиональный сервис для работы с обратной связью
"""
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

from app.core.logging import logger
from app.infrastructure.database import db_manager, Feedback as FeedbackModel
from app.monitoring.metrics import metrics_collector


from app.core.logging import logger

# Инициализация метрик - безопасный импорт
try:
    from app.monitoring.metrics import metrics_collector
    METRICS_AVAILABLE = True
    logger.info("✅ Метрики доступны")
except ImportError as e:
    logger.warning(f"⚠️ Метрики недоступны: {e}. Используем заглушку.")
    METRICS_AVAILABLE = False
    
    # Создаем минимальную заглушку
    class MetricsCollectorStub:
        def record_feedback(self, *args, **kwargs):
            pass
        def record_question_processing(self, *args, **kwargs):
            pass
        def record_error(self, *args, **kwargs):
            pass
        def record(self, *args, **kwargs):
            pass
    
    metrics_collector = MetricsCollectorStub()

@dataclass
class FeedbackData:
    """Модель обратной связи"""
    id: str
    conversation_id: int  # ✅ Исправлено: было session_id
    message_id: int
    rating: int
    helpful: Optional[bool]
    comment: Optional[str]
    meta_data: Dict[str, Any]  # ✅ Исправлено: было context
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return asdict(self)


class FeedbackService:
    """Сервис для работы с обратной связью"""
    
    def __init__(self):
        self.metrics = metrics_collector
        self.cache = {}  # Простой кэш для быстрого доступа
    
    async def save_feedback(
        self,
        conversation_id: int,  # ✅ Исправлено: было session_id
        message_id: int,
        rating: int,
        helpful: Optional[bool] = None,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Сохранение обратной связи с валидацией и обогащением
        
        Args:
            conversation_id: ID диалога
            message_id: ID сообщения
            rating: Рейтинг 1-5
            helpful: Полезность
            comment: Комментарий
            metadata: Дополнительные метаданные
        
        Returns:
            Результат операции
        """
        try:
            # Валидация входных данных
            if not self._validate_feedback(rating, helpful, comment):
                return {
                    "success": False,
                    "error": "Некорректные данные обратной связи"
                }
            
            # Создание ID обратной связи
            feedback_id = str(uuid.uuid4())
            
            # Подготовка метаданных
            meta_data = metadata or {}
            meta_data.update({
                "feedback_id": feedback_id,
                "timestamp": datetime.now().isoformat(),
                "source": "chat_api"
            })
            
            # Создание модели БД
            feedback = FeedbackModel(
                conversation_id=conversation_id,  # ✅ Исправлено
                message_id=message_id,
                rating=rating,
                helpful=helpful,
                comment=comment,
                meta_data=meta_data  # ✅ Исправлено
            )
            
            # Сохранение в базу данных
            async with db_manager.get_session() as session:
                session.add(feedback)
                await session.commit()
                
                # Получаем ID созданной записи
                await session.refresh(feedback)
            
            # Создаем объект обратной связи
            feedback_data = FeedbackData(
                id=feedback_id,
                conversation_id=conversation_id,
                message_id=message_id,
                rating=rating,
                helpful=helpful,
                comment=comment,
                meta_data=meta_data,
                created_at=datetime.now()
            )
            
            # Кэшируем
            self.cache[feedback_id] = feedback_data
            
            # Отправляем метрики
            await self._send_metrics(feedback_data)
            
            # Логируем успех
            logger.info(
                "Обратная связь сохранена",
                extra={
                    "feedback_id": feedback_id,
                    "conversation_id": conversation_id,
                    "rating": rating,
                    "helpful": helpful
                }
            )
            
            return {
                "success": True,
                "feedback_id": feedback_id,
                "message": "Обратная связь успешно сохранена"
            }
            
        except Exception as e:
            logger.error(
                "Ошибка сохранения обратной связи",
                exc_info=True,
                extra={
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "error": str(e)
                }
            )
            
            return {
                "success": False,
                "error": str(e),
                "message": "Не удалось сохранить обратную связь"
            }
    
    # feedback_service.py
    async def submit_feedback(
        self,
        conversation_id: int,
        message_id: int,
        rating: int,
        helpful: Optional[bool] = None,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Полный процесс обработки обратной связи
        
        Args:
            conversation_id: ID диалога
            message_id: ID сообщения
            rating: Рейтинг 1-5
            helpful: Полезность
            comment: Комментарий
            metadata: Дополнительные метаданные
        
        Returns:
            Результат операции
        """
        # Записываем метрики
        if self.metrics:
            try:
                self.metrics.record_feedback(rating, helpful)
            except:
                pass
        
        # Сохраняем в БД
        result = await self.save_feedback(
            conversation_id=conversation_id,
            message_id=message_id,
            rating=rating,
            helpful=helpful,
            comment=comment,
            metadata=metadata
        )
        
        if not result["success"]:
            return result
        
        # Отправляем в RL агент
        try:
            from .rl_agent import rl_agent  # Относительный импорт
            #from app.services.enhancements.rl_agent import rl_agent
            
            await rl_agent.receive_feedback(
                conversation_id=conversation_id,
                message_id=message_id,
                rating=rating,
                helpful=helpful,
                meta_data=metadata
            )
            logger.info(f"RL агент обновлен для conversation_id={conversation_id}")
            
        except ImportError as e:
            logger.warning(f"RL агент не доступен (ImportError): {e}")
        except Exception as e:
            logger.error(f"Ошибка отправки в RL агент: {e}")
        
        return result
    
    async def get_feedback(
        self,
        feedback_id: str
    ) -> Optional[FeedbackData]:
        """
        Получение обратной связи по ID
        
        Args:
            feedback_id: ID обратной связи
        
        Returns:
            Объект обратной связи или None
        """
        # Проверяем кэш
        if feedback_id in self.cache:
            return self.cache[feedback_id]
        
        try:
            async with db_manager.get_session() as session:
                from sqlalchemy import select
                
                result = await session.execute(
                    select(FeedbackModel).where(
                        FeedbackModel.meta_data["feedback_id"].astext == feedback_id
                    )
                )
                
                feedback = result.scalar_one_or_none()
                
                if feedback:
                    return FeedbackData(
                        id=feedback_id,
                        conversation_id=feedback.conversation_id,  # ✅ Исправлено
                        message_id=feedback.message_id,
                        rating=feedback.rating,
                        helpful=feedback.helpful,
                        comment=feedback.comment,
                        meta_data=feedback.meta_data or {},  # ✅ Исправлено
                        created_at=feedback.created_at or datetime.now()
                    )
        
        except Exception as e:
            logger.error(f"Ошибка получения обратной связи: {e}")
        
        return None
    
    async def get_conversation_feedback(
        self,
        conversation_id: int,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Получение обратной связи по диалогу
        
        Args:
            conversation_id: ID диалога
            limit: Лимит
            offset: Смещение
        
        Returns:
            Список обратной связи с пагинацией
        """
        try:
            async with db_manager.get_session() as session:
                from sqlalchemy import select, func, desc
                
                # Общее количество
                total = await session.scalar(
                    select(func.count()).where(
                        FeedbackModel.conversation_id == conversation_id  # ✅ Исправлено
                    )
                )
                
                # Данные с пагинацией
                result = await session.execute(
                    select(FeedbackModel)
                    .where(FeedbackModel.conversation_id == conversation_id)  # ✅ Исправлено
                    .order_by(desc(FeedbackModel.created_at))
                    .offset(offset)
                    .limit(limit)
                )
                
                feedbacks = result.scalars().all()
                
                feedback_list = []
                for fb in feedbacks:
                    feedback_list.append({
                        "id": fb.meta_data.get("feedback_id", str(fb.id)),
                        "conversation_id": fb.conversation_id,  # ✅ Исправлено
                        "message_id": fb.message_id,
                        "rating": fb.rating,
                        "helpful": fb.helpful,
                        "comment": fb.comment,
                        "created_at": fb.created_at.isoformat() if fb.created_at else None,
                        "meta_data": fb.meta_data or {}  # ✅ Исправлено
                    })
                
                return {
                    "conversation_id": conversation_id,
                    "feedbacks": feedback_list,
                    "total": total or 0,
                    "limit": limit,
                    "offset": offset,
                    "has_more": total and (offset + limit < total)
                }
        
        except Exception as e:
            logger.error(f"Ошибка получения обратной связи диалога: {e}")
            return {
                "conversation_id": conversation_id,
                "feedbacks": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "has_more": False,
                "error": str(e)
            }
    
    async def get_statistics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Получение статистики обратной связи
        
        Args:
            days: Количество дней для анализа
        
        Returns:
            Статистика обратной связи
        """
        try:
            async with db_manager.get_session() as session:
                from sqlalchemy import select, func, and_
                from datetime import datetime, timedelta
                
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Общая статистика
                total = await session.scalar(
                    select(func.count()).select_from(FeedbackModel)
                )
                
                # Средний рейтинг
                avg_rating = await session.scalar(
                    select(func.avg(FeedbackModel.rating))
                )
                
                # Распределение рейтингов
                rating_counts = {}
                for rating in range(1, 6):
                    count = await session.scalar(
                        select(func.count()).where(FeedbackModel.rating == rating)
                    )
                    rating_counts[f"rating_{rating}"] = count or 0
                
                # Полезность
                helpful_stats = {
                    "helpful": await session.scalar(
                        select(func.count()).where(FeedbackModel.helpful == True)
                    ) or 0,
                    "not_helpful": await session.scalar(
                        select(func.count()).where(FeedbackModel.helpful == False)
                    ) or 0,
                    "unknown": await session.scalar(
                        select(func.count()).where(FeedbackModel.helpful == None)
                    ) or 0
                }
                
                # Последние N дней
                recent_total = await session.scalar(
                    select(func.count()).where(
                        FeedbackModel.created_at >= cutoff_date
                    )
                )
                
                recent_avg = await session.scalar(
                    select(func.avg(FeedbackModel.rating)).where(
                        FeedbackModel.created_at >= cutoff_date
                    )
                )
                
                return {
                    "overall": {
                        "total_feedback": total or 0,
                        "average_rating": float(avg_rating or 0) if avg_rating else 0,
                        "rating_distribution": rating_counts,
                        "helpful_distribution": helpful_stats
                    },
                    f"last_{days}_days": {
                        "total_feedback": recent_total or 0,
                        "average_rating": float(recent_avg or 0) if recent_avg else 0,
                        "period": f"{days} дней"
                    },
                    "calculated_at": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {
                "error": str(e),
                "overall": {
                    "total_feedback": 0,
                    "average_rating": 0,
                    "rating_distribution": {},
                    "helpful_distribution": {}
                }
            }
    
    def _validate_feedback(
        self,
        rating: int,
        helpful: Optional[bool],
        comment: Optional[str]
    ) -> bool:
        """Валидация данных обратной связи"""
        # Валидация рейтинга
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            return False
        
        # Валидация helpful (если указан)
        if helpful is not None and not isinstance(helpful, bool):
            return False
        
        # Валидация комментария
        if comment is not None:
            if not isinstance(comment, str):
                return False
            if len(comment) > 1000:  # Максимальная длина
                return False
        
        return True
    
    async def _send_metrics(self, feedback: FeedbackData) -> None:
        """Отправка метрик обратной связи"""
        if not METRICS_AVAILABLE:
            return  # Просто выходим, если метрики недоступны
        
        try:
            metrics_collector.record_feedback(
                rating=feedback.rating,
                helpful=feedback.helpful,
                comment=feedback.comment
            )
            logger.debug(f"📊 Метрики фидбека отправлены: rating={feedback.rating}")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось отправить метрики: {e}")


# Глобальный экземпляр для использования
feedback_service = FeedbackService()