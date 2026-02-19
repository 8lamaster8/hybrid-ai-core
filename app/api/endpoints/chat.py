"""
API для чата с AI
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.core.brain import brain
from app.core.logging import logger
from app.services.feedback.feedback_service import FeedbackService

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# ========== МОДЕЛИ ЗАПРОСОВ/ОТВЕТОВ ==========

class ChatRequest(BaseModel):
    """Модель запроса чата"""
    message: str = Field(..., min_length=1, max_length=2000, description="Сообщение пользователя")
    session_id: Optional[str] = Field(None, description="ID сессии (если нет - создается новая)")
    use_knowledge: bool = Field(True, description="Использовать базу знаний")

class ChatResponse(BaseModel):
    """Модель ответа чата"""
    answer: str = Field(..., description="Ответ AI")
    session_id: str = Field(..., description="ID сессии")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность AI в ответе")
    processing_time_ms: float = Field(..., ge=0.0, description="Время обработки в миллисекундах")
    suggestions: List[str] = Field(default_factory=list, description="Предложения для продолжения диалога")
    message_id: Optional[int] = Field(None, description="ID сообщения в истории")

class FeedbackRequest(BaseModel):
    """Модель запроса обратной связи"""
    conversation_id: int = Field(..., ge=0, description="ID беседы в БД")
    message_id: int = Field(..., ge=0, description="ID сообщения в истории сессии")
    rating: int = Field(..., ge=1, le=5, description="Рейтинг (1-5)")
    helpful: Optional[bool] = Field(None, description="Был ли ответ полезен")
    comment: Optional[str] = Field(None, max_length=1000, description="Комментарий пользователя")
    context: Optional[Dict[str, Any]] = Field(None, description="Дополнительный контекст")

    @validator('rating')
    def validate_rating(cls, v):
        if v not in range(1, 6):
            raise ValueError('Рейтинг должен быть от 1 до 5')
        return v

class FeedbackResponse(BaseModel):
    """Модель ответа обратной связи"""
    success: bool = Field(..., description="Успешность операции")
    message: str = Field(..., description="Сообщение о результате")
    feedback_id: Optional[str] = Field(None, description="ID сохраненной обратной связи")

# ========== ЭНДПОИНТЫ ==========

@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest) -> ChatResponse:
    """
    Обработка вопроса пользователя
    
    - Принимает вопрос и опционально session_id
    - Возвращает ответ AI с метаданными
    """
    try:
        logger.info(
            f"Запрос чата: '{request.message[:50]}...'",
            extra={"session_id": request.session_id, "use_knowledge": request.use_knowledge}
        )
        
        response = await brain.ask(
            question=request.message,
            session_id=request.session_id,
            use_knowledge=request.use_knowledge
        )
        
        return ChatResponse(
            answer=response.get("answer", "Извините, не удалось получить ответ"),
            session_id=request.session_id or response.get("session_id", "new_session"),
            confidence=response.get("confidence", 0.0),
            processing_time_ms=response.get("processing_time_ms", 0),
            suggestions=response.get("followup_suggestions", []),
            message_id=response.get("message_id")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обработки вопроса: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка при обработке запроса"
        )

@router.post("/conversations", response_model=Dict[str, Any])
async def create_conversation(conversation_data: Dict[str, Any]):
    """Создание новой беседы"""
    try:
        from app.infrastructure.database import db_manager, Conversation
        
        async with db_manager.get_session() as session:
            conversation = Conversation(
                session_id=conversation_data.get("session_id"),
                title=conversation_data.get("title", "Новая беседа")
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            
            return {"success": True, "id": conversation.id, "message": "Беседа создана"}
            
    except Exception as e:
        logger.error(f"Ошибка создания беседы: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations/by_session/{session_id}")
async def get_conversation_by_session(session_id: str, include_messages: bool = False):
    """Получение беседы по session_id с опциональным включением сообщений"""
    try:
        from app.infrastructure.database import db_manager, Conversation, Message
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        
        async with db_manager.get_session() as session:
            query = select(Conversation).where(Conversation.session_id == session_id)
            
            # Если нужно включить сообщения
            if include_messages:
                query = query.options(selectinload(Conversation.messages))
            
            query = query.order_by(Conversation.created_at.desc()).limit(1)
            
            result = await session.execute(query)
            conversation = result.scalar_one_or_none()
            
            if not conversation:
                return {
                    "found": False,
                    "message": "No conversation found for this session"
                }
            
            response = {
                "found": True,
                "conversation": {
                    "id": conversation.id,
                    "session_id": conversation.session_id,
                    "title": conversation.title,
                    "message_count": conversation.message_count,
                    "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
                    "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None
                }
            }
            
            # Добавляем сообщения если нужно
            if include_messages and conversation.messages:
                response["conversation"]["messages"] = [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": msg.created_at.isoformat() if msg.created_at else None
                    }
                    for msg in conversation.messages
                ]
            
            return response
            
    except Exception as e:
        logger.error(f"Ошибка получения беседы: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Сохранение обратной связи пользователя
    """
    try:
        logger.info(
            "Получена обратная связь",
            extra={
                "conversation_id": request.conversation_id,
                "message_id": request.message_id,
                "rating": request.rating,
                "helpful": request.helpful
            }
        )
        
        # Используем глобальный экземпляр
        from app.services.feedback.feedback_service import feedback_service
        
        result = await feedback_service.submit_feedback(
            conversation_id=request.conversation_id,  # Исправлено: используем conversation_id
            message_id=request.message_id,
            rating=request.rating,
            helpful=request.helpful,
            comment=request.comment,
            metadata=request.context  # Передаем context как metadata
        )
        
        if result["success"]:
            return FeedbackResponse(
                success=True,
                message="Спасибо за обратную связь! Это поможет улучшить ответы.",
                feedback_id=result.get("feedback_id")
            )
        else:
            logger.error(f"Ошибка сохранения feedback: {result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Не удалось сохранить обратную связь")
            )
    except Exception as e:
        logger.error(f"Ошибка обработки обратной связи: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Проверка здоровья системы чата
    
    - Проверяет доступность всех компонентов
    - Возвращает подробный статус
    """
    try:
        health_status = await brain.health_check()
        
        all_healthy = all(health_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": health_status,
            "version": "1.0.0",
            "uptime": _get_uptime()
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="System health check failed"
        )

@router.get("/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    limit: int = 20,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Получение истории сессии
    
    - Возвращает историю сообщений для указанной сессии
    - Поддерживает пагинацию через limit/offset
    """
    try:
        history = await brain.get_session_history(session_id, limit)
        
        return {
            "session_id": session_id,
            "history": history[offset:offset + limit],
            "total": len(history),
            "returned": min(limit, len(history) - offset),
            "has_more": len(history) > offset + limit
        }
    
    except Exception as e:
        logger.error(f"Ошибка получения истории сессии: {e}")
        raise HTTPException(
            status_code=404,
            detail="Сессия не найдена или произошла ошибка"
        )

@router.delete("/session/{session_id}")
async def clear_session(session_id: str) -> Dict[str, Any]:
    """
    Очистка сессии
    
    - Удаляет историю сообщений для указанной сессии
    - Сохраняет метаданные сессии
    """
    try:
        success = await brain.clear_session(session_id)
        
        if success:
            logger.info(f"Сессия очищена: {session_id}")
            return {
                "success": True,
                "message": f"Сессия {session_id} очищена"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="Сессия не найдена"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка очистки сессии: {e}")
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка при очистке сессии"
        )

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

async def _update_rl_agent(feedback: FeedbackRequest) -> None:
    """Обновление RL агента на основе обратной связи"""
    try:
        # Проверяем доступность RL агента
        if hasattr(brain, 'rl_agent') and brain.rl_agent:
            # Получаем историю сессии для контекста
            session_history = await brain.get_session_history(feedback.session_id)
            
            if session_history and len(session_history) > feedback.message_id:
                message_data = session_history[feedback.message_id]
                
                # Вычисляем награду на основе обратной связи
                reward = _calculate_reward(feedback.rating, feedback.helpful)
                
                # Логируем обновление RL агента
                logger.info(
                    f"Обновление RL агента: reward={reward}",
                    extra={"rating": feedback.rating, "helpful": feedback.helpful}
                )
                
                # Здесь должна быть логика обновления RL агента
                # await brain.rl_agent.update_with_feedback(...)
    except Exception as e:
        logger.warning(f"Не удалось обновить RL агент: {e}")

def _calculate_reward(rating: int, helpful: Optional[bool]) -> float:
    """Вычисление награды для RL агента"""
    if helpful is False:
        return -1.0
    elif helpful is True:
        return 1.0
    
    # Маппинг рейтинга на награду
    rating_rewards = {
        1: -0.8,
        2: -0.4,
        3: 0.0,
        4: 0.6,
        5: 1.0
    }
    return rating_rewards.get(rating, 0.0)

def _get_uptime() -> str:
    """Получение времени работы системы"""
    # В реальной системе здесь вычисление uptime
    return "0:00:00"