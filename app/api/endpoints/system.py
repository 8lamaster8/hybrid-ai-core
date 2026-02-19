"""
API для мониторинга и управления системой
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import psutil
import os

from app.core.brain import brain
from app.core.logging import logger
from app.monitoring.metrics import get_metrics

router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/health")
async def health_check():
    """Проверка здоровья системы"""
    try:
        health = await brain.health_check()
        
        # Проверяем, что все компоненты здоровы
        all_healthy = all(health.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": health,
            "all_healthy": all_healthy,
            "version": "1.0.0"
        }
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="System unhealthy")


@router.get("/metrics")
async def get_system_metrics():
    """Получение метрик системы"""
    try:
        metrics = get_metrics()
        return metrics
    
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_system_info():
    """Получение полной информации о системе"""
    try:
        info = await brain.get_system_info()
        return info
    
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_system_stats():
    """Получение статистики системы"""
    try:
        system_info = await brain.get_system_info()
        stats = system_info.get("stats", {})
        
        return {
            "questions_processed": stats.get("questions_processed", 0),
            "sessions_created": stats.get("sessions_created", 0),
            "cache_hits": stats.get("cache_hits", 0),
            "errors": stats.get("errors", 0),
            "average_processing_time_ms": (
                stats["total_processing_time_ms"] / stats["questions_processed"]
                if stats["questions_processed"] > 0 else 0
            )
        }
    
    except Exception as e:
        logger.error(f"System stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resources")
async def get_system_resources():
    """Получение информации об использовании ресурсов"""
    try:
        # Информация о памяти
        memory = psutil.virtual_memory()
        
        # Информация о CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Информация о диске
        disk = psutil.disk_usage('/')
        
        # Информация о процессе
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()
        
        return {
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": memory.percent,
                "used_gb": round(memory.used / (1024**3), 2),  # ДОБАВЛЕНО
                "process_rss_mb": round(process_memory.rss / (1024**2), 2)
            },
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count()
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": disk.percent
            }
        }
    
    except Exception as e:
        logger.error(f"Resources error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-cache")
async def clear_cache():
    """Очистка кэша системы"""
    try:
        if brain.cache:
            await brain.cache.clear()
            return {"success": True, "message": "Кэш очищен"}
        else:
            raise HTTPException(status_code=500, detail="Кэш не инициализирован")
    
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart")
async def restart_system():
    """Перезапуск системы (только для админов)"""
    try:
        # В реальной системе здесь должен быть auth check
        logger.warning("System restart requested via API")
        
        return {
            "success": True,
            "message": "Перезапуск инициирован",
            "note": "В реальной системе здесь будет перезапуск"
        }
    
    except Exception as e:
        logger.error(f"Restart error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/{session_id}")
async def get_session_memory(session_id: str, limit: int = 20):
    """Получение истории сессии"""
    try:
        history = await brain.get_session_history(session_id, limit)
        return {
            "session_id": session_id,
            "history": history,
            "count": len(history)
        }
    
    except Exception as e:
        logger.error(f"Get session memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/{session_id}")
async def clear_session_memory(session_id: str):
    """Очистка памяти сессии"""
    try:
        success = await brain.clear_session(session_id)
        
        if success:
            return {"success": True, "message": f"Сессия {session_id} очищена"}
        else:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    except Exception as e:
        logger.error(f"Clear session memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_recent_logs(limit: int = 100, level: str = "ERROR"):
    """Получение последних логов (для отладки)"""
    try:
        # В реальной системе здесь запрос к системе логирования
        # Пока заглушка
        return {
            "success": True,
            "limit": limit,
            "level": level,
            "logs": []
        }
    
    except Exception as e:
        logger.error(f"Get logs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))