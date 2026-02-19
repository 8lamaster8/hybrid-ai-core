import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.core.logging import logger
from app.core.brain import brain

# ПРАВИЛЬНЫЕ ИМПОРТЫ - используем app.api.endpoints
try:
    from app.api.endpoints.chat import router as chat_router
    from app.api.endpoints.knowledge import router as knowledge_router
    from app.api.endpoints.system import router as system_router
    logger.info("✅ Все роутеры импортированы из app.api.endpoints")
except ImportError as e:
    logger.error(f"❌ Ошибка импорта роутеров: {e}")
    logger.error("Проверьте что файлы лежат в app/api/endpoints/")
    logger.error("Или измените импорты на правильные пути")
    raise

async def startup_event():
    """Событие запуска приложения"""
    logger.info("Starting AI Core application...")
    
    # Инициализируем AI Brain
    await brain.initialize()
    logger.info("AI Brain initialized")

async def shutdown_event():
    """Событие остановки приложения"""
    logger.info("Shutting down AI Core application...")

def create_app() -> FastAPI:
    """Создание FastAPI приложения"""
    app = FastAPI(
        title="AI Knowledge Assistant",
        description="Production-ready AI system with knowledge base",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Events
    app.add_event_handler("startup", startup_event)
    app.add_event_handler("shutdown", shutdown_event)
    
    # Подключаем ВСЕ роутеры - больше никаких дубликатов!
    app.include_router(chat_router)
    app.include_router(knowledge_router)
    app.include_router(system_router)
    
    # ТОЛЬКО корневой эндпоинт - больше никаких дубликатов!
    @app.get("/")
    async def root():
        return {"message": "AI Knowledge Assistant API", "status": "running"}
    
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=settings.API_WORKERS if not settings.DEBUG else 1
    )