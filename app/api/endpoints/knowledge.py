"""
API для управления базой знаний
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import List, Optional, Dict, Any
import json
from pydantic import BaseModel, validator
from pathlib import Path
from datetime import datetime
import logging

from app.core.brain import brain
from app.core.logging import logger

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


class KnowledgeAddRequest(BaseModel):
    """Модель запроса для добавления знаний"""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    source: str = "api"
    tags: Optional[List[str]] = None
    
    @validator('metadata', 'tags', pre=True)
    def parse_string_fields(cls, v):
        """Парсит строки JSON в объекты Python"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v


@router.post("/add")
async def add_knowledge(
    content: str = Form(...),
    metadata: Optional[str] = Form(None),
    source: str = Form("api"),
    tags: Optional[str] = Form(None)
):
    """Добавление знаний в систему"""
    try:
        # Парсим JSON строки
        metadata_dict = json.loads(metadata) if metadata else None
        tags_list = json.loads(tags) if tags else None
        
        result = await brain.add_knowledge(
            content=content,
            metadata=metadata_dict,
            source=source,
            tags=tags_list
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Добавлено {result['chunk_count']} фрагментов знаний",
                "data": result
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON in metadata or tags")
    except Exception as e:
        logger.error(f"API error in add_knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-add")
async def batch_add_knowledge(
    background_tasks: BackgroundTasks,
    content: List[str] = Form(...),
    metadata: Optional[str] = Form(None),
    source: str = Form("api"),
    tags: Optional[str] = Form(None)
):
    """Пакетное добавление знаний (асинхронное)"""
    try:
        parsed_metadata = json.loads(metadata) if metadata else None
        parsed_tags = json.loads(tags) if tags else None
        
        # Запускаем в фоне, чтобы не блокировать ответ
        background_tasks.add_task(
            brain.add_knowledge,
            content=content,
            metadata=parsed_metadata,
            source=source,
            tags=parsed_tags
        )
        
        return {
            "success": True,
            "message": "Задача на добавление знаний запущена в фоне",
            "items_count": len(content)
        }
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON in metadata or tags")
    except Exception as e:
        logger.error(f"API error in batch_add_knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    source: str = Form("upload"),
    tags: Optional[str] = Form(None)
):
    """Загрузка документа для добавления в базу знаний"""
    try:
        # Проверка расширения файла
        allowed_extensions = {'.txt', '.pdf', '.docx', '.json', '.csv', '.md'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Неподдерживаемый формат файла. Допустимые: {', '.join(allowed_extensions)}"
            )
        
        # Проверка размера файла (макс 10MB)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400, 
                detail="Файл слишком большой. Максимальный размер: 10MB"
            )
        
        text = content.decode("utf-8", errors='replace')
        
        # Убираем BOM если есть
        if text.startswith('\ufeff'):
            text = text[1:]
        
        parsed_tags = json.loads(tags) if tags else None
        
        logger.info(f"Начинаем обработку файла: {file.filename} ({len(text)} символов)")
        
        result = await brain.add_knowledge(
            content=text,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(content),
                "upload_time": datetime.utcnow().isoformat()
            },
            source=source,
            tags=parsed_tags
        )
        
        if result["success"]:
            logger.info(f"Файл {file.filename} успешно обработан: {result['chunk_count']} чанков")
            return {
                "success": True,
                "message": f"Документ '{file.filename}' обработан, добавлено {result['chunk_count']} фрагментов",
                "data": result
            }
        else:
            logger.error(f"Ошибка обработки документа: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get("error", "Ошибка обработки"))
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Невозможно декодировать файл как UTF-8")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail=f"Некорректный JSON в тегах: {e}")
    except Exception as e:
        logger.error(f"API error in upload_document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {str(e)}")


@router.get("/search")
async def search_knowledge(
    query: str,
    top_k: int = 10,
    filters: Optional[str] = None
):
    """Поиск в базе знаний"""
    try:
        parsed_filters = json.loads(filters) if filters else None
        
        results = await brain.search_knowledge(
            query=query,
            top_k=top_k,
            filters=parsed_filters
        )
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results)
        }
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON in filters")
    except Exception as e:
        logger.error(f"API error in search_knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_knowledge_info():
    """Получение информации о базе знаний"""
    try:
        system_info = await brain.get_system_info()
        kb_info = system_info.get("knowledge_base", {})
        
        return {
            "success": True,
            "knowledge_base": kb_info
        }
    
    except Exception as e:
        logger.error(f"API error in get_knowledge_info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_knowledge_base():
    """Очистка базы знаний (только для админов в разработке)"""
    try:
        # В продакшене этот endpoint должен быть защищен
        if brain.vector_store:
            await brain.vector_store.clear()
            return {"success": True, "message": "База знаний очищена"}
        else:
            raise HTTPException(status_code=500, detail="Векторное хранилище не инициализировано")
    
    except Exception as e:
        logger.error(f"API error in clear_knowledge_base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export")
async def export_knowledge(format: str = "json"):
    """Экспорт базы знаний"""
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
        # Здесь должен быть код экспорта
        # Пока заглушка
        return {
            "success": True,
            "message": f"Экспорт в формате {format}",
            "format": format,
            "data": []
        }
    
    except Exception as e:
        logger.error(f"API error in export_knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))