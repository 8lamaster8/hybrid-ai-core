"""
Health checks для системы
"""
from typing import Dict, Any, List
import asyncio
from datetime import datetime, timedelta
import socket
import aiohttp
import psutil

from app.core.logging import logger
from app.core.config import settings
from app.infrastructure.cache import cache
from app.services.knowledge.vector_store import vector_store


class HealthChecker:
    """Комплексная проверка здоровья системы"""
    
    def __init__(self, brain_instance=None, check_interval: int = 30):
        self.check_interval = check_interval
        self.brain_instance = brain_instance  # Принимаем brain как параметр
        self.last_check = None
        self.last_results = {}
        self._check_lock = asyncio.Lock()
        
    def set_brain_instance(self, brain_instance):
        """Устанавливаем экземпляр brain позже, после его создания"""
        self.brain_instance = brain_instance
    
    async def perform_full_health_check(self) -> Dict[str, Any]:
        """Выполнение комплексной проверки здоровья"""
        async with self._check_lock:
            start_time = datetime.now()
            
            try:
                checks = {}
                
                # 1. Проверка AI Brain (только если brain_instance установлен)
                if self.brain_instance:
                    brain_health = await self.brain_instance.health_check()
                    checks["brain"] = {
                        "status": all(brain_health.values()),
                        "components": brain_health
                    }
                else:
                    checks["brain"] = {
                        "status": False,
                        "message": "Brain instance not available"
                    }
                
                # 2. Проверка базы данных
                checks["database"] = await self._check_database()
                
                # 3. Проверка кэша
                checks["cache"] = await self._check_cache()
                
                # 4. Проверка векторного хранилища
                checks["vector_store"] = await self._check_vector_store()
                
                # 5. Проверка внешних зависимостей
                checks["external"] = await self._check_external_dependencies()
                
                # 6. Проверка системных ресурсов
                checks["resources"] = await self._check_system_resources()
                
                # 7. Проверка API
                checks["api"] = await self._check_api()
                
                # Общий статус
                all_healthy = all(
                    check.get("status", False)
                    for check in checks.values()
                )
                
                checks["overall"] = {
                    "status": all_healthy,
                    "timestamp": datetime.now().isoformat(),
                    "check_duration_seconds": (datetime.now() - start_time).total_seconds()
                }
                
                # Сохраняем результаты
                self.last_check = datetime.now()
                self.last_results = checks
                
                # Логируем результаты
                if all_healthy:
                    logger.info("✅ Health check passed")
                else:
                    failed = [
                        name for name, check in checks.items()
                        if not check.get("status", True)
                    ]
                    logger.warning(f"⚠️ Health check failed for: {failed}")
                
                return checks
                
            except Exception as e:
                logger.error(f"❌ Health check error: {e}", exc_info=True)
                
                return {
                    "overall": {
                        "status": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                }
    
    async def _check_database(self) -> Dict[str, Any]:
        """Проверка базы данных"""
        try:
            from app.infrastructure.database import db_manager
            
            async with db_manager.get_session() as session:
                result = await session.execute("SELECT 1")
                success = result.scalar() == 1
            
            return {
                "status": success,
                "message": "Database connection successful" if success else "Database connection failed"
            }
            
        except Exception as e:
            return {
                "status": False,
                "error": str(e),
                "message": "Database connection failed"
            }
    
    async def _check_cache(self) -> Dict[str, Any]:
        """Проверка кэша"""
        try:
            stats = await cache.get_stats()
            
            return {
                "status": stats.get("redis_available", False) or stats.get("memory_cache_size", 0) > 0,
                "stats": stats,
                "message": "Cache is available"
            }
            
        except Exception as e:
            return {
                "status": False,
                "error": str(e),
                "message": "Cache check failed"
            }
    
    async def _check_vector_store(self) -> Dict[str, Any]:
        """Проверка векторного хранилища"""
        try:
            if vector_store:
                info = await vector_store.get_info()
                
                return {
                    "status": "total_chunks" in info,
                    "info": info,
                    "message": "Vector store is available"
                }
            else:
                return {
                    "status": False,
                    "message": "Vector store not initialized"
                }
            
        except Exception as e:
            return {
                "status": False,
                "error": str(e),
                "message": "Vector store check failed"
            }
    
    async def _check_external_dependencies(self) -> Dict[str, Any]:
        """Проверка внешних зависимостей"""
        checks = {}
        
        # Проверка необходимых портов
        checks["ports"] = await self._check_ports()
        
        return {
            "status": all(checks.values()) if checks else True,
            "checks": checks,
            "message": "External dependencies check completed"
        }
    
    async def _check_ports(self) -> Dict[str, bool]:
        """Проверка доступности портов"""
        ports_to_check = {
            "postgres": 5432,
            "redis": 6379,
            "chromadb": 8000
        }
        
        results = {}
        
        for service, port in ports_to_check.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                results[service] = result == 0
            except:
                results[service] = False
        
        return results
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Проверка системных ресурсов"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage('/')
            
            checks = {
                "memory_ok": memory.percent < 90,
                "cpu_ok": cpu_percent < 80,
                "disk_ok": disk.percent < 90
            }
            
            return {
                "status": all(checks.values()),
                "metrics": {
                    "memory_percent": memory.percent,
                    "cpu_percent": cpu_percent,
                    "disk_percent": disk.percent
                },
                "message": "System resources within limits"
            }
            
        except Exception as e:
            return {
                "status": False,
                "error": str(e),
                "message": "System resources check failed"
            }
    
    async def _check_api(self) -> Dict[str, Any]:
        """Проверка API endpoints"""
        try:
            async with aiohttp.ClientSession() as session:
                # Проверка health endpoint
                url = f"http://localhost:{settings.API_PORT}/api/v1/system/health"
                async with session.get(url, timeout=5) as response:
                    data = await response.json()
                    api_healthy = data.get("status") == "healthy"
                
                return {
                    "status": api_healthy,
                    "response": data,
                    "message": "API is responding"
                }
                
        except Exception as e:
            return {
                "status": False,
                "error": str(e),
                "message": "API check failed"
            }
    
    async def start_periodic_checks(self):
        """Запуск периодических проверок здоровья"""
        if self.check_interval <= 0:
            return
            
        logger.info(f"Starting periodic health checks every {self.check_interval} seconds")
        
        while True:
            try:
                await self.perform_full_health_check()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")
                await asyncio.sleep(60)  # Подождать минуту при ошибке
    
    async def get_service_status(self) -> Dict[str, str]:
        """
        Получение простого статуса сервисов
        
        Returns:
            Статус сервисов в виде словаря
        """
        checks = await self.perform_full_health_check()
        
        status_map = {}
        for service, check in checks.items():
            if service != "overall":
                status_map[service] = "healthy" if check.get("status", False) else "unhealthy"
        
        return status_map


# Создаем экземпляр без brain (будет установлен позже)
health_checker = HealthChecker(brain_instance=None)


async def start_health_monitoring(brain_instance=None):
    """Запуск мониторинга здоровья"""
    if hasattr(settings, 'METRICS_ENABLED') and settings.METRICS_ENABLED:
        # Устанавливаем brain instance
        if brain_instance and hasattr(health_checker, 'set_brain_instance'):
            health_checker.set_brain_instance(brain_instance)
        
        # Запускаем периодические проверки в фоне
        try:
            asyncio.create_task(health_checker.start_periodic_checks())
            logger.info("Health monitoring started")
        except Exception as e:
            logger.error(f"Failed to start health monitoring: {e}")