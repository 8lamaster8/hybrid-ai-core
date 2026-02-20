"""
🧠 ENGRAM ПАМЯТЬ - Долговременное хранилище знаний
Простая и надёжная версия
"""

import asyncio
import json
import os
import sqlite3
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime

from appp.core.logging import logger


class EngramService:
    """
    Engram память - быстрый доступ к часто используемым знаниям.
    Хранит ключ -> текст, метаданные, частоту использования.
    """
    
    def __init__(
        self,
        db_path: str = "./data/engram/engram.db",
        max_records: int = 100000
    ):
        self.db_path = db_path
        self.max_records = max_records
        self.cache: Dict[str, Dict] = {}
        
        # Статистика
        self.stats = {
            'total_records': 0,
            'hits': 0,
            'misses': 0,
            'queries': 0,
            'errors': 0
        }
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        logger.info(f"🧠 EngramService создан (db: {db_path})")
    
    async def initialize(self):
        """Загрузка данных из SQLite"""
        logger.info("🔄 Загрузка Engram памяти...")
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._load_from_db
            )
            self.stats['total_records'] = len(self.cache)
            logger.info(f"✅ Engram загружен: {len(self.cache)} записей")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки Engram: {e}")
            return False
    
    def _load_from_db(self):
        """Загрузка из SQLite (синхронно)"""
        # Создаём таблицу, если нет
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS engram (
                key TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                access_count INTEGER,
                last_access REAL,
                created_at TEXT
            )
        ''')
        conn.commit()
        
        # Читаем данные
        c.execute('SELECT key, content, metadata, access_count, last_access, created_at FROM engram')
        rows = c.fetchall()
        for row in rows:
            key, content, metadata_json, access_count, last_access, created_at = row
            self.cache[key] = {
                'key': key,
                'content': content,
                'metadata': json.loads(metadata_json),
                'access_count': access_count,
                'last_access': last_access,
                'created_at': created_at
            }
        conn.close()
    
    async def save(self):
        """Сохранение состояния в SQLite"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._save_to_db
            )
            logger.debug(f"💾 Engram сохранён: {len(self.cache)} записей")
        except Exception as e:
            logger.error(f"Ошибка сохранения Engram: {e}")
            self.stats['errors'] += 1
    
    def _save_to_db(self):
        """Синхронное сохранение в БД"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM engram')
        for key, record in self.cache.items():
            c.execute('''
                INSERT INTO engram (key, content, metadata, access_count, last_access, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                key,
                record['content'],
                json.dumps(record['metadata'], ensure_ascii=False),
                record['access_count'],
                record['last_access'],
                record['created_at']
            ))
        conn.commit()
        conn.close()
    
    async def store(
        self,
        key: str,
        content: str,
        metadata: Optional[Dict] = None,
        confidence: float = 1.0
    ) -> str:
        """Сохранение записи"""
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'confidence': confidence,
            'stored_at': datetime.now().isoformat()
        })
        
        record = {
            'key': key,
            'content': content,
            'metadata': metadata,
            'access_count': 0,
            'last_access': datetime.now().timestamp(),
            'created_at': datetime.now().isoformat()
        }
        
        self.cache[key] = record
        self.stats['total_records'] = len(self.cache)
        
        # Автосохранение
        await self.save()
        
        return key
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.0
    ) -> List[Dict]:
        """Поиск по ключу (простое совпадение)"""
        self.stats['queries'] += 1
        results = []
        
        # Прямое совпадение ключа
        if query in self.cache:
            record = self.cache[query]
            confidence = record['metadata'].get('confidence', 1.0)
            if confidence >= min_confidence:
                record['access_count'] += 1
                record['last_access'] = datetime.now().timestamp()
                results.append({
                    'key': query,
                    'content': record['content'],
                    'metadata': record['metadata'],
                    'confidence': confidence
                })
        
        # Поиск по частичному вхождению (если прямое совпадение не найдено)
        if not results:
            for key, record in self.cache.items():
                if query.lower() in key.lower():
                    confidence = record['metadata'].get('confidence', 1.0)
                    if confidence >= min_confidence:
                        record['access_count'] += 1
                        record['last_access'] = datetime.now().timestamp()
                        results.append({
                            'key': key,
                            'content': record['content'],
                            'metadata': record['metadata'],
                            'confidence': confidence * 0.9
                        })
                        if len(results) >= top_k:
                            break
        
        if results:
            self.stats['hits'] += 1
        else:
            self.stats['misses'] += 1
        
        return results[:top_k]
    
    async def delete(self, key: str) -> bool:
        """Удаление записи"""
        if key not in self.cache:
            return False
        del self.cache[key]
        self.stats['total_records'] = len(self.cache)
        await self.save()
        return True
    
    async def clear(self):
        """Очистка всей памяти"""
        self.cache.clear()
        await self.save()
        logger.info("🧹 Engram память полностью очищена")
    
    async def get_stats(self) -> Dict:
        """Статистика"""
        return {
            'total_records': self.stats['total_records'],
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses']),
            'queries': self.stats['queries'],
            'errors': self.stats['errors'],
            'cache_size': len(self.cache),
            'max_records': self.max_records
        }

    async def get_all_keys(self) -> List[str]:
        """Возвращает список всех ключей (тем) в Engram."""
        return list(self.cache.keys())
    
    async def close(self):
        """Завершение работы"""
        await self.save()
        logger.info("✅ EngramService завершил работу")
    
    async def health_check(self) -> Dict:
        """Проверка здоровья"""
        return {
            'healthy': True,
            'records': len(self.cache),
            'message': 'Engram is operational',
            'timestamp': datetime.now().isoformat()
        }