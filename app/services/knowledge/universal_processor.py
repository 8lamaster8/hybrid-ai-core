"""
Универсальный процессор для любых форматов
"""
import re
import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
from io import StringIO

from app.core.logging import logger
from app.services.knowledge.base import KnowledgeChunk


class UniversalDatasetProcessor:
    """Универсальный процессор для любых форматов датасетов"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = 10  # Минимальная длина чанка
    
    def detect_format(self, content: str) -> str:
        """Определение формата контента"""
        try:
            content_stripped = content.strip()
            
            if not content_stripped:
                return "empty"
            
            # Удаляем BOM если есть
            if content_stripped.startswith('\ufeff'):
                content_stripped = content_stripped[1:]
            
            # 1. JSON массив
            if content_stripped.startswith('[') and content_stripped.endswith(']'):
                try:
                    data = json.loads(content_stripped)
                    if isinstance(data, list):
                        logger.info("📋 Обнаружен JSON массив")
                        return "json_array"
                except:
                    pass
            
            # 2. JSON объект
            if content_stripped.startswith('{') and content_stripped.endswith('}'):
                try:
                    json.loads(content_stripped)
                    logger.info("📋 Обнаружен JSON объект")
                    return "json_object"
                except:
                    pass
            
            # 3. JSONL
            lines = content_stripped.split('\n')[:20]
            valid_json_lines = 0
            for line in lines:
                line = line.strip()
                if line and line.startswith('{') and line.endswith('}'):
                    try:
                        json.loads(line)
                        valid_json_lines += 1
                    except:
                        continue
            
            if valid_json_lines >= 2:
                logger.info(f"📋 Обнаружен JSONL ({valid_json_lines} валидных строк)")
                return "jsonl"
            
            # 4. CSV
            lines = content_stripped.split('\n')[:10]
            if len(lines) >= 2:
                # Проверяем на CSV
                try:
                    dialect = csv.Sniffer().sniff(lines[0])
                    if dialect:
                        logger.info("📋 Обнаружен CSV")
                        return "csv"
                except:
                    # Проверяем на наличие запятых или точек с запятой
                    first_line = lines[0]
                    if ',' in first_line or ';' in first_line:
                        # Проверяем, что есть хотя бы одно буквенное значение
                        if any(c.isalpha() for c in first_line):
                            logger.info("📋 Обнаружен CSV (по паттерну)")
                            return "csv"
            
            # 5. QA текстовый формат
            qa_patterns = [
                r'Вопрос[:\s]+.*?Ответ[:\s]+',
                r'Question[:\s]+.*?Answer[:\s]+',
                r'Q:[^A]*A:',
                r'вопрос[:\s].*?ответ[:\s]'
            ]
            
            for pattern in qa_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                    logger.info("📋 Обнаружен текстовый QA формат")
                    return "qa_text"
            
            # 6. Простой текст
            logger.info("📋 Обнаружен простой текст")
            return "text"
            
        except Exception as e:
            logger.error(f"❌ Ошибка определения формата: {e}")
            return "text"
    
    def process_content(self, content: str, file_name: str, metadata: Optional[Dict] = None) -> List[KnowledgeChunk]:
        """Обработка контента любого формата"""
        try:
            format_type = self.detect_format(content)
            logger.info(f"🔄 Обрабатываем файл '{file_name}' как {format_type}")
            
            if not content.strip():
                logger.warning("⚠️ Файл пустой")
                return []
            
            # Обработка в зависимости от формата
            if format_type == "json_array":
                chunks = self._process_json_array(content, file_name, metadata)
            elif format_type == "json_object":
                chunks = self._process_json_object(content, file_name, metadata)
            elif format_type == "jsonl":
                chunks = self._process_jsonl(content, file_name, metadata)
            elif format_type == "csv":
                chunks = self._process_csv(content, file_name, metadata)
            elif format_type == "qa_text":
                chunks = self._process_qa_text(content, file_name, metadata)
            else:
                chunks = self._process_text(content, file_name, metadata)
            
            logger.info(f"✅ Файл '{file_name}' обработан: {len(chunks)} чанков")
            
            if not chunks:
                logger.warning(f"⚠️ Не удалось создать чанки из файла '{file_name}'")
                # Пробуем создать хотя бы один чанк из всего содержимого
                if len(content.strip()) > self.min_chunk_length:
                    chunk = self._create_text_chunk(
                        text=content.strip(),
                        index=0,
                        file_name=file_name,
                        metadata=metadata
                    )
                    chunks = [chunk]
                    logger.info(f"✅ Создан один чанк из всего содержимого")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка обработки файла {file_name}: {e}", exc_info=True)
            # Пробуем создать хотя бы один чанк
            try:
                if content and len(content.strip()) > self.min_chunk_length:
                    chunk = self._create_text_chunk(
                        text=content.strip()[:5000],
                        index=0,
                        file_name=file_name,
                        metadata={**(metadata or {}), "error": str(e)[:100]}
                    )
                    logger.info(f"✅ Создан аварийный чанк")
                    return [chunk]
            except:
                pass
            return []
    
    def _process_json_array(self, content: str, file_name: str, metadata: Optional[Dict]) -> List[KnowledgeChunk]:
        """Обработка JSON массива"""
        try:
            data = json.loads(content)
            chunks = []
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, (str, int, float, bool)):
                        chunk = self._create_text_chunk(
                            text=str(item),
                            index=i,
                            file_name=file_name,
                            metadata={**(metadata or {}), "json_type": "primitive"}
                        )
                        chunks.append(chunk)
                    elif isinstance(item, dict):
                        # Преобразуем dict в текст
                        text = json.dumps(item, ensure_ascii=False, indent=2)
                        chunk = self._create_text_chunk(
                            text=text,
                            index=i,
                            file_name=file_name,
                            metadata={**(metadata or {}), "json_type": "object"}
                        )
                        chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"❌ Ошибка обработки JSON массива: {e}")
            return []
    
    def _process_json_object(self, content: str, file_name: str, metadata: Optional[Dict]) -> List[KnowledgeChunk]:
        """Обработка JSON объекта"""
        try:
            data = json.loads(content)
            text = json.dumps(data, ensure_ascii=False, indent=2)
            
            chunk = self._create_text_chunk(
                text=text,
                index=0,
                file_name=file_name,
                metadata={**(metadata or {}), "json_type": "single_object"}
            )
            return [chunk]
        except Exception as e:
            logger.error(f"❌ Ошибка обработки JSON объекта: {e}")
            return []
    
    def _process_jsonl(self, content: str, file_name: str, metadata: Optional[Dict]) -> List[KnowledgeChunk]:
        """Обработка JSON Lines"""
        chunks = []
        lines = content.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    text = json.dumps(data, ensure_ascii=False, indent=2)
                    chunk = self._create_text_chunk(
                        text=text,
                        index=i,
                        file_name=file_name,
                        metadata={**(metadata or {}), "json_type": "jsonl"}
                    )
                    chunks.append(chunk)
                elif isinstance(data, (str, int, float, bool)):
                    chunk = self._create_text_chunk(
                        text=str(data),
                        index=i,
                        file_name=file_name,
                        metadata={**(metadata or {}), "json_type": "jsonl_primitive"}
                    )
                    chunks.append(chunk)
            except:
                # Если не JSON, обрабатываем как текст
                if len(line) > self.min_chunk_length:
                    chunk = self._create_text_chunk(
                        text=line,
                        index=i,
                        file_name=file_name,
                        metadata={**(metadata or {}), "json_type": "text_line"}
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _process_csv(self, content: str, file_name: str, metadata: Optional[Dict]) -> List[KnowledgeChunk]:
        """Обработка CSV"""
        try:
            chunks = []
            
            # Пробуем разные разделители
            for delimiter in [',', ';', '\t']:
                try:
                    reader = csv.reader(StringIO(content), delimiter=delimiter)
                    rows = list(reader)
                    
                    if len(rows) > 1:  # Есть хотя бы заголовок и одна строка
                        for i, row in enumerate(rows):
                            if row:  # Пропускаем пустые строки
                                text = f"Строка {i+1}: {', '.join(row)}"
                                chunk = self._create_text_chunk(
                                    text=text,
                                    index=i,
                                    file_name=file_name,
                                    metadata={**(metadata or {}), "csv_row": i, "delimiter": delimiter}
                                )
                                chunks.append(chunk)
                        break
                except:
                    continue
            
            return chunks
        except Exception as e:
            logger.error(f"❌ Ошибка обработки CSV: {e}")
            return []
    
    def _process_qa_text(self, content: str, file_name: str, metadata: Optional[Dict]) -> List[KnowledgeChunk]:
        """Обработка текстового QA формата"""
        chunks = []
        
        # Паттерны для поиска QA пар
        patterns = [
            r'(?:Вопрос|Question|Q)[:\s]*(.*?)(?:Ответ|Answer|A)[:\s]*(.*?)(?=(?:Вопрос|Question|Q)[:\s]|$)',
            r'(?:В|Q)[:\s\.]*(.*?)(?:О|A)[:\s\.]*(.*?)(?=(?:В|Q)[:\s\.]|$)',
            r'([^:\n]+)[:\s]*(.*?)\n([^:\n]+)[:\s]*(.*?)(?=\n[^:\n]+[:\s]|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if matches:
                for i, match in enumerate(matches):
                    if len(match) >= 2:
                        question = match[0].strip()
                        answer = match[1].strip()
                        
                        if question and answer and len(question) > 3 and len(answer) > 3:
                            chunk = self._create_qa_chunk(
                                question=question,
                                answer=answer,
                                index=i,
                                file_name=file_name,
                                metadata=metadata
                            )
                            chunks.append(chunk)
                
                if chunks:
                    logger.info(f"📝 Найдено {len(chunks)} QA пар")
                    break
        
        return chunks
    
    def _process_text(self, content: str, file_name: str, metadata: Optional[Dict]) -> List[KnowledgeChunk]:
        """Обработка обычного текста"""
        chunks = []
        
        # 1. Пробуем разбить на абзацы по двойным переносам строк
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # 2. Если не нашли, пробуем по одинарным переносам
        if not paragraphs:
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        # 3. Если все еще нет, пробуем разбить по точкам
        if not paragraphs:
            sentences = re.split(r'[.!?]+', content)
            paragraphs = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # 4. Если очень длинный текст без разделителей, разбиваем на чанки фиксированного размера
        if not paragraphs and len(content) > self.chunk_size:
            for i in range(0, len(content), self.chunk_size - self.overlap):
                chunk_text = content[i:i + self.chunk_size]
                if len(chunk_text.strip()) > self.min_chunk_length:
                    chunk = self._create_text_chunk(
                        text=chunk_text,
                        index=i,
                        file_name=file_name,
                        metadata={**(metadata or {}), "chunk_type": "fixed_size"}
                    )
                    chunks.append(chunk)
            return chunks
        
        # Обработка найденных параграфов
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > self.min_chunk_length:
                # Если параграф слишком длинный, разбиваем его
                if len(paragraph) > self.chunk_size:
                    for j in range(0, len(paragraph), self.chunk_size - self.overlap):
                        chunk_text = paragraph[j:j + self.chunk_size]
                        if len(chunk_text.strip()) > self.min_chunk_length:
                            chunk = self._create_text_chunk(
                                text=chunk_text,
                                index=f"{i}_{j}",
                                file_name=file_name,
                                metadata={**(metadata or {}), "chunk_type": "paragraph_split"}
                            )
                            chunks.append(chunk)
                else:
                    chunk = self._create_text_chunk(
                        text=paragraph,
                        index=i,
                        file_name=file_name,
                        metadata={**(metadata or {}), "chunk_type": "paragraph"}
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _create_qa_chunk(self, question: str, answer: str, index: int, file_name: str, metadata: Optional[Dict]) -> KnowledgeChunk:
        """Создание QA чанка"""
        content = f"Вопрос: {question}\n\nОтвет: {answer}"
        
        return KnowledgeChunk(
            id=f"{Path(file_name).stem}_qa_{index}_{hash(content[:50])}",
            content=content[:5000],  # Ограничиваем длину
            metadata={
                "source": file_name,
                "file_name": file_name,
                "type": "qa_pair",
                "question": question[:500],
                "answer": answer[:2000],
                "index": index,
                "content_type": "qa",
                **(metadata or {})
            }
        )
    
    def _create_text_chunk(self, text: str, index: int, file_name: str, metadata: Optional[Dict]) -> KnowledgeChunk:
        """Создание текстового чанка"""
        # Очищаем текст от лишних пробелов
        text = ' '.join(text.split())
        
        return KnowledgeChunk(
            id=f"{Path(file_name).stem}_txt_{index}_{hash(text[:50])}",
            content=text[:5000],  # Ограничиваем длину
            metadata={
                "source": file_name,
                "file_name": file_name,
                "type": "text",
                "index": index,
                "content_type": "text",
                "text_length": len(text),
                **(metadata or {})
            }
        )