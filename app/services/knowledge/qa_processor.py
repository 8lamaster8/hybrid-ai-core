"""
Простой QA процессор
"""
import json
import re
from typing import List, Dict, Optional
from pathlib import Path

from app.core.logging import logger
from app.services.knowledge.base import KnowledgeChunk


class QADatasetProcessor:
    """Процессор для простых QA датасетов"""
    
    def __init__(self):
        self.min_question_length = 5
        self.min_answer_length = 5
    
    def process_content(self, content: str, file_name: str, metadata: Optional[Dict] = None) -> List[KnowledgeChunk]:
        """Обработка содержимого файла"""
        try:
            content_stripped = content.strip()
            logger.info(f"🔄 QADatasetProcessor обрабатывает '{file_name}'")
            
            # Пытаемся определить формат
            if self._is_json(content_stripped):
                return self._process_json(content_stripped, file_name, metadata)
            elif self._is_jsonl(content_stripped):
                return self._process_jsonl(content_stripped, file_name, metadata)
            else:
                # Пробуем найти структурированные данные в тексте
                chunks = self._extract_qa_from_text(content_stripped, file_name, metadata)
                if chunks:
                    return chunks
                
                # Если не нашли QA, пробуем просто разбить текст
                return self._split_into_chunks(content_stripped, file_name, metadata)
                
        except Exception as e:
            logger.error(f"❌ Ошибка обработки файла {file_name}: {e}", exc_info=True)
            return []
    
    def _is_json(self, content: str) -> bool:
        """Проверка, является ли содержание JSON"""
        content = content.strip()
        return (content.startswith('[') and content.endswith(']')) or \
               (content.startswith('{') and content.endswith('}'))
    
    def _is_jsonl(self, content: str) -> bool:
        """Проверка на JSON Lines формат"""
        lines = content.strip().split('\n')[:5]
        if len(lines) < 2:
            return False
        
        json_lines = 0
        for line in lines:
            line = line.strip()
            if line and line.startswith('{') and line.endswith('}'):
                try:
                    json.loads(line)
                    json_lines += 1
                except:
                    pass
        
        return json_lines >= 2
    
    def _process_json(self, content: str, file_name: str, metadata: Optional[Dict]) -> List[KnowledgeChunk]:
        """Обработка JSON содержимого"""
        try:
            data = json.loads(content)
            chunks = []
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    chunks.extend(self._process_qa_item(item, i, file_name, metadata))
            elif isinstance(data, dict):
                chunks.extend(self._process_qa_item(data, 0, file_name, metadata))
            
            logger.info(f"✅ Обработан JSON файл {file_name}: {len(chunks)} QA пар")
            return chunks
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Не удалось распарсить JSON: {e}")
            return []
    
    def _process_jsonl(self, content: str, file_name: str, metadata: Optional[Dict]) -> List[KnowledgeChunk]:
        """Обработка JSON Lines содержимого"""
        chunks = []
        lines = content.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                chunks.extend(self._process_qa_item(item, i, file_name, metadata))
            except:
                # Пропускаем невалидные строки
                pass
        
        logger.info(f"✅ Обработан JSONL файл {file_name}: {len(chunks)} QA пар")
        return chunks
    
    def _extract_qa_from_text(self, content: str, file_name: str, metadata: Optional[Dict]) -> List[KnowledgeChunk]:
        """Извлечение QA пар из текста"""
        chunks = []
        
        # Паттерны для поиска QA пар
        patterns = [
            r'(?:Вопрос|Question|Q)[:\s]*(.*?)[\s\n]*(?:Ответ|Answer|A)[:\s]*(.*?)(?=\n\n|\n(?:Вопрос|Question|Q)[:\s]|$)',
            r'(?:Q:|Question:|Вопрос:|В:)[\s\n]*(.*?)[\s\n]*(?:A:|Answer:|Ответ:|О:)[\s\n]*(.*?)(?=\n\n|\n(?:Q:|Question:|Вопрос:|В:)|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if matches:
                for i, match in enumerate(matches):
                    if len(match) >= 2:
                        question = match[0].strip()
                        answer = match[1].strip()
                        
                        if (len(question) >= self.min_question_length and 
                            len(answer) >= self.min_answer_length):
                            chunks.append(self._create_qa_chunk(
                                question=question,
                                answer=answer,
                                index=i,
                                file_name=file_name,
                                metadata=metadata
                            ))
                
                if chunks:
                    logger.info(f"📝 Найдено {len(chunks)} QA пар в тексте")
                    break
        
        return chunks
    
    def _split_into_chunks(self, content: str, file_name: str, metadata: Optional[Dict]) -> List[KnowledgeChunk]:
        """Разбивка на чанки"""
        chunks = []
        
        # Разбиваем на абзацы
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 10:
                chunks.append(self._create_text_chunk(
                    text=paragraph,
                    index=i,
                    file_name=file_name,
                    metadata={**(metadata or {}), "type": "paragraph"}
                ))
        
        logger.info(f"📄 Разбит на {len(chunks)} абзацев: {file_name}")
        return chunks
    
    def _process_qa_item(self, item: any, index: int, file_name: str, metadata: Optional[Dict]) -> List[KnowledgeChunk]:
        """Обработка одного элемента QA данных"""
        chunks = []
        
        if isinstance(item, dict):
            # Пробуем разные ключи для вопроса и ответа
            question_keys = ['question', 'input', 'q', 'вопрос', 'query', 'prompt']
            answer_keys = ['answer', 'output', 'a', 'ответ', 'response', 'completion']
            
            question = None
            answer = None
            
            for q_key in question_keys:
                if q_key in item:
                    question = str(item[q_key])
                    break
            
            for a_key in answer_keys:
                if a_key in item:
                    answer = str(item[a_key])
                    break
            
            if question and answer:
                if (len(question) >= self.min_question_length and 
                    len(answer) >= self.min_answer_length):
                    chunks.append(self._create_qa_chunk(
                        question=question,
                        answer=answer,
                        index=index,
                        file_name=file_name,
                        metadata=metadata
                    ))
            elif question:
                # Только вопрос
                if len(question) >= self.min_question_length:
                    chunks.append(self._create_text_chunk(
                        text=question,
                        index=index,
                        file_name=file_name,
                        metadata={**(metadata or {}), "type": "question_only"}
                    ))
            elif answer:
                # Только ответ
                if len(answer) >= self.min_answer_length:
                    chunks.append(self._create_text_chunk(
                        text=answer,
                        index=index,
                        file_name=file_name,
                        metadata={**(metadata or {}), "type": "answer_only"}
                    ))
            else:
                # Если не нашли структурированных данных, сохраняем весь объект как текст
                text = json.dumps(item, ensure_ascii=False)[:1000]
                chunks.append(self._create_text_chunk(
                    text=text,
                    index=index,
                    file_name=file_name,
                    metadata={**(metadata or {}), "type": "json_object"}
                ))
        
        elif isinstance(item, str):
            # Просто строка
            if len(item.strip()) > 10:
                chunks.append(self._create_text_chunk(
                    text=item,
                    index=index,
                    file_name=file_name,
                    metadata=metadata
                ))
        
        return chunks
    
    def _create_qa_chunk(self, question: str, answer: str, index: int, file_name: str, metadata: Optional[Dict]) -> KnowledgeChunk:
        """Создание чанка для QA пары"""
        content = f"Вопрос: {question}\n\nОтвет: {answer}"
        
        return KnowledgeChunk(
            id=f"{Path(file_name).stem}_qa_{index}_{hash(question[:50])}",
            content=content,
            metadata={
                "source": file_name,
                "file_name": file_name,
                "type": "qa_pair",
                "question": question[:200],
                "answer": answer[:500],
                "index": index,
                **(metadata or {})
            }
        )
    
    def _create_text_chunk(self, text: str, index: int, file_name: str, metadata: Optional[Dict]) -> KnowledgeChunk:
        """Создание обычного текстового чанка"""
        return KnowledgeChunk(
            id=f"{Path(file_name).stem}_text_{index}_{hash(text[:50])}",
            content=text[:3000],
            metadata={
                "source": file_name,
                "file_name": file_name,
                "type": metadata.get("type", "text") if metadata else "text",
                "index": index,
                **(metadata or {})
            }
        )