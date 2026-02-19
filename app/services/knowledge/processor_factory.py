"""
Фабрика процессоров с автоматическим выбором формата
"""
import re
import json
from typing import List, Dict, Optional, Any
from pathlib import Path

from app.core.logging import logger
from app.services.knowledge.universal_processor import UniversalDatasetProcessor
from app.services.knowledge.qa_processor import QADatasetProcessor
from app.services.knowledge.base import KnowledgeChunk


class ProcessorFactory:
    """Фабрика для автоматического выбора оптимального процессора"""
    
    @staticmethod
    def get_processor(content: str, file_name: str, metadata: Optional[Dict] = None):
        """
        Автоматически выбирает лучший процессор для контента
        """
        try:
            content_sample = content[:5000]  # Берем первые 5KB для анализа
            
            logger.info(f"🔍 Анализируем файл '{file_name}'")
            logger.info(f"📊 Размер файла: {len(content)} символов")
            logger.info(f"📝 Первые 500 символов: {content_sample[:500]}...")
            
            # Правила выбора:
            
            # 1. Для простых QA датасетов (только вопросы-ответы)
            if ProcessorFactory._is_simple_qa(content_sample):
                logger.info(f"✅ Для файла '{file_name}' выбран QADatasetProcessor (простой QA)")
                return QADatasetProcessor()
            
            # 2. Для сложных/смешанных форматов - Universal
            elif ProcessorFactory._is_complex_format(content_sample, file_name):
                logger.info(f"✅ Для файла '{file_name}' выбран UniversalDatasetProcessor (сложный формат)")
                return UniversalDatasetProcessor()
            
            # 3. По умолчанию - Universal (самый мощный)
            else:
                logger.info(f"✅ Для файла '{file_name}' выбран UniversalDatasetProcessor (по умолчанию)")
                return UniversalDatasetProcessor()
                
        except Exception as e:
            logger.error(f"❌ Ошибка в фабрике процессоров: {e}")
            # Всегда возвращаем Universal как запасной вариант
            return UniversalDatasetProcessor()
    
    @staticmethod
    def _is_simple_qa(content: str) -> bool:
        """Определяет, является ли контент простым QA датасетом"""
        try:
            # Проверяем на простые JSON массивы с QA
            content_stripped = content.strip()
            
            # Если это JSON массив
            if content_stripped.startswith('[') and content_stripped.endswith(']'):
                try:
                    data = json.loads(content_stripped)
                    if isinstance(data, list) and len(data) > 0:
                        first_item = data[0]
                        if isinstance(first_item, dict):
                            has_question = any(key in first_item for key in ['question', 'input', 'вопрос', 'q'])
                            has_answer = any(key in first_item for key in ['answer', 'output', 'ответ', 'a'])
                            if has_question and has_answer:
                                logger.info("📋 Обнаружен JSON с QA парами")
                                return True
                except json.JSONDecodeError:
                    pass
            
            # Если это JSONL с QA парами
            lines = content_stripped.split('\n')
            qa_lines = 0
            for line in lines[:10]:  # Проверяем первые 10 строк
                line = line.strip()
                if line and line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            has_question = any(key in data for key in ['question', 'input', 'вопрос', 'q'])
                            has_answer = any(key in data for key in ['answer', 'output', 'ответ', 'a'])
                            if has_question and has_answer:
                                qa_lines += 1
                    except:
                        pass
            
            if qa_lines >= 3:  # Если хотя бы 3 строки выглядят как QA
                logger.info(f"📋 Обнаружен JSONL с {qa_lines} QA парами")
                return True
            
            # Если это простой текст с вопросами-ответами
            qa_patterns = [
                r'Вопрос[:\s]+.*?Ответ[:\s]+',
                r'Question[:\s]+.*?Answer[:\s]+',
                r'Q:[^A]*A:',
                r'Q\.[^A]*A\.'
            ]
            
            for pattern in qa_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                if matches:
                    logger.info(f"📋 Обнаружено {len(matches)} QA пар в тексте")
                    return True
            
            # Проверяем наличие маркеров вопросов и ответов
            question_indicators = ['вопрос:', 'question:', 'q:', 'в:', '?']
            answer_indicators = ['ответ:', 'answer:', 'a:', 'о:', '!', '.']
            
            has_questions = any(indicator in content.lower() for indicator in question_indicators)
            has_answers = any(indicator in content.lower() for indicator in answer_indicators)
            
            if has_questions and has_answers:
                logger.info("📋 Обнаружены маркеры вопросов и ответов")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка при проверке формата QA: {e}")
            return False
    
    @staticmethod
    def _is_complex_format(content: str, file_name: str) -> bool:
        """Определяет сложные форматы, требующие UniversalProcessor"""
        try:
            # 1. CSV файлы
            if file_name.lower().endswith('.csv'):
                logger.info("📊 Обнаружен CSV файл")
                return True
            
            # 2. JSON с вложенными структурами
            if content.strip().startswith('['):
                try:
                    data = json.loads(content)
                    if isinstance(data, list) and len(data) > 0:
                        first_item = data[0]
                        if isinstance(first_item, dict):
                            # Проверяем на сложные поля
                            complex_fields = ['variants', 'options', 'metadata', 'context', 'examples']
                            if any(field in first_item for field in complex_fields):
                                logger.info("📊 Обнаружен JSON со сложной структурой")
                                return True
                except:
                    pass
            
            # 3. Текст с JSON внутри
            json_matches = re.findall(r'\{\s*".*?"\s*:\s*".*?"\s*\}', content)
            if len(json_matches) >= 3:
                logger.info(f"📊 Обнаружено {len(json_matches)} JSON объектов в тексте")
                return True
            
            # 4. Смешанные форматы
            complex_keywords = ['variant', 'category', 'confidence', 'source', 'metadata', 'context', 'example']
            if any(keyword in content.lower() for keyword in complex_keywords):
                logger.info("📊 Обнаружены ключевые слова сложного формата")
                return True
            
            # 5. Разметка таблицы
            if '|' in content and '-' in content and '\n' in content:
                lines_with_pipe = [line for line in content.split('\n') if '|' in line]
                if len(lines_with_pipe) >= 3:
                    logger.info("📊 Обнаружена табличная разметка")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка при проверке сложного формата: {e}")
            return False