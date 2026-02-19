"""
🧹 Текстовый процессор - очистка, нормализация, извлечение контента
"""

import re
import html
import unicodedata
from typing import List, Optional


class TextCleaner:
    """
    Очистка и нормализация текста.
    - Удаление HTML-тегов
    - Нормализация пробелов
    - Удаление мусорных символов
    - Приведение к читаемому виду
    """
    
    def __init__(self):
        # Регулярные выражения для удаления мусора
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.extra_spaces_pattern = re.compile(r'\s+')
        self.non_printable_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        self.reference_pattern = re.compile(r'\[\d+\]|\[\w+\]|\[[^]]+\]')
        self.control_chars = re.compile(r'[\r\n\t]+')
        
        # Стоп-фразы, которые часто являются мусором
        self.junk_phrases = [
            'архивная копия', 'wayback machine', '↑', 'комм.', 
            'источник:', 'ссылка:', 'примечание', 'сноска',
            'редактировать', 'править', 'страница обсуждения',
            'категория:', 'википедия', 'wikipedia', 'facebook',
            'twitter', 'instagram', 'tiktok', 'youtube', 'pinterest',
            'купить', 'реклама', 'спам', 'подписаться', 'поделиться'
        ]
    
    def clean(self, text: str, remove_junk: bool = True, normalize_whitespace: bool = True) -> str:
        """
        Основной метод очистки текста.
        
        Args:
            text: Исходный текст
            remove_junk: Удалять мусорные фразы
            normalize_whitespace: Нормализовать пробелы
            
        Returns:
            Очищенный текст
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Декодирование HTML-сущностей
        text = html.unescape(text)
        
        # 2. Удаление HTML-тегов
        text = self.html_tag_pattern.sub(' ', text)
        
        # 3. Удаление URL
        text = self.url_pattern.sub(' ', text)
        
        # 4. Удаление ссылочных маркеров [1], [a], и т.д.
        text = self.reference_pattern.sub(' ', text)
        
        # 5. Удаление непечатных символов
        text = self.non_printable_pattern.sub('', text)
        
        # 6. Замена управляющих символов пробелами
        text = self.control_chars.sub(' ', text)
        
        # 7. Нормализация пробелов
        if normalize_whitespace:
            text = self.extra_spaces_pattern.sub(' ', text)
        
        # 8. Удаление мусорных фраз
        if remove_junk:
            text = self._remove_junk_phrases(text)
        
        # 9. Обрезка лишних пробелов по краям
        text = text.strip()
        
        return text
    
    def _remove_junk_phrases(self, text: str) -> str:
        """Удаление мусорных фраз"""
        lower_text = text.lower()
        for phrase in self.junk_phrases:
            if phrase in lower_text:
                # Удаляем фразу и окружающие пробелы
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                text = pattern.sub(' ', text)
        
        # Повторная нормализация пробелов после удаления
        text = self.extra_spaces_pattern.sub(' ', text)
        return text
    
    def extract_sentences(self, text: str, min_length: int = 20) -> List[str]:
        """
        Разбиение текста на предложения.
        
        Args:
            text: Исходный текст
            min_length: Минимальная длина предложения (символов)
            
        Returns:
            Список предложений
        """
        # Простое разбиение по знакам препинания
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Фильтрация по длине и удаление пустых
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [s for s in sentences if len(s) >= min_length]
        
        return sentences
    
    def normalize_unicode(self, text: str) -> str:
        """Нормализация Unicode (NFKC)"""
        return unicodedata.normalize('NFKC', text)
    
    def remove_repetitions(self, text: str, threshold: int = 3) -> str:
        """
        Удаление повторяющихся фраз (простейшая эвристика).
        """
        sentences = self.extract_sentences(text, min_length=10)
        unique_sentences = []
        seen = set()
        
        for sent in sentences:
            # Берем первые 50 символов как сигнатуру
            sig = sent[:50].lower()
            if sig not in seen:
                seen.add(sig)
                unique_sentences.append(sent)
        
        return ' '.join(unique_sentences)


class ContentExtractor:
    """
    Извлечение основного контента из HTML.
    (Упрощенная версия; в продакшене лучше использовать 
    readability-lxml или аналоги)
    """
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
    
    def extract_from_html(self, html_content: str) -> str:
        """
        Простейшее извлечение текста из HTML.
        Реальная реализация должна использовать BeautifulSoup или lxml.
        """
        # Это заглушка; в реальном коде используется BeautifulSoup/lxml
        # Оставляем здесь для совместимости; настоящая логика в detective.py
        return html_content