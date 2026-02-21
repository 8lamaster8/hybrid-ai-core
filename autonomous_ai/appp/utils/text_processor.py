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


import re
from typing import Optional

try:
    from bs4 import BeautifulSoup, Comment
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    # Можно использовать lxml, если оно есть, но для простоты оставим заглушку
    BeautifulSoup = None

from appp.utils.text_processor import TextCleaner


class ContentExtractor:
    """
    Извлечение основного контента из HTML.
    Использует BeautifulSoup для парсинга и эвристики для выделения текста.
    """

    # Селекторы для удаления мусорных блоков
    REMOVE_SELECTORS = [
        'script', 'style', 'noscript', 'meta', 'link',
        'nav', 'header', 'footer', 'aside',
        '.sidebar', '#sidebar', '.comments', '#comments',
        '.advertisement', '.ads', '.banner',
        '.cookie-notice', '.popup', '.modal',
        'form', 'input', 'button',
        '.social-share', '.share-buttons',
        '.related-posts', '.recommendations'
    ]

    # Селекторы для приоритетного контента
    CONTENT_SELECTORS = [
        'article',
        'main',
        '[role="main"]',
        '.post-content',
        '.entry-content',
        '.article-content',
        '.content-body',
        '#content',
        '.content'
    ]

    def __init__(self, use_readability_if_available: bool = False):
        self.text_cleaner = TextCleaner()
        self.use_readability = use_readability_if_available

        # Попытка импортировать readability-lxml, если запрошено
        self.readability = None
        if self.use_readability:
            try:
                from readability import Document
                self.readability = Document
            except ImportError:
                pass

    def extract_from_html(self, html_content: str, url: Optional[str] = None) -> str:
        """
        Извлекает основной текст из HTML.

        Args:
            html_content: Исходный HTML
            url: URL страницы (может использоваться для улучшения извлечения)

        Returns:
            Очищенный текст
        """
        if not html_content:
            return ""

        # 1. Если включён readability и он доступен — используем его
        if self.readability:
            try:
                doc = self.readability(html_content, url=url)
                return self.text_cleaner.clean(doc.summary())
            except Exception as e:
                # В случае ошибки падаем на BeautifulSoup
                pass

        # 2. Используем BeautifulSoup
        if not BS4_AVAILABLE:
            # Крайний случай: удаляем теги простым regexp (очень грязно)
            text = re.sub(r'<[^>]+>', ' ', html_content)
            return self.text_cleaner.clean(text)

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Удаляем ненужные элементы
            self._remove_unwanted(soup)

            # Пытаемся найти основной контент
            content_container = self._find_content_container(soup)

            if content_container:
                # Извлекаем текст из контейнера
                raw_text = content_container.get_text(separator='\n', strip=True)
            else:
                # Если не нашли — берём весь body (без удалённых элементов)
                body = soup.find('body')
                if body:
                    raw_text = body.get_text(separator='\n', strip=True)
                else:
                    # Если и body нет — весь суп
                    raw_text = soup.get_text(separator='\n', strip=True)

            # 3. Очистка через TextCleaner
            cleaned = self.text_cleaner.clean(raw_text)
            return cleaned

        except Exception as e:
            # В случае любой ошибки парсинга — падаем на простую очистку
            text = re.sub(r'<[^>]+>', ' ', html_content)
            return self.text_cleaner.clean(text)

    def _remove_unwanted(self, soup: BeautifulSoup) -> None:
        """Удаляет из супа мусорные элементы по селекторам."""
        # Удаляем комментарии
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Удаляем элементы по селекторам
        for selector in self.REMOVE_SELECTORS:
            for element in soup.select(selector):
                element.decompose()

        # Дополнительно: удаляем пустые элементы (опционально)
        # for element in soup.find_all():
        #     if not element.get_text(strip=True) and element.name not in ['br', 'hr', 'img']:
        #         element.decompose()

    def _find_content_container(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Ищет контейнер с основным контентом по приоритетным селекторам."""
        for selector in self.CONTENT_SELECTORS:
            container = soup.select_one(selector)
            if container:
                return container

        # Если не нашли по селекторам, попробуем эвристику: тег с наибольшим количеством текста
        candidates = []
        for tag in soup.find_all(['div', 'section', 'article']):
            text_len = len(tag.get_text(strip=True))
            if text_len > 200:  # Минимальный порог
                candidates.append((text_len, tag))

        if candidates:
            # Возвращаем тег с самым длинным текстом
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None