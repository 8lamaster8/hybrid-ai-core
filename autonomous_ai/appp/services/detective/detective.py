"""
🚀 Детектив — загрузка страниц, извлечение контента
Улучшенная версия с использованием trafilatura и RobustTextCleaner
"""

import asyncio
import aiohttp
import aiofiles
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse, quote_plus

from bs4 import BeautifulSoup

# Попытка импорта продвинутых парсеров
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    print("⚠️ trafilatura не установлена, используем базовый парсинг")

try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False

from appp.core.logging import logger
from appp.utils.text_processor import TextCleaner
from appp.services.real_search import hybrid_searcher


class Detective:
    def __init__(self, config: Dict):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_dir = "./data/cache/detective"
        self.cache = {}
        self.cache_ttl = 3600
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119.0.0.0 Safari/537.36',
        ]
        self.blacklist = set(config.get('blacklist_domains', []))
        self.priority_domains = set(config.get('priority_domains', []))
        self.text_cleaner = TextCleaner()
        #self.robust_cleaner = RobustTextCleaner()
        self.stats = {
            'searches': 0,
            'pages_processed': 0,
            'requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'avg_response_time': 0,
            'unique_domains': set(),
            'total_content_length': 0
        }
        logger.info("🚀 Detective создан")

    async def initialize(self):
        logger.info("🔄 Инициализация Detective...")
        os.makedirs(self.cache_dir, exist_ok=True)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=100, ssl=True)#ssl=False
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        await self._load_cache()
        # Инициализация глобального поисковика
        if not getattr(hybrid_searcher, '_initialized', False):
            await hybrid_searcher.initialize()
            hybrid_searcher._initialized = True
        logger.info("✅ Detective инициализирован")
        return True

    async def fetch_page_content(self, url: str, query: str = "") -> Optional[Dict]:
        """Загружает страницу и возвращает структуру с контентом"""
        logger.info(f"📥 Загрузка страницы: {url}")
        result = await self._fetch_and_parse(url, query)
        if result.get('success'):
            content_len = len(result.get('content', ''))
            logger.info(f"✅ Успешно: {url} ({content_len} символов)")
            self.stats['pages_processed'] += 1
            return result
        else:
            logger.warning(f"❌ Не удалось загрузить {url}: {result.get('error')}")
            return None

    async def search(self, query: str, num_results: int = 20) -> Dict:
        """Поиск ссылок через гибридный поисковик с фильтрацией мусора"""
        start = time.time()
        self.stats['searches'] += 1
        logger.info(f"🔍 Поиск: '{query}'")
        try:
            results = await hybrid_searcher.smart_search(query, num_results * 2)

            is_russian = bool(re.search('[а-яё]', query.lower()))
            filtered_results = []
            for r in results:
                url = r.get('url', '')
                title = r.get('title', '').lower()
                snippet = r.get('snippet', '').lower()

                if any(bad in url for bad in ['doubleclick', 'googleadservices', 'youtube.com', 'facebook.com', 'instagram', 'tiktok']):
                    continue

                if is_russian:
                    domain = urlparse(url).netloc
                    bad_domains = ['es.wikipedia', 'en.wikipedia', 'de.wikipedia', 'fr.wikipedia',
                                   'it.wikipedia', 'pt.wikipedia', 'ja.wikipedia', 'zh.wikipedia',
                                   'wikidata', 'wikimedia', 'wiktionary']
                    if any(bad in domain for bad in bad_domains):
                        topic_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
                        title_words = set(re.findall(r'\b\w{4,}\b', title))
                        snippet_words = set(re.findall(r'\b\w{4,}\b', snippet))
                        if not (topic_words & title_words) and not (topic_words & snippet_words):
                            continue

                filtered_results.append(r)

            formatted = []
            for r in filtered_results[:num_results]:
                url = r.get('url')
                if not url:
                    continue
                domain = urlparse(url).netloc
                formatted.append({
                    'url': url,
                    'title': r.get('title', ''),
                    'snippet': r.get('snippet', ''),
                    'domain': domain,
                    'is_priority': any(p in domain for p in self.priority_domains),
                    'relevance_score': r.get('final_score', r.get('relevance', 0.7))
                })

            formatted = [f for f in formatted if not self._is_blacklisted(f['url'])]
            formatted.sort(key=lambda x: x.get('is_priority', False), reverse=True)

            elapsed = time.time() - start
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['searches'] - 1) + elapsed)
                / self.stats['searches']
            )
            return {
                'success': True,
                'query': query,
                'results': formatted[:num_results],
                'stats': {'total_found': len(results), 'filtered': len(formatted), 'time': elapsed}
            }
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            self.stats['errors'] += 1
            return {'success': False, 'error': str(e), 'results': []}

    async def investigate_topic_advanced(self, topic: str, questions: List[str] = None) -> Dict:
        """Исследование темы: несколько поисков, загрузка страниц, извлечение чанков"""
        start = time.time()
        logger.info(f"🔬 ГЛУБОКОЕ ИССЛЕДОВАНИЕ: {topic}")
        if not questions:
            questions = self._generate_search_queries(topic)
        logger.info(f"🔎 Поисковые запросы: {questions}")

        search_tasks = [self.search(q, num_results=10) for q in questions[:5]]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        all_urls = []
        for res in search_results:
            if isinstance(res, dict) and res.get('success'):
                all_urls.extend([r['url'] for r in res['results']])

        unique_urls = list(dict.fromkeys(all_urls))
        logger.info(f"📊 Найдено уникальных URL: {len(unique_urls)}")

        fetch_tasks = []
        for url in unique_urls[:10]:
            if not self._is_blacklisted(url):
                fetch_tasks.append(self.fetch_page_content(url, topic))
        pages = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        valid_pages = [p for p in pages if isinstance(p, dict) and p.get('success')]
        logger.info(f"✅ Успешно загружено: {len(valid_pages)} страниц")

        all_chunks = []
        for page in valid_pages:
            chunks = self._extract_chunks(page.get('content', ''), page['url'])
            all_chunks.extend(chunks)

        unique_chunks = self._deduplicate_chunks(all_chunks)
        logger.info(f"📦 Извлечено чанков: {len(all_chunks)}, уникальных: {len(unique_chunks)}")

        return {
            'success': True,
            'topic': topic,
            'pages_processed': len(valid_pages),
            'content_chunks': unique_chunks,
            'stats': {
                'total_chunks': len(all_chunks),
                'unique': len(unique_chunks),
                'time': time.time() - start if 'start' in locals() else 0
            }
        }

    async def _fetch_and_parse(self, url: str, query: str) -> Dict:
        """Загружает HTML и извлекает контент (с использованием trafilatura при возможности)"""
        try:
            html = await self._fetch_url(url)
            if not html:
                return {'success': False, 'url': url, 'error': 'fetch failed'}

            extracted = None

            # 1. Пробуем trafilatura
            if TRAFILATURA_AVAILABLE:
                try:
                    # Извлекаем основной текст
                    text = trafilatura.extract(html, include_comments=False, include_tables=False, no_fallback=False)
                    if text and len(text) > 500:
                        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
                        title = title_match.group(1).strip() if title_match else ''
                        extracted = {
                            'title': title,
                            'content': text,
                            'language': 'ru' if re.search('[а-яё]', text) else 'en',
                            'has_images': False,
                            'has_tables': False
                        }
                        logger.debug(f"✅ trafilatura извлекла {len(text)} символов для {url}")
                except Exception as e:
                    logger.warning(f"⚠️ trafilatura error for {url}: {e}")

            # 2. Если trafilatura не сработала, пробуем readability
            if not extracted and READABILITY_AVAILABLE:
                try:
                    doc = Document(html)
                    content = doc.summary()
                    title = doc.title()
                    # Очищаем от оставшихся HTML-тегов
                    content = re.sub(r'<[^>]+>', ' ', content)
                    content = re.sub(r'\s+', ' ', content).strip()
                    if content and len(content) > 300:
                        extracted = {
                            'title': title,
                            'content': content,
                            'language': 'ru' if re.search('[а-яё]', content) else 'en',
                            'has_images': False,
                            'has_tables': False
                        }
                        logger.debug(f"✅ readability извлекла {len(content)} символов для {url}")
                except Exception as e:
                    logger.warning(f"⚠️ readability error for {url}: {e}")

            # 3. Если ничего не сработало, используем наш парсер
            if not extracted:
                loop = asyncio.get_event_loop()
                extracted = await loop.run_in_executor(
                    None,
                    self._parse_html,
                    html, url
                )

            # Проверка длины контента
            content_len = len(extracted.get('content', ''))
            min_len = self.config.get('min_content_length', 800)
            if content_len < min_len:
                logger.debug(f"⚠️ Контент слишком короткий ({content_len} < {min_len}): {url}")
                return {'success': False, 'url': url, 'error': f'content too short ({content_len})'}

            # Оценка качества
            quality = self._calc_quality(extracted, query)
            extracted['quality_score'] = quality

            return {
                'success': True,
                'url': url,
                'title': extracted.get('title', ''),
                'content': extracted.get('content', ''),
                'quality_score': quality,
                'metadata': {
                    'language': extracted.get('language', 'ru'),
                    'has_images': extracted.get('has_images', False),
                    'has_tables': extracted.get('has_tables', False),
                }
            }
        except Exception as e:
            logger.error(f"Ошибка обработки {url}: {e}")
            return {'success': False, 'url': url, 'error': str(e)}

    async def _fetch_url(self, url: str) -> Optional[str]:
        """Загрузка URL с кэшем и обработкой ошибок"""
        key = hashlib.md5(url.encode()).hexdigest()
        if key in self.cache:
            if time.time() - self.cache[key]['timestamp'] < self.cache_ttl:
                self.stats['cache_hits'] += 1
                logger.debug(f"🔵 Кэш HIT: {url}")
                return self.cache[key]['content']
        self.stats['cache_misses'] += 1

        try:
            await asyncio.sleep(random.uniform(0.3, 0.7))
            headers = {'User-Agent': random.choice(self.user_agents)}
            async with self.session.get(url, headers=headers, ssl=False, timeout=15) as resp:
                if resp.status != 200:
                    logger.debug(f"⚠️ HTTP {resp.status} для {url}")
                    return None
                try:
                    content = await resp.text()
                except UnicodeDecodeError:
                    raw = await resp.read()
                    content = raw.decode('utf-8', errors='ignore')
                self.cache[key] = {'content': content, 'timestamp': time.time()}
                self.stats['requests'] += 1
                logger.debug(f"🟢 Загружено: {url} ({len(content)} байт)")
                return content
        except asyncio.TimeoutError:
            logger.debug(f"⏰ Таймаут {url}")
            return None
        except Exception as e:
            logger.debug(f"⚠️ Ошибка загрузки {url}: {e}")
            return None

    def _parse_html(self, html: str, url: str) -> Dict:
        """Синхронный парсинг через BeautifulSoup с агрессивной очисткой (резервный метод)"""
        result = {
            'title': '',
            'content': '',
            'language': 'ru',
            'has_images': False,
            'has_tables': False
        }
        try:
            soup = BeautifulSoup(html, 'lxml')

            # Специальная обработка habr.com
            if 'habr.com' in url:
                article_body = soup.find('div', class_='article__body')
                if article_body:
                    for tag in article_body.find_all(['script', 'style', 'aside', 'div.comments',
                                                      'div.article__meta', 'div.post-meta', 'div.author-info',
                                                      'div.company-info', 'div.tags', 'div.hubs', 'div.stats',
                                                      'div.share', 'div.subscribe', 'div.banner', 'div.advertisement',
                                                      'div.recommendations', 'div.related', 'div.footer',
                                                      'div.article__footer', 'div.article__aside', 'div.article__header',
                                                      'div.voting', 'div.favs_count', 'div.views_count', 'div.time',
                                                      'span.time', 'span.views', 'span.comments', 'span.favs']):
                        tag.decompose()
                    paragraphs = []
                    for p in article_body.find_all(['p', 'li', 'blockquote', 'div.paragraph']):
                        text = p.get_text(strip=True)
                        if len(text) > 30:
                            paragraphs.append(text)
                    content = '\n\n'.join(paragraphs)
                    title = ''
                    for selector in ['h1', 'h2', 'h3', '.post__title', '.article__title', 'title', 'meta[property="og:title"]']:
                        elem = soup.select_one(selector)
                        if elem:
                            if elem.name == 'meta':
                                title = elem.get('content', '')
                            else:
                                title = elem.get_text().strip()
                            if title:
                                break
                    if not title and soup.title:
                        title = soup.title.string.strip()
                    if len(content) > 500000:
                        content = content[:500000]
                    result = {
                        'title': title[:200],
                        'content': content,
                        'language': 'ru',
                        'has_images': bool(soup.find_all('img')),
                        'has_tables': bool(soup.find_all('table'))
                    }
                    return result

            # Общая очистка
            for tag in soup(['script', 'style', 'noscript', 'iframe', 'nav', 'footer',
                           'header', 'aside', 'form', 'button', 'input', 'meta', 'link']):
                tag.decompose()

            for cls in ['comment', 'comments', 'sidebar', 'widget', 'advertisement',
                       'banner', 'popup', 'modal', 'cookie', 'newsletter', 'subscribe',
                       'share', 'social', 'menu', 'breadcrumb', 'pagination', 'related']:
                for elem in soup.find_all(class_=lambda c: c and cls in c.lower()):
                    elem.decompose()

            h1 = soup.find('h1')
            if h1:
                result['title'] = h1.get_text().strip()[:200]
            if not result['title'] and soup.title:
                result['title'] = soup.title.string.strip()[:200]

            for selector in ['article', 'main', 'div.content', 'div.article', 'div.post', '#content', '.content']:
                elem = soup.select_one(selector)
                if elem:
                    content = elem.get_text(separator='\n', strip=True)
                    break
            else:
                if soup.body:
                    lines = soup.body.get_text(separator='\n', strip=True).split('\n')
                    lines = [l.strip() for l in lines if len(l.strip()) > 40]
                    content = '\n'.join(lines)
                else:
                    content = ''

            # Очистка Википедии
            if 'wikipedia.org' in url:
                lines = content.split('\n')
                skip = [
                    'Материал из Википедии', 'Стабильная версия', 'проверенная',
                    'Перейти к навигации', 'Перейти к поиску', 'Скрытые категории',
                    'Категория:', 'Источник —', 'Эта страница в последний раз',
                    'Лицензия Creative Commons', 'Для улучшения этой статьи',
                    'ISBN', 'Шаблон:', 'Внешние ссылки', 'Примечания', 'Ссылки',
                    '^ [0-9]+ ', '↑', '↓', '←', '→'
                ]
                for pattern in skip:
                    lines = [l for l in lines if pattern not in l]

                processed_lines = []
                for l in lines:
                    original = l
                    l = re.sub(r'\[\[[^\]]+\]\]', '', l)
                    l = re.sub(r'\{\{[^}]+\}\}', '', l)
                    l = re.sub(r'={2,}', '', l)
                    l = re.sub(r'^\*\s*', '', l)
                    l = re.sub(r'^#\s*', '', l)
                    l = re.sub(r'^\|\s*', '', l)

                    l = re.sub(r'\[\d+\]', '', l)
                    l = re.sub(r'\^\{\d+\}', '', l)
                    l = re.sub(r'\|\^?\{\d+\}', '', l)
                    l = re.sub(r'\s+\d+(?:\s+\d+)*\s*$', '', l)
                    l = re.sub(r'\s+\d+\.\d+\.\d+\s*', ' ', l)

                    l = re.sub(r'\\[a-zA-Z]+', ' ', l)
                    l = re.sub(r'[{}]', ' ', l)

                    l = l.strip()
                    if len(l) < 30:
                        if len(original.strip()) > 100:
                            l = original.strip()
                            l = re.sub(r'\[\d+\]', '', l)
                            l = re.sub(r'[{}]', '', l)
                            l = re.sub(r'\s+', ' ', l)
                            l = l.strip()
                            if len(l) >= 30:
                                processed_lines.append(l)
                        continue
                    processed_lines.append(l)
                lines = processed_lines

                seen = set()
                unique = []
                for l in lines:
                    norm = l[:100].lower()
                    if norm not in seen:
                        seen.add(norm)
                        unique.append(l)
                content = '\n'.join(unique)

            content = self.text_cleaner.clean(content)
            max_len = self.config.get('max_content_length', 100000)
            if len(content) > max_len:
                content = content[:max_len]

            result['content'] = content
            result['has_images'] = bool(soup.find_all('img'))
            result['has_tables'] = bool(soup.find_all('table'))
            result['language'] = 'ru' if re.search('[а-яё]', content) else 'en'

        except Exception as e:
            logger.error(f"Ошибка парсинга {url}: {e}")
        return result

    def _extract_chunks(self, content: str, source_url: str) -> List[Dict]:
        """Извлечение чанков с очисткой через TextCleaner"""
        if not content:
            return []

        # Применяем базовую очистку
        content = self.text_cleaner.clean(content)

        # Разбиваем на параграфы
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 100]

        if not paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', content)
            paragraphs = [s for s in sentences if len(s) > 80]

        chunks = []
        for i, para in enumerate(paragraphs[:8]):  # максимум 8 чанков
            # Простая проверка: предложение должно быть осмысленным
            if len(para) < 50 or not any(c.isalpha() for c in para):
                continue
            chunks.append({
                'chunk_id': f"{hashlib.md5(source_url.encode()).hexdigest()[:8]}_{i}",
                'text': para[:8000],
                'source_url': source_url,
                'length': len(para)
            })
        return chunks

    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for c in chunks:
            sig = hashlib.md5(c['text'][:200].encode()).hexdigest()
            if sig not in seen:
                seen.add(sig)
                unique.append(c)
        return unique

    def _calc_quality(self, extracted: Dict, query: str) -> float:
        score = 0.0
        content = extracted.get('content', '')
        title = extracted.get('title', '')
        l = len(content)
        if l >= 3000:
            score += 0.4
        elif l >= 1500:
            score += 0.3
        elif l >= 800:
            score += 0.2
        elif l >= 300:
            score += 0.1
        if query.lower() in title.lower():
            score += 0.3
        elif any(w in title.lower() for w in query.lower().split()):
            score += 0.2
        return min(1.0, score)

    def _generate_search_queries(self, topic: str) -> List[str]:
        clean = topic.strip().rstrip('?')
        prefixes = ['что такое ', 'кто такой ', 'кто такая ', 'определение ', 'биография ']
        for p in prefixes:
            if clean.lower().startswith(p):
                clean = clean[len(p):].strip()
                break
        queries = [
            f"определение {clean}",
            f"что такое {clean}",
            f"{clean} биография",
            f"{clean} википедия",
            f"{clean} информация",
        ]
        parts = clean.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            queries.append(f"{last_name} {parts[0]}")
        return list(dict.fromkeys(queries))[:5]

    def _is_blacklisted(self, url: str) -> bool:
        domain = urlparse(url).netloc.lower()
        for bad in self.blacklist:
            if bad in domain:
                return True
        return False

    async def _load_cache(self):
        try:
            f = os.path.join(self.cache_dir, 'cache.json')
            if os.path.exists(f):
                async with aiofiles.open(f, 'r', encoding='utf-8') as fp:
                    self.cache = json.loads(await fp.read())
                logger.info(f"📦 Загружен кэш детектива: {len(self.cache)} записей")
        except Exception as e:
            logger.error(f"Ошибка загрузки кэша: {e}")

    async def _save_cache(self):
        try:
            f = os.path.join(self.cache_dir, 'cache.json')
            async with aiofiles.open(f, 'w', encoding='utf-8') as fp:
                await fp.write(json.dumps(self.cache, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша: {e}")

    async def clear_cache(self):
        self.cache.clear()
        await self._save_cache()
        logger.info("🧹 Кэш детектива очищен")

    async def get_stats(self) -> Dict:
        return {
            'searches': self.stats['searches'],
            'pages_processed': self.stats['pages_processed'],
            'requests': self.stats['requests'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
            'errors': self.stats['errors'],
            'avg_response_time': self.stats['avg_response_time'],
            'unique_domains': len(self.stats['unique_domains'])
        }

    async def health_check(self) -> Dict:
        try:
            async with self.session.get('https://www.google.com', timeout=5) as resp:
                return {'healthy': resp.status == 200}
        except:
            return {'healthy': False}

    async def cleanup(self):
        if self.session:
            await self.session.close()
        await self._save_cache()
        logger.info("✅ Detective завершён")