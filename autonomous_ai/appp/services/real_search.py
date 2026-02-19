"""
🌐 Единый поисковый движок — DuckDuckGo + Google + Fallback
Используется всеми компонентами.
"""

import asyncio
import aiohttp
import re
import random
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import quote_plus, urlparse
from collections import OrderedDict

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
    logger.info("✅ Использую ddgs")
except ImportError:
    try:
        from duckduckgo_search import DDGS
        logger.warning("⚠️ duckduckgo-search (старая версия)")
    except ImportError:
        DDGS = None
        logger.error("❌ Не установлен duckduckgo-search")


class QueryAnalyzer:
    """Анализ поисковых запросов"""

    def __init__(self):
        self.patterns = {
            'factual': ['что такое', 'кто такой', 'когда', 'где', 'сколько', 'зачем'],
            'how_to': ['как сделать', 'как создать', 'инструкция', 'руководство'],
            'news': ['новости', 'сегодня', 'вчера', 'события'],
            'academic': ['исследование', 'теория', 'доказательство', 'научный'],
            'tutorial': ['курс', 'обучение', 'урок', 'практика'],
            'comparison': ['vs', 'против', 'сравнение', 'лучше чем']
        }
        self.stop_words = {
            'и', 'в', 'на', 'с', 'по', 'для', 'из', 'от', 'к', 'у', 'о', 'не',
            'что', 'как', 'где', 'когда', 'почему', 'зачем', 'кто', 'сколько'
        }
        self.stats = {'queries': 0, 'types': {}}

    def analyze(self, query: str) -> Dict:
        ql = query.lower()
        qtype = 'general'
        for t, pats in self.patterns.items():
            if any(p in ql for p in pats):
                qtype = t
                break
        words = re.findall(r'\b\w+\b', ql)
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        self.stats['queries'] += 1
        self.stats['types'][qtype] = self.stats['types'].get(qtype, 0) + 1
        return {
            'original': query,
            'type': qtype,
            'keywords': list(set(keywords)),
            'language': 'ru' if re.search('[а-яё]', ql) else 'en',
            'complexity': 'high' if len(keywords) > 3 else 'medium' if len(keywords) > 1 else 'low',
            'word_count': len(query.split()),
            'analyzed_at': datetime.now().isoformat()
        }

    def get_stats(self) -> Dict:
        return self.stats.copy()


class RealSearchEngine:
    """Реальный поиск с кэшем и защитой от блокировок"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119.0.0.0 Safari/537.36',
        ]
        self.session: Optional[aiohttp.ClientSession] = None
        # Кэш с лимитом (LRU)
        self._cache = OrderedDict()
        self.cache_maxsize = 1000
        self.cache_ttl = 6 * 3600  # 6 часов
        self.stats = {
            'ddg': 0,
            'google': 0,
            'fallback': 0,
            'success_rate': 0.0,
            'avg_results': 0.0
        }

    async def initialize(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': random.choice(self.user_agents)},
        )
        return True

    async def search(self, query: str, max_results: int = 10) -> List[Dict]:
        cache_key = f"{query}_{max_results}"
        # Проверка кэша
        if cache_key in self._cache:
            ts, results = self._cache[cache_key]
            if (datetime.now() - ts).total_seconds() < self.cache_ttl:
                self._cache.move_to_end(cache_key)  # LRU touch
                return results

        results = []
        methods = [
            self._ddg_api,
            self._google_organic,
            self._google_ajax,
            self._ddg_html,
            self._fallback
        ]

        for method in methods:
            if len(results) >= max_results:
                break
            try:
                res = await method(query, max_results - len(results))
                if res:
                    seen = set(r['url'] for r in results)
                    for r in res:
                        if r['url'] not in seen:
                            results.append(r)
                            seen.add(r['url'])
                    logger.info(f"{method.__name__}: +{len(res)}")
                    break  # Первый успешный метод даёт результат
            except Exception as e:
                logger.debug(f"{method.__name__} error: {e}")

        # Сохраняем в кэш с управлением размером
        self._cache[cache_key] = (datetime.now(), results)
        if len(self._cache) > self.cache_maxsize:
            self._cache.popitem(last=False)  # удаляем самую старую

        # Статистика
        if results:
            source = results[0]['source']
            if source in ('duckduckgo', 'ddg_html'):
                self.stats['ddg'] += 1
            elif 'google' in source:
                self.stats['google'] += 1
            else:
                self.stats['fallback'] += 1
        else:
            self.stats['fallback'] += 1

        return results[:max_results]

    async def _ddg_api(self, query: str, max_results: int) -> List[Dict]:
        if not DDGS:
            return []
        ddgs = DDGS()
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: list(ddgs.text(
                    query,
                    region='ru-ru',
                    safesearch='moderate',
                    max_results=max_results
                ))
            )
        except:
            return []
        formatted = []
        for item in results:
            url = item.get('href') or item.get('url')
            if not url:
                continue
            formatted.append({
                'url': url,
                'title': item.get('title', ''),
                'snippet': item.get('body', ''),
                'source': 'duckduckgo',
                'relevance': 0.8
            })
        return formatted

    async def _google_organic(self, query: str, max_results: int) -> List[Dict]:
        try:
            from googlesearch import search
            loop = asyncio.get_event_loop()
            urls = await loop.run_in_executor(
                None,
                lambda: list(search(query, num_results=max_results, lang='ru'))
            )
            return [{
                'url': u,
                'title': '',
                'snippet': '',
                'source': 'google',
                'relevance': 0.7
            } for u in urls]
        except ImportError:
            return []
        except:
            return []

    async def _google_ajax(self, query: str, max_results: int) -> List[Dict]:
        url = f"https://www.google.com/search?q={quote_plus(query)}&hl=ru&num={max_results}"
        async with self.session.get(url) as resp:
            if resp.status != 200:
                return []
            html = await resp.text()
            results = []
            pattern = r'<a href="/url\?q=(https?://[^&]+)&[^"]+"[^>]*><h3[^>]*>([^<]+)</h3>'
            for match in re.finditer(pattern, html, re.IGNORECASE):
                url = match.group(1)
                if any(bad in url for bad in ['google.com', 'youtube.com', 'facebook.com']):
                    continue
                results.append({
                    'url': url,
                    'title': match.group(2),
                    'snippet': '',
                    'source': 'google_ajax',
                    'relevance': 0.6
                })
            return results[:max_results]

    async def _ddg_html(self, query: str, max_results: int) -> List[Dict]:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        async with self.session.get(url) as resp:
            if resp.status != 200:
                return []
            html = await resp.text()
            results = []
            pattern = r'class="result__title".*?href="([^"]+)".*?class="result__snippet"[^>]*>([^<]+)'
            for match in re.finditer(pattern, html, re.DOTALL):
                url = match.group(1)
                snippet = match.group(2).strip()
                title_match = re.search(
                    r'class="result__title"[^>]*>([^<]+)',
                    html[match.start():match.end()]
                )
                title = title_match.group(1) if title_match else ''
                results.append({
                    'url': url,
                    'title': title.strip(),
                    'snippet': snippet,
                    'source': 'ddg_html',
                    'relevance': 0.7
                })
            return results[:max_results]

    async def _fallback(self, query: str, max_results: int) -> List[Dict]:
        clean = query.lower().strip()
        domains = [
            'ru.wikipedia.org',
            'habr.com',
            'vc.ru',
            'tproger.ru',
            'ria.ru'
        ]
        encoded = quote_plus(clean)
        results = []
        for i, dom in enumerate(domains[:max_results]):
            if 'wikipedia' in dom:
                url = f"https://{dom}/wiki/{clean.replace(' ', '_')}"
            else:
                url = f"https://{dom}/search?q={encoded}"
            results.append({
                'url': url,
                'title': f"{clean} — {dom}",
                'snippet': f"Информация по запросу '{clean}'",
                'source': 'fallback',
                'relevance': 0.6 - i * 0.05
            })
        return results

    async def get_stats(self) -> Dict:
        total = self.stats['ddg'] + self.stats['google'] + self.stats['fallback']
        self.stats['success_rate'] = (self.stats['ddg'] + self.stats['google']) / max(total, 1)
        return {
            'search_engine': self.stats.copy(),
            'cache_size': len(self._cache),
            'cache_maxsize': self.cache_maxsize
        }

    async def clear_cache(self):
        self._cache.clear()

    async def close(self):
        if self.session:
            await self.session.close()


class HybridSearcher:
    """Умный поиск с анализом запроса и ранжированием"""

    def __init__(self, config: Dict = None):
        self.searcher = RealSearchEngine(config)
        self.analyzer = QueryAnalyzer()
        self.source_weights = {
            'duckduckgo': 0.4,
            'google': 0.3,
            'google_ajax': 0.2,
            'ddg_html': 0.25,
            'fallback': 0.1
        }
        self._initialized = False

    async def initialize(self):
        if not self._initialized:
            await self.searcher.initialize()
            self._initialized = True

    async def smart_search(self, query: str, max_results: int = 10) -> List[Dict]:
        analysis = self.analyzer.analyze(query)
        raw = await self.searcher.search(query, max_results * 2)
        scored = []
        for r in raw:
            score = self._score(r, analysis)
            scored.append({**r, 'final_score': score})
        scored.sort(key=lambda x: x['final_score'], reverse=True)
        return scored[:max_results]

    def _score(self, result: Dict, analysis: Dict) -> float:
        base = result.get('relevance', 0.5)
        url = result.get('url', '').lower()
        domain = urlparse(url).netloc.lower()
        title = result.get('title', '').lower()
        # Приоритетные домены
        if any(p in domain for p in ['wikipedia', 'habr', 'arxiv', 'stackoverflow']):
            base += 0.2
        # Ключевые слова в заголовке
        keywords = analysis.get('keywords', [])
        title_matches = sum(1 for kw in keywords if kw in title)
        base += 0.05 * title_matches
        # Бонус за источник
        source = result.get('source', '')
        base += self.source_weights.get(source, 0.0)
        # Штраф за мусор
        if any(bad in domain for bad in ['spam', 'clickbait', 'adult']):
            base -= 0.5
        return max(0.0, min(1.0, base))

    async def get_stats(self) -> Dict:
        return {
            'hybrid': {'source_weights': self.source_weights.copy()},
            'searcher': await self.searcher.get_stats(),
            'analyzer': self.analyzer.get_stats()
        }

    async def close(self):
        await self.searcher.close()


# Глобальный экземпляр (единый для всех)
hybrid_searcher = HybridSearcher()