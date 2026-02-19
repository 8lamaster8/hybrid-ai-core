"""
⚖️ КОМИТЕТ КАЧЕСТВА - Многоуровневая фильтрация контента
Оценка релевантности, качества, уникальности, отбрасывание мусора
"""

import re
import os
import yaml
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import Counter
from difflib import SequenceMatcher
from urllib.parse import urlparse

from appp.core.logging import logger
from appp.utils.text_processor import TextCleaner


class QualityCommittee:
    """
    Комитет по оценке качества информации.
    Состоит из нескольких экспертов, каждый оценивает свой аспект.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Пороговые значения (из config или дефолтные)
        self.min_relevance_score = config.get('min_relevance_score', 0.65)
        self.min_quality_score = config.get('min_quality_score', 0.7)
        self.min_uniqueness_score = config.get('min_uniqueness_score', 0.6)
        self.max_text_similarity = config.get('max_text_similarity', 0.85)
        
        # Блокировки (из config)
        self.blacklist_domains = set(config.get('blacklist_domains', []))
        self.blocked_keywords = config.get('blocked_keywords', [])
        self.required_keywords = config.get('required_keywords', [])
        
        # Настройки проверок
        self.enable_embedding_check = config.get('enable_embedding_check', True)
        self.embedding_threshold = config.get('embedding_threshold', 0.75)
        self.min_sentences = config.get('min_sentences', 3)
        self.max_sentence_length = config.get('max_sentence_length', 200)
        self.language = config.get('language', 'ru')
        
        # --- ЗАГРУЗКА КОНФИГУРАЦИИ КАЧЕСТВА ИЗ YAML ---
        self._load_quality_config()
        # ------------------------------------------------
        
        # Инструменты
        self.text_cleaner = TextCleaner()
        
        # Кэш для уже проверенных URL (чтобы не дублировать)
        self.checked_urls: Set[str] = set()
        
        # Статистика
        self.stats = {
            'evaluations': 0,
            'approved': 0,
            'rejected': 0,
            'rejection_reasons': Counter(),
            'avg_relevance': 0.0,
            'avg_quality': 0.0,
            'avg_uniqueness': 0.0
        }
        
        logger.info("⚖️ QualityCommittee создан")
    
    def _load_quality_config(self):
        """Загрузка дополнительной конфигурации из quality.yaml"""
        # Определяем путь к файлу конфигурации
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Поднимаемся на уровень проекта: ../../../configs/quality.yaml
        config_path = os.path.join(current_dir, '..', '..', '..', 'configs', 'quality.yaml')
        config_path = os.path.normpath(config_path)
        
        # Значения по умолчанию
        self.priority_domains = set()
        self.low_trust_domains = set()
        self.ad_indicators = []
        self.junk_phrases = []
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yml = yaml.safe_load(f)
                
                if yml:
                    self.priority_domains = set(yml.get('priority_domains', []))
                    self.low_trust_domains = set(yml.get('low_trust_domains', []))
                    self.ad_indicators = yml.get('ad_indicators', [])
                    self.junk_phrases = yml.get('junk_phrases', [])
                    
                    logger.info(f"📁 Загружена конфигурация качества из {config_path}")
                    logger.info(f"   Приоритетных доменов: {len(self.priority_domains)}")
                    logger.info(f"   Низкокачественных доменов: {len(self.low_trust_domains)}")
            except Exception as e:
                logger.error(f"Ошибка загрузки quality.yaml: {e}")
        else:
            logger.warning(f"⚠️ Файл {config_path} не найден, используются значения по умолчанию")
    
    async def initialize(self):
        """Инициализация комитета"""
        logger.info("🔄 Инициализация QualityCommittee...")
        return True
    
    async def evaluate(self, document: Dict) -> Dict:
        """
        Полная оценка документа всеми экспертами.
        
        Args:
            document: Документ с полями url, title, content, snippet и др.
            
        Returns:
            Dict с решением и детализацией
        """
        self.stats['evaluations'] += 1
        
        url = document.get('url', '')
        content = document.get('content', document.get('snippet', ''))
        title = document.get('title', '')
        
        # Проверяем, не оценивали ли этот URL ранее
        if url in self.checked_urls:
            uniqueness_score = 0.3
        else:
            self.checked_urls.add(url)
            uniqueness_score = 1.0
        
        # 1. Эксперт по релевантности
        relevance_result = await self._check_relevance(document)
        
        # 2. Эксперт по качеству контента
        quality_result = await self._check_quality(content, title)
        
        # 3. Эксперт по уникальности (антиплагиат)
        uniqueness_result = await self._check_uniqueness(content, url, uniqueness_score)
        
        # 4. Эксперт по спаму/черным спискам
        spam_result = self._check_spam(document)
        
        # Собираем все оценки
        scores = {
            'relevance': relevance_result['score'],
            'quality': quality_result['score'],
            'uniqueness': uniqueness_result['score'],
            'spam': spam_result['score']
        }
        
        # Финальное решение
        approved = (
            scores['relevance'] >= self.min_relevance_score and
            scores['quality'] >= self.min_quality_score and
            scores['uniqueness'] >= self.min_uniqueness_score and
            scores['spam'] >= 0.5
        )
        
        # Собираем причины отказа
        rejection_reasons = []
        if scores['relevance'] < self.min_relevance_score:
            rejection_reasons.append(f"low_relevance ({scores['relevance']:.2f})")
        if scores['quality'] < self.min_quality_score:
            rejection_reasons.append(f"low_quality ({scores['quality']:.2f})")
        if scores['uniqueness'] < self.min_uniqueness_score:
            rejection_reasons.append(f"low_uniqueness ({scores['uniqueness']:.2f})")
        if scores['spam'] < 0.5:
            rejection_reasons.append(f"spam_or_blacklisted ({scores['spam']:.2f})")
        
        # Обновляем статистику
        if approved:
            self.stats['approved'] += 1
        else:
            self.stats['rejected'] += 1
            for reason in rejection_reasons:
                self.stats['rejection_reasons'][reason] += 1
        
        # Обновляем средние
        n = self.stats['evaluations']
        self.stats['avg_relevance'] = (self.stats['avg_relevance'] * (n - 1) + scores['relevance']) / n
        self.stats['avg_quality'] = (self.stats['avg_quality'] * (n - 1) + scores['quality']) / n
        self.stats['avg_uniqueness'] = (self.stats['avg_uniqueness'] * (n - 1) + scores['uniqueness']) / n
        
        return {
            'approved': approved,
            'scores': scores,
            'details': {
                'relevance': relevance_result.get('details', {}),
                'quality': quality_result.get('details', {}),
                'uniqueness': uniqueness_result.get('details', {}),
                'spam': spam_result.get('details', {})
            },
            'rejection_reasons': rejection_reasons,
            'document_url': url
        }
    
    async def batch_evaluate(self, documents: List[Dict]) -> List[Dict]:
        """
        Пакетная оценка нескольких документов.
        Возвращает только одобренные документы.
        """
        approved_docs = []
        
        for doc in documents:
            result = await self.evaluate(doc)
            if result['approved']:
                doc['committee_scores'] = result['scores']
                doc['committee_approved'] = True
                approved_docs.append(doc)
        
        logger.info(f"⚖️ Пакетная оценка: {len(documents)} -> одобрено {len(approved_docs)}")
        return approved_docs
    
    async def _check_relevance(self, document: Dict) -> Dict:
        score = 0.0
        details = {}
        
        query = document.get('query', '')
        title = document.get('title', '')
        snippet = document.get('snippet', '')
        domain = document.get('domain', '')
        
        if query:
            query_lower = query.lower()
            title_lower = title.lower()
            snippet_lower = snippet.lower()
            
            # 1. Совпадение в заголовке (вес 0.4)
            if query_lower in title_lower:
                title_match = 1.0
            else:
                query_words = set(query_lower.split())
                title_words = set(title_lower.split())
                title_match = len(query_words & title_words) / max(1, len(query_words))
            score += title_match * 0.4
            details['title_match'] = title_match
            
            # 2. Совпадение в сниппете (вес 0.3)
            if query_lower in snippet_lower:
                snippet_match = 1.0
            else:
                query_words = set(query_lower.split())
                snippet_words = set(snippet_lower.split())
                snippet_match = len(query_words & snippet_words) / max(1, len(query_words))
            score += snippet_match * 0.3
            details['snippet_match'] = snippet_match
            
            # 3. Приоритетный домен (вес 0.2)
            domain_bonus = 0.2 if any(pd in domain for pd in self.priority_domains) else 0.0
            score += domain_bonus
            details['domain_bonus'] = domain_bonus
        else:
            score = 0.7
            details['no_query'] = True
        
        return {'score': min(score, 1.0), 'details': details}
    
    async def _check_quality(self, content: str, title: str = '') -> Dict:
        """Эксперт по качеству контента"""
        score = 0.0
        details = {}
        
        if not content:
            return {'score': 0.3, 'details': {'error': 'empty_content'}}
        
        # 1. Длина контента (вес 0.3)
        content_length = len(content)
        if content_length >= 3000:
            length_score = 1.0
        elif content_length >= 1500:
            length_score = 0.9
        elif content_length >= 800:
            length_score = 0.8
        elif content_length >= 400:
            length_score = 0.6
        elif content_length >= 200:
            length_score = 0.5
        else:
            length_score = 0.3
        score += length_score * 0.3
        details['length'] = content_length
        details['length_score'] = length_score
        
        # 2. Количество предложений (вес 0.2)
        sentences = self._split_sentences(content)
        num_sentences = len(sentences)
        if num_sentences >= 3:
            sentences_score = min(num_sentences / 8, 1.0)
        else:
            sentences_score = num_sentences / 3
        score += sentences_score * 0.2
        details['sentences'] = num_sentences
        details['sentences_score'] = sentences_score
        
        # 3. Разнообразие слов (вес 0.15)
        words = content.split()
        if words:
            unique_words = set(words)
            lexical_diversity = len(unique_words) / len(words)
            diversity_score = min(lexical_diversity * 1.2, 1.0)
            score += diversity_score * 0.15
            details['lexical_diversity'] = lexical_diversity
            details['diversity_score'] = diversity_score
        else:
            details['lexical_diversity'] = 0
        
        # 4. Читаемость (вес 0.15)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(num_sentences, 1)
        if 10 <= avg_sentence_length <= 25:
            readability_score = 1.0
        elif avg_sentence_length < 5:
            readability_score = 0.5
        elif avg_sentence_length > 40:
            readability_score = 0.5
        else:
            readability_score = 1 - abs(avg_sentence_length - 17) / 35
        score += readability_score * 0.15
        details['avg_sentence_length'] = avg_sentence_length
        details['readability_score'] = readability_score
        
        # 5. Отсутствие мусора (вес 0.2)
        junk_indicators = [r'http\S+', r'www\.\S+', r'\[\d+\]', r'↑', r'архивная копия']
        junk_count = 0
        for pattern in junk_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                junk_count += 1
        junk_score = max(0, 1 - junk_count * 0.1)
        score += junk_score * 0.2
        details['junk_count'] = junk_count
        details['junk_score'] = junk_score
        
        # 6. Наличие заголовка (вес 0.1)
        title_score = 1.0 if title and len(title) > 5 else 0.6 if title else 0.3
        score += title_score * 0.1
        details['title_score'] = title_score
        
        return {'score': min(score, 1.0), 'details': details}
    
    async def _check_uniqueness(self, content: str, url: str, base_score: float = 1.0) -> Dict:
        """Эксперт по уникальности."""
        score = base_score
        
        if len(content) < 500:
            score *= 0.7
        elif len(content) < 200:
            score *= 0.4
        
        if url in self.checked_urls:
            score *= 0.9
            is_duplicate = True
        else:
            self.checked_urls.add(url)
            is_duplicate = False
        
        return {
            'score': min(score, 1.0),
            'details': {
                'base_score': base_score,
                'length_penalty': score / base_score if base_score > 0 else 0,
                'is_duplicate_url': is_duplicate,
                'final_score': score
            }
        }
    
    def _check_spam(self, document: Dict) -> Dict:
        """
        Эксперт по спаму и черным спискам.
        """
        score = 1.0
        details = {}
        
        url = document.get('url', '')
        domain = document.get('domain', '')
        title = document.get('title', '')
        content = document.get('content', '') or document.get('snippet', '')
        
        # 1. Проверка домена по черному списку
        for black_domain in self.blacklist_domains:
            if black_domain in domain:
                score *= 0.2
                details['blacklisted_domain'] = black_domain
                break
        
        # --- НИЗКОКАЧЕСТВЕННЫЕ ДОМЕНЫ (из quality.yaml) ---
        for bad in self.low_trust_domains:
            if bad in domain:
                score *= 0.3
                details['low_trust_domain'] = bad
                break
        
        # --- РЕКЛАМНЫЕ МАРКЕРЫ (из quality.yaml) ---
        content_lower = (title + ' ' + content).lower()
        for ad in self.ad_indicators:
            if ad.lower() in content_lower:
                score *= 0.7
                if 'ad_indicators_hit' not in details:
                    details['ad_indicators_hit'] = []
                details['ad_indicators_hit'].append(ad)
        
        # 2. Проверка ключевых слов-маркеров спама (из config)
        for keyword in self.blocked_keywords:
            if keyword.lower() in content_lower:
                score *= 0.7
                if 'blocked_keywords_hit' not in details:
                    details['blocked_keywords_hit'] = []
                details['blocked_keywords_hit'].append(keyword)
        
        # 3. Проверка на наличие требуемых ключевых слов (если заданы)
        if self.required_keywords:
            required_ok = all(kw.lower() in content_lower for kw in self.required_keywords)
            if not required_ok:
                score *= 0.5
                details['required_keywords_missing'] = True
        
        # 4. Слишком много ссылок в тексте
        link_count = content.count('http') + content.count('www.')
        if link_count > 5:
            score *= 0.8
            details['too_many_links'] = link_count
        
        return {
            'score': score,
            'details': details
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Разбивка на предложения"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def get_stats(self) -> Dict:
        """Статистика комитета"""
        return {
            'evaluations': self.stats['evaluations'],
            'approved': self.stats['approved'],
            'rejected': self.stats['rejected'],
            'approval_rate': self.stats['approved'] / max(1, self.stats['evaluations']),
            'rejection_reasons': dict(self.stats['rejection_reasons'].most_common(10)),
            'avg_relevance': self.stats['avg_relevance'],
            'avg_quality': self.stats['avg_quality'],
            'avg_uniqueness': self.stats['avg_uniqueness'],
            'checked_urls_count': len(self.checked_urls),
            'priority_domains_count': len(self.priority_domains),
            'low_trust_domains_count': len(self.low_trust_domains)
        }
    
    async def get_committee_stats(self) -> Dict:
        """Алиас для get_stats"""
        return await self.get_stats()
    
    async def health_check(self) -> Dict:
        """Проверка здоровья"""
        return {
            'healthy': True,
            'message': 'QualityCommittee is operational',
            'timestamp': datetime.now().isoformat()
        }