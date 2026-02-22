"""
📚 АНАЛИТИК ЗНАНИЙ — извлечение фактов с метаданными + ранжирование
"""

import asyncio
import re
import hashlib
import yaml
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from appp.core.logging import logger
from appp.utils.text_processor import TextCleaner
from appp.services.ranking.ranking_service import RankingService, Fact
from appp.services.embedding.bge_m3 import embedder

# Для NER (GLiNER2)
try:
    from gliner2 import GLiNER2
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    logger.warning("⚠️ GLiNER2 не установлена, NER отключён")

# Для семантической дедупликации
try:
    from sentence_transformers import util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ sentence-transformers не установлена, семантическая дедупликация отключена")


class KnowledgeAnalyst:
    def __init__(self, config: Dict):
        self.config = config
        self.chunk_size = config.get('chunk_size', 1500)
        self.chunk_overlap = config.get('chunk_overlap', 300)
        self.min_chunk_length = config.get('min_chunk_length', 500)
        self.max_chunks_per_document = config.get('max_chunks_per_document', 50)
        self.enable_summarization = config.get('enable_summarization', True)
        self.language = config.get('language', 'ru')
        self.min_confidence = config.get('min_confidence', 0.6)
        self._last_facts_metadata = []

        self.text_cleaner = TextCleaner()

        # Загружаем конфигурации из YAML (корень проекта)
        self.profiles_config = self._load_profiles_config()
        self.quality_config = self._load_quality_config()

        # Инициализация NER (GLiNER2)
        self.ner_enabled = config.get('enable_ner', True) and GLINER_AVAILABLE
        self.ner_model_name = config.get('ner_model', 'fastino/gliner2-base-v1')
        
        if self.ner_enabled:
            try:
                self.gliner_model = GLiNER2.from_pretrained(self.ner_model_name)
                logger.info(f"🧠 GLiNER2 NER загружен (модель: {self.ner_model_name})")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки GLiNER2: {e}")
                self.ner_enabled = False

        # Динамический маппинг меток под разные миры (из profiles.yaml)
        self.ner_competencies = self.profiles_config.get('profiles', {})
        self.base_mapping = self.profiles_config.get('base_mapping', {})
        # Regex-паттерны для fallback (из profiles.yaml)
        self.regex_patterns = self._prepare_regex_patterns(self.profiles_config.get('regex_patterns', {}))

        # Загружаем списки доменов и мусорных фраз из quality.yaml
        self.priority_domains = set(self.quality_config.get('priority_domains', []))
        self.low_trust_domains = set(self.quality_config.get('low_trust_domains', []))
        self.junk_phrases = self.quality_config.get('junk_phrases', [])
        self.ad_indicators = self.quality_config.get('ad_indicators', [])

        self.ranking_service = RankingService()
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'errors': 0,
            'ner_used': 0
        }

        logger.info("📚 KnowledgeAnalyst создан, все конфиги загружены из YAML")

    def _load_profiles_config(self) -> dict:
        """Загружает конфигурацию профилей (NER-метки и regex) из YAML-файла в корневой configs."""
        default_config = {
            'profiles': {},
            'base_mapping': {
                'person': 10.0, 'PER': 10.0,
                'location': 8.0, 'LOC': 8.0,
                'organization': 8.0, 'ORG': 8.0,
                'event': 7.0, 'date': 5.0, 'DATE': 5.0
            },
            'regex_patterns': {}
        }
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        config_path = os.path.join(base_dir, 'configs', 'profiles.yaml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                if user_config:
                    if 'profiles' in user_config:
                        default_config['profiles'] = user_config['profiles']
                    if 'base_mapping' in user_config:
                        default_config['base_mapping'] = user_config['base_mapping']
                    if 'regex_patterns' in user_config:
                        default_config['regex_patterns'] = user_config['regex_patterns']
                logger.info(f"📁 Загружена конфигурация профилей из {config_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки profiles.yaml: {e}, используются defaults")
        else:
            logger.warning(f"⚠️ {config_path} не найден, используются значения по умолчанию")
        return default_config

    def _load_quality_config(self) -> dict:
        """Загружает конфигурацию качества (домены, мусорные фразы) из quality.yaml в корневой configs."""
        default_config = {
            'priority_domains': [],
            'low_trust_domains': [],
            'ad_indicators': [],
            'junk_phrases': []
        }
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        config_path = os.path.join(base_dir, 'configs', 'quality.yaml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                if user_config:
                    for key in default_config:
                        if key in user_config:
                            default_config[key] = user_config[key]
                logger.info(f"📁 Загружена конфигурация качества из {config_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки quality.yaml: {e}")
        else:
            logger.warning(f"⚠️ {config_path} не найден, используются пустые списки")
        return default_config

    def _prepare_regex_patterns(self, regex_dict: dict) -> List[Tuple[str, str]]:
        """Преобразует словарь regex_patterns в список кортежей (profile, pattern)."""
        patterns = []
        for profile, pattern_list in regex_dict.items():
            if isinstance(pattern_list, list):
                combined = '|'.join(pattern_list)
            else:
                combined = pattern_list
            patterns.append((profile, combined))
        patterns.append(('default', '.*'))
        return patterns

    async def initialize(self):
        """Асинхронная инициализация."""
        return True

    # ----------------------------------------------------------------------
    # ОПРЕДЕЛЕНИЕ ПРОФИЛЯ (NER + regex)
    # ----------------------------------------------------------------------
    async def _get_query_ner_profile(self, query: str) -> Optional[str]:
        """Определяет профиль по NER-меткам самого запроса."""
        if not self.ner_enabled:
            return None
        try:
            all_labels = list(self.base_mapping.keys())
            for prof in self.ner_competencies.values():
                all_labels.extend(prof['labels'])
            all_labels = list(set(all_labels))

            entities = await asyncio.to_thread(
                self.gliner_model.extract_entities, query, all_labels, threshold=0.5
            )

            profile_scores = {p: 0.0 for p in self.ner_competencies}
            for ent in entities:
                label = ent['label']
                score = ent['score']
                for prof_name, comp in self.ner_competencies.items():
                    if label in comp['labels']:
                        profile_scores[prof_name] += comp['weight'] * score
            
            if profile_scores:
                best = max(profile_scores, key=profile_scores.get)
                if profile_scores[best] > 5.0:
                    logger.debug(f"NER запроса выбрал профиль {best} со счётом {profile_scores[best]:.1f}")
                    return best
        except Exception as e:
            logger.debug(f"NER на запросе не удался: {e}")
        return None

    def _detect_query_profile_regex(self, query: str) -> str:
        """Определяет тип вопроса по регулярным выражениям (из загруженного конфига)."""
        q = query.lower()
        for profile, pattern in self.regex_patterns:
            if re.search(pattern, q):
                return profile
        return 'default'

    async def detect_profile(self, query: str) -> str:
        """Определяет профиль по NER, затем по regex."""
        profile = await self._get_query_ner_profile(query)
        if profile is not None:
            return profile
        return self._detect_query_profile_regex(query)

    # ----------------------------------------------------------------------
    # ОСНОВНОЙ МЕТОД АНАЛИЗА
    # ----------------------------------------------------------------------
    async def analyze(self, documents: List[Dict], query: str = "", is_discovery: bool = False) -> Dict[str, Any]:
        """
        Анализ документов: извлечение чанков, фактов, ранжирование, дедупликация.
        Возвращает структуру с ключами: success, summary, key_points, key_facts_metadata, profile, query, confidence, ...
        Количество key_points динамическое: от 2 до 20, в зависимости от качества.
        """
        start_time = datetime.now()
        profile_name = await self.detect_profile(query)
        logger.info(f"🔍 Определён профиль запроса: {profile_name}")

        try:
            # 1. Извлекаем чанки из документов
            all_chunks = []
            for doc in documents:
                doc_result = await self._process_document(doc, is_discovery=is_discovery)
                all_chunks.extend(doc_result.get('chunks', []))

            unique_chunks = self._deduplicate_chunks(all_chunks)
            logger.info(f"   ✅ Уникальных чанков: {len(unique_chunks)}")

            # 2. Извлекаем факты
            fact_objects = await self._extract_key_facts(
                chunks=unique_chunks,
                query=query,
                top_k=50,
                is_discovery=is_discovery
            )
            if fact_objects is None:
                fact_objects = []

            logger.info(f"   ✅ Извлечено фактов до дедупликации: {len(fact_objects)}")

            # 3. Семантическая дедупликация (если не discovery)
            if not is_discovery and len(fact_objects) > 10 and SENTENCE_TRANSFORMERS_AVAILABLE:
                fact_objects = await self._semantic_deduplication(fact_objects, threshold=0.85)
                logger.info(f"   ✅ После семантической дедупликации: {len(fact_objects)} фактов")

            # 4. Ранжирование и динамический отбор
            top_facts = []
            key_points = []
            if fact_objects:
                facts_for_ranking = []
                for fo in fact_objects:
                    fact = Fact(
                        text=fo['text'],
                        source_domain=fo.get('domain', ''),
                        position_ratio=fo.get('position_ratio', 0.5),
                        ner_score=fo.get('ner_score', 0.0),
                        ner_types=fo.get('ner_types', []),
                        length=fo.get('length', len(fo['text'])),
                        contains_definition=fo.get('contains_definition', False),
                        contains_causal=fo.get('contains_causal', False)
                    )
                    facts_for_ranking.append(fact)

                query_emb = await self._get_query_embedding(query)
                ranked = await self.ranking_service.rank(
                    query=query,
                    facts=facts_for_ranking,
                    query_embedding=query_emb,
                    priority_domains=self.priority_domains,
                    low_trust_domains=self.low_trust_domains,
                    profile_name=profile_name
                )
                logger.info(f"   ✅ Ранжирование завершено, фактов после ранга: {len(ranked)}")

                if ranked:
                    scores = [score for _, score in ranked]
                    max_score = max(scores)
                    threshold = max_score * 0.6  # порог 60%

                    filtered = [(fact, score) for fact, score in ranked if score >= threshold]
                    if len(filtered) < 2:
                        filtered = ranked[:2]
                    if len(filtered) > 20:
                        filtered = filtered[:20]

                    top_facts = [fact for fact, _ in filtered]
                    key_points = [fact.text for fact in top_facts]

                    logger.info(f"   🔍 Динамический отбор: исходно {len(ranked)}, после порога {len(filtered)}")
                    if filtered:
                        logger.info(f"   🔍 Пример факта: {filtered[0][0].text[:100]}... (оценка {filtered[0][1]:.2f})")
                else:
                    logger.info("   🔍 ranked пуст")

                self._last_facts_metadata = top_facts[:15] if top_facts else []
            else:
                logger.info("   🔍 Нет фактов для ранжирования")

            # 5. Суммаризация
            summary = ""
            if self.enable_summarization and unique_chunks:
                best_chunk = max(unique_chunks, key=lambda x: x.get('quality_score', 0))
                if best_chunk and best_chunk.get('text'):
                    summary = self._generate_extractive_summary(best_chunk['text'], sentences_count=3)

            # Уверенность
            confidence = self._calculate_confidence(unique_chunks, fact_objects, query)

            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['documents_processed'] += len(documents)
            self.stats['chunks_created'] += len(unique_chunks)

            logger.info(f"   🔍 key_points перед возвратом: {len(key_points)}")

            return {
                'success': True,
                'documents_count': len(documents),
                'summary': summary,
                'key_points': key_points,
                'key_facts_metadata': top_facts[:15] if top_facts else [],
                'profile': profile_name,
                'query': query,
                'confidence': confidence,
                'processing_time': processing_time
            }
        except Exception as e:
            logger.error(f"Ошибка анализа: {e}", exc_info=True)
            self.stats['errors'] += 1
            return {
                'success': False,
                'error': str(e),
                'summary': '',
                'key_points': [],
                'profile': profile_name,
                'query': query
            }

    # ----------------------------------------------------------------------
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ----------------------------------------------------------------------
    async def _extract_key_facts(self, chunks, query, top_k=50, is_discovery=False):
        """Извлекает факты из чанков с улучшенной фильтрацией и оценкой."""
        if not chunks:
            return []
        
        # Ключевые слова из запроса
        keywords = [w.lower() for w in re.findall(r'\b\w{4,}\b', query) 
                    if w.lower() not in {'когда','что','как','где','почему','зачем','какой','какая','какие','кто'}]
        
        all_facts = []
        junk_phrases = self.junk_phrases
        ad_indicators = self.ad_indicators
        
        for chunk in chunks:
            text = chunk.get('text', '')
            if not text:
                continue
            
            sentences = re.split(r'(?<=[.!?])\s+', text)
            source_url = chunk.get('source_url', '')
            domain = self._extract_domain(source_url)
            position_ratio = chunk.get('position_ratio', 0.5)
            
            for sent in sentences:
                sent = sent.strip()
                sent_lower = sent.lower()
                
                # Длина
                if len(sent) < 40 or len(sent) > 600:
                    continue
                
                # Мусорные фразы
                if any(phrase in sent_lower for phrase in junk_phrases):
                    continue
                if any(ad in sent_lower for ad in ad_indicators):
                    continue
                if re.search(r'https?://|www\.', sent):
                    continue
                if sent.count('!') > 2:
                    continue
                
                # Доля заглавных букв
                words = sent.split()
                if len(words) >= 3:
                    capitalized = sum(1 for w in words[1:] if w and w[0].isupper())
                    if capitalized / max(len(words)-1, 1) > 0.4:
                        continue
                
                # Доля стоп-слов
                stop_words = {'в', 'на', 'с', 'со', 'к', 'по', 'из', 'за', 'у', 'от', 'до', 'для', 'о', 'об', 'под', 'над', 'перед', 'через', 'и', 'а', 'но', 'да', 'или', 'либо', 'то', 'как', 'так', 'что', 'чтобы', 'если', 'потому', 'поэтому'}
                word_count = len(words)
                if word_count > 0:
                    stop_count = sum(1 for w in words if w.lower() in stop_words)
                    stop_ratio = stop_count / word_count
                    if stop_ratio > 0.6:
                        continue
                
                # Проверка на вопросительные заголовки
                question_words = {'когда', 'где', 'почему', 'зачем', 'как', 'что', 'кто'}
                first_word = sent_lower.split()[0] if sent_lower.split() else ''
                if first_word in question_words and len(sent) < 100:
                    if any(kw in sent_lower for kw in keywords):
                        continue
                
                # Базовая оценка
                base_score = 0.0
                for kw in keywords:
                    if kw in sent_lower:
                        base_score += 1.0
                
                position_bonus = 1.0 - position_ratio
                length_score = 0.5 if 80 <= len(sent) <= 250 else 0.3 if 250 < len(sent) <= 400 else 0.0
                digit_bonus = 0.5 if re.search(r'\b\d+\b', sent) else 0.0
                date_bonus = 0.5 if re.search(r'\b\d{4}\b', sent) or re.search(r'\bмлн лет\b', sent_lower) else 0.0
                def_bonus = 0.3 if re.search(r'—| это | является |определяет', sent_lower) else 0.0
                
                total_score = base_score + position_bonus + length_score + digit_bonus + date_bonus + def_bonus

                # Бонусы за домены
                if domain in self.priority_domains:
                    total_score *= 1.2
                elif domain in self.low_trust_domains:
                    total_score *= 0.5
                
                all_facts.append({
                    'text': sent,
                    'domain': domain,
                    'source_url': source_url,
                    'position_ratio': position_ratio,
                    'ner_score': 0.0,
                    'ner_types': [],
                    'length': len(sent),
                    'contains_definition': bool(re.search(r'—| это | является |определяет', sent_lower)),
                    'contains_causal': bool(re.search(r'потому что|так как|следовательно|поэтому|из-за|вследствие', sent_lower)),
                    'base_score': base_score,
                    'position_bonus': position_bonus,
                    'digit_bonus': digit_bonus,
                    'date_bonus': date_bonus,
                    'def_bonus': def_bonus,
                    'total_score': total_score,
                    'chunk_id': ''
                })
        
        unique = self._deduplicate_facts_by_text(all_facts)
        unique.sort(key=lambda x: x['total_score'], reverse=True)
        return unique[:top_k]

    async def _semantic_deduplication(self, facts: List[Dict], threshold: float = 0.85) -> List[Dict]:
        if len(facts) < 2:
            return facts
        try:
            from appp.services.embedding.bge_m3 import embedder
            texts = [f['text'] for f in facts]
            embeddings = await embedder.embed(texts)
            if not embeddings or len(embeddings) != len(facts):
                return facts
            unique_facts = []
            unique_embeddings = []
            for i, (fact, emb) in enumerate(zip(facts, embeddings)):
                is_duplicate = False
                for unique_emb in unique_embeddings:
                    try:
                        similarity = util.cos_sim(emb, unique_emb).item()
                        if similarity > threshold:
                            is_duplicate = True
                            break
                    except:
                        pass
                if not is_duplicate:
                    unique_facts.append(fact)
                    unique_embeddings.append(emb)
            return unique_facts
        except Exception as e:
            logger.warning(f"Ошибка при семантической дедупликации: {e}")
            return facts

    def _extract_domain(self, url: str) -> str:
        from urllib.parse import urlparse
        try:
            return urlparse(url).netloc.lower()
        except:
            return ''

    def _deduplicate_facts_by_text(self, facts: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for f in facts:
            sig = f['text'][:100].lower()
            if sig not in seen:
                seen.add(sig)
                unique.append(f)
        return unique

    async def _get_query_embedding(self, query: str):
        try:
            if embedder is None or embedder.model is None:
                return None
            return await asyncio.wait_for(embedder.embed(query), timeout=10.0)
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга запроса: {e}")
            return None

    def _generate_extractive_summary(self, text: str, sentences_count: int = 3) -> str:
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lsa import LsaSummarizer
            if not text or len(text) < 200:
                return ""
            parser = PlaintextParser.from_string(text, Tokenizer("russian"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count)
            return " ".join(str(sentence) for sentence in summary)
        except ImportError:
            return ""
        except Exception as e:
            logger.error(f"Ошибка суммаризации: {e}")
            return ""

    def _calculate_confidence(self, chunks, facts, query):
        if not chunks:
            return 0.0
        avg_chunk_quality = sum(c.get('quality_score', 0) for c in chunks) / len(chunks)
        domain_scores = []
        for chunk in chunks:
            url = chunk.get('source_url', '')
            domain = self._extract_domain(url)
            if domain in self.priority_domains:
                domain_scores.append(1.0)
            elif domain in self.low_trust_domains:
                domain_scores.append(0.3)
            else:
                domain_scores.append(0.6)
        avg_domain_trust = sum(domain_scores) / len(domain_scores) if domain_scores else 0.5
        unique_sources = set(chunk.get('source_url', '') for chunk in chunks if chunk.get('source_url'))
        source_count_score = min(len(unique_sources) / 5.0, 1.0)
        consistency_score = 0.5
        if facts and len(facts) >= 3:
            all_text = " ".join([f.get('text', '') for f in facts[:10]])
            words = re.findall(r'\b\w{4,}\b', all_text.lower())
            from collections import Counter
            word_counts = Counter(words)
            common_words = [w for w, c in word_counts.most_common(5) if c >= 3]
            consistency_score = min(len(common_words) / 3.0, 1.0) if common_words else 0.4
        ner_score = 0.0
        if facts:
            ner_count = sum(1 for f in facts if f.get('ner_types'))
            ner_score = min(ner_count / max(len(facts), 1) * 2, 1.0)
        confidence = (
            avg_chunk_quality * 0.3 +
            avg_domain_trust * 0.25 +
            source_count_score * 0.2 +
            consistency_score * 0.15 +
            ner_score * 0.1
        )
        if len(unique_sources) == 1:
            confidence *= 0.8
        elif len(unique_sources) == 0:
            confidence *= 0.5
        return min(confidence, 1.0)

    async def _process_document(self, doc: Dict, is_discovery: bool = False) -> Dict:
        content = doc.get('content', '')
        if not content:
            return {'chunks': []}
        cleaned_content = self.text_cleaner.clean(content)
        chunks = self._split_into_chunks(cleaned_content, doc.get('url', ''))
        total_chunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk['chunk_index'] = idx
            chunk['total_chunks'] = total_chunks
            chunk['quality_score'] = self._evaluate_chunk_quality(chunk['text'])
        quality_threshold = 0.2 if is_discovery else 0.5
        chunks = [c for c in chunks if c['quality_score'] >= quality_threshold]
        max_chunks = self.max_chunks_per_document * 2 if is_discovery else self.max_chunks_per_document
        chunks = chunks[:max_chunks]
        return {'chunks': chunks}

    def _split_into_chunks(self, text: str, source_url: str) -> List[Dict]:
        if not text:
            return []
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        for para in paragraphs:
            para_len = len(para)
            if para_len >= self.chunk_size * 0.7:
                if current_chunk:
                    chunks.append(self._create_chunk('\n\n'.join(current_chunk), source_url))
                    current_chunk = []
                    current_length = 0
                chunks.append(self._create_chunk(para, source_url))
            else:
                if current_length + para_len <= self.chunk_size:
                    current_chunk.append(para)
                    current_length += para_len
                else:
                    chunks.append(self._create_chunk('\n\n'.join(current_chunk), source_url))
                    current_chunk = [para]
                    current_length = para_len
        if current_chunk:
            chunks.append(self._create_chunk('\n\n'.join(current_chunk), source_url))
        return chunks

    def _create_chunk(self, text: str, source_url: str) -> Dict:
        chunk_id = hashlib.md5(f"{source_url}_{text[:100]}".encode()).hexdigest()[:12]
        return {
            'chunk_id': chunk_id,
            'text': text,
            'source_url': source_url,
            'length': len(text),
            'created_at': datetime.now().isoformat()
        }

    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for chunk in chunks:
            sig = chunk['text'][:200]
            if sig not in seen:
                seen.add(sig)
                unique.append(chunk)
        return unique

    def _evaluate_chunk_quality(self, text: str) -> float:
        length = len(text)
        if length >= 800:
            return 0.8
        elif length >= 500:
            return 0.6
        elif length >= 300:
            return 0.4
        else:
            return 0.2

    async def get_analyst_stats(self) -> Dict:
        return self.stats

    async def health_check(self) -> Dict:
        return {
            'healthy': True,
            'message': 'KnowledgeAnalyst is operational',
            'timestamp': datetime.now().isoformat()
        }