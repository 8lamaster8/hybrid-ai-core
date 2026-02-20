"""
📚 АНАЛИТИК ЗНАНИЙ — извлечение фактов с метаданными + ранжирование + шаблоны
"""

import asyncio
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from appp.core.logging import logger
from appp.utils.text_processor import TextCleaner
from appp.services.ranking.ranking_service import RankingService, Fact

# Для NER (GLiNER2)
try:
    from gliner2 import GLiNER2
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    logger.warning("⚠️ GLiNER2 не установлена, NER отключён")

# Для косинусной близости
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

        # Динамический маппинг меток под разные миры (Квантмех, Философия и т.д.)
        # GLiNER2 лучше понимает английские ключи для поиска в русском тексте
        self.ner_competencies = {
            'scientific_concept': {
                'labels': ["Scientific Law", "Quantum Phenomenon", "Hypothesis", "Chemical Compound", "Scientist"],
                'weight': 15.0  # Наука в научном запросе — приоритет №1
            },
            'quantum_physics_deep': {
                'labels': ["Quantum Phenomenon", "Scientific Law", "Physicist"],
                'weight': 14.0
            },
            'philosophical_concept': {
                'labels': ["Philosophical Doctrine", "Subjective Experience", "Thinker", "Ontological Term"],
                'weight': 15.0
            },
            'mathematical_theorem': {
                'labels': ["Mathematical Theorem", "Axiom", "Mathematical Notation", "Formula"],
                'weight': 14.0
            },
            'biological_system': {
                'labels': ["Biological Mechanism", "Anatomical Structure", "Process", "Species"],
                'weight': 13.0
            },
            'programming_concept': {
                'labels': ["Programming Language", "Algorithm", "Framework", "Library"],
                'weight': 11.0
            }
        }

        # Базовый маппинг для общегражданских сущностей
        self.base_mapping = {
            'person': 10.0, 'PER': 10.0,
            'location': 8.0, 'LOC': 8.0,
            'organization': 8.0, 'ORG': 8.0,
            'event': 7.0, 'date': 5.0
        }

        # Загружаем списки доменов из конфига комитета
        self.priority_domains = set(config.get('priority_domains', []))
        self.low_trust_domains = set(config.get('low_trust_domains', []))

        self.ranking_service = RankingService()
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'errors': 0,
            'ner_used': 0
        }
        logger.info("📚 KnowledgeAnalyst создан с динамическим NER")

    async def initialize(self):
        return True

    # ----------------------------------------------------------------------
    # ОСНОВНОЙ МЕТОД АНАЛИЗА (с глобальной дедупликацией)
    # ----------------------------------------------------------------------
    async def analyze(self, documents: List[Dict], query: str = "", is_discovery: bool = False) -> Dict[str, Any]:
        """
        Анализ документов:
        - извлечение чанков
        - выделение фактов с метаданными
        - СЕМАНТИЧЕСКАЯ ДЕДУПЛИКАЦИЯ (кроме discovery)
        - ранжирование через RankingService
        - опциональная суммаризация
        - улучшенный расчёт уверенности
        """
        start_time = datetime.now()

        unique_chunks = []
        fact_objects = []
        ranked = []
        key_points = []
        summary = ""
        top_facts = []
        profile_name = self._detect_query_profile(query)

        try:
            # 1. Извлекаем чанки из документов
            all_chunks = []
            for doc in documents:
                doc_result = await self._process_document(doc)
                all_chunks.extend(doc_result.get('chunks', []))

            unique_chunks = self._deduplicate_chunks(all_chunks)
            logger.info(f"   ✅ Уникальных чанков: {len(unique_chunks)}")

            # 2. Извлекаем факты (передаём is_discovery)
            fact_objects = await self._extract_key_facts(
                chunks=unique_chunks,
                query=query,
                top_k=50,
                is_discovery=is_discovery
            )
            logger.info(f"   ✅ Извлечено фактов ДО дедупликации: {len(fact_objects)}")

            # 3. Семантическая дедупликация (только если не discovery)
            if not is_discovery and len(fact_objects) > 10 and SENTENCE_TRANSFORMERS_AVAILABLE:
                fact_objects = await self._semantic_deduplication(fact_objects, threshold=0.85)
                logger.info(f"   ✅ После семантической дедупликации: {len(fact_objects)} фактов")

            # 4. Ранжирование (всегда)
            if fact_objects:
                profile_name = self._detect_query_profile(query)
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

                top_facts = [fact for fact, score in ranked[:15]]
                key_points = [fact.text for fact in top_facts]
                self._last_facts_metadata = top_facts[:15] if top_facts else []

            # 5. Суммаризация
            if self.enable_summarization and unique_chunks:
                best_chunk = max(unique_chunks, key=lambda x: x.get('quality_score', 0))
                if best_chunk and best_chunk.get('text'):
                    summary = self._generate_extractive_summary(best_chunk['text'], sentences_count=3)

            # Улучшенный расчёт уверенности
            confidence = self._calculate_confidence(unique_chunks, fact_objects, query)

            processing_time = (datetime.now() - start_time).total_seconds()

            self.stats['documents_processed'] += len(documents)
            self.stats['chunks_created'] += len(unique_chunks)

            return {
                'success': True,
                'documents_count': len(documents),
                'summary': summary,
                'key_points': key_points[:15],
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
                'key_points': []
            }

    # ----------------------------------------------------------------------
    # СЕМАНТИЧЕСКАЯ ДЕДУПЛИКАЦИЯ
    # ----------------------------------------------------------------------
    async def _semantic_deduplication(self, facts: List[Dict], threshold: float = 0.85) -> List[Dict]:
        """Удаляет семантические дубликаты (похожие по смыслу) из списка фактов."""
        if len(facts) < 2:
            return facts
        
        try:
            from appp.services.embedding.bge_m3 import embedder
            
            texts = [f['text'] for f in facts]
            embeddings = await embedder.embed(texts)
            
            if not embeddings or len(embeddings) != len(facts):
                logger.warning("⚠️ Не удалось получить эмбеддинги для дедупликации")
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
            
            logger.debug(f"   Семантическая дедупликация: {len(facts)} -> {len(unique_facts)}")
            return unique_facts
            
        except Exception as e:
            logger.warning(f"Ошибка при семантической дедупликации: {e}")
            return facts

    # ----------------------------------------------------------------------
    # ИЗВЛЕЧЕНИЕ ФАКТОВ С МЕТАДАННЫМИ (С ЖЁСТКОЙ ФИЛЬТРАЦИЕЙ)
    # ----------------------------------------------------------------------
    async def _extract_key_facts(
        self,
        chunks: List[Dict],
        query: str,
        top_k: int = 50,
        is_discovery: bool = False
    ) -> List[Dict]:
        """Извлекает факты из чанков и обогащает их метаданными."""
        if not chunks:
            return []

        # Ключевые слова из запроса (для базового скоринга)
        keywords = [w.lower() for w in re.findall(r'\b\w{4,}\b', query)
                    if w.lower() not in {'когда','что','как','где','почему','зачем',
                                        'какой','какая','какие','кто'}]

        all_facts = []
        junk_phrases = self._get_junk_phrases()

        for chunk in chunks:
            text = chunk.get('text', '')
            if not text:
                continue

            sentences = re.split(r'(?<=[.!?])\s+', text)
            source_url = chunk.get('source_url', '')
            domain = self._extract_domain(source_url)

            chunk_index = chunk.get('chunk_index', 0)
            total_chunks = chunk.get('total_chunks', 1)
            position_ratio = chunk_index / max(total_chunks, 1)

            for sent in sentences:
                sent = sent.strip()
                sent_lower = sent.lower()
                
                # --- Фильтрация ---
                if is_discovery:
                    # Мягкие условия для discovery
                    if len(sent) < 20:
                        continue
                    if re.search(r'[{}\[\]<>]', sent) and len(re.findall(r'[{}\[\]<>]', sent)) > 5:
                        continue
                    if any(phrase in sent_lower for phrase in junk_phrases):
                        continue
                    if re.search(r'https?://|www\.', sent):
                        continue
                else:
                    # Жёсткие условия для обычных вопросов
                    if len(sent) < 40:
                        continue
                    if re.search(r'[{}\[\]<>]', sent) and len(re.findall(r'[{}\[\]<>]', sent)) > 3:
                        continue
                    alpha_ratio = sum(c.isalpha() for c in sent) / len(sent)
                    if alpha_ratio < 0.5:
                        continue
                    if sent[0].islower() and len(sent) < 100:
                        if not any(word in sent_lower for word in ['является', 'был', 'была', 'были', 'есть', 'имеет', 'можно', 'нужно']):
                            continue
                    words = sent.split()
                    if len(words) < 5:
                        continue
                    long_words = [w for w in words if len(w) > 3]
                    if len(long_words) < 2:
                        continue
                    if any(phrase in sent_lower for phrase in junk_phrases):
                        continue
                    if sent.startswith(('[[', ']]', '{{', '}}', '==', '*', '#', '|', ';', ':', '^')):
                        continue
                    if re.search(r'https?://|www\.', sent):
                        continue
                    if re.match(r'^[А-ЯA-Z][^.]*:', sent):
                        continue
                    if re.match(r'^[^—]{1,30} —', sent):
                        continue
                    if re.match(r'^[IVX]+\.|^[A-ZА-Я]\.', sent):
                        continue
                    if len(words) >= 3:
                        capitalized = sum(1 for w in words[1:] if w and w[0].isupper())
                        if capitalized / max(len(words)-1, 1) > 0.5:
                            continue
                    if sent[-1] not in {'.', '!', '?'}:
                        continue

                # --- Вычисляем признаки (одинаково для обоих режимов) ---
                ner_score, ner_types = await self._compute_ner_features(sent, query)
                length = len(sent)
                contains_definition = bool(re.search(r'—| это | является |определяет', sent))
                contains_causal = bool(re.search(r'потому что|так как|следовательно|поэтому|из-за|вследствие', sent_lower))

                base_score = 0.0
                if re.search(r'\b\d{4}\b', sent):
                    base_score += 3.0
                if re.search(r'\b\d+\b', sent):
                    base_score += 1.0
                for kw in keywords:
                    if kw in sent_lower:
                        base_score += 1.0

                total_score = base_score + ner_score

                fact_entry = {
                    'text': sent,
                    'domain': domain,
                    'source_url': source_url,
                    'position_ratio': position_ratio,
                    'ner_score': ner_score,
                    'ner_types': ner_types,
                    'length': length,
                    'contains_definition': contains_definition,
                    'contains_causal': contains_causal,
                    'base_score': base_score,
                    'total_score': total_score,
                    'chunk_id': chunk.get('chunk_id', '')
                }
                all_facts.append(fact_entry)

        # Дедупликация по тексту (всегда)
        unique_facts = self._deduplicate_facts_by_text(all_facts)

        # Сортировка по total_score
        unique_facts.sort(key=lambda x: x['total_score'], reverse=True)

        return unique_facts[:top_k]

    # ----------------------------------------------------------------------
    # ПОДГОТОВКА ДАННЫХ ДЛЯ ШАБЛОНОВ С ДЕДУПЛИКАЦИЕЙ
    # ----------------------------------------------------------------------
    def _prepare_template_data(self, profile: str, facts_metadata: List[Fact], query: str) -> dict:
        """Извлекает из фактов данные для конкретного шаблона с жёсткой дедупликацией между полями."""
        data = {
            'query': query,
            'default_answer': '\n'.join([f.text for f in facts_metadata[:10]])
        }
        
        # Словарь для отслеживания уже использованных текстов
        used_texts = set()
        
        def get_unique_facts(facts_list: List[Fact], max_count: int, seen_set: set) -> List[str]:
            """Возвращает уникальные факты, которых ещё не было в seen_set"""
            unique = []
            for fact in facts_list:
                norm_text = ' '.join(fact.text.lower().split())
                
                is_duplicate = False
                for used in seen_set:
                    if len(norm_text) > 50 and len(used) > 50:
                        shorter = min(len(norm_text), len(used))
                        longer = max(len(norm_text), len(used))
                        if shorter / longer > 0.7:
                            words1 = set(norm_text.split())
                            words2 = set(used.split())
                            intersection = words1 & words2
                            if len(intersection) / max(len(words1), len(words2)) > 0.6:
                                is_duplicate = True
                                break
                
                if not is_duplicate and len(unique) < max_count:
                    unique.append(fact.text)
                    seen_set.add(norm_text)
            return unique

        if profile == 'mathematical_theorem':
            data['theorem_name'] = query.strip().rstrip('?').replace('теорема', '').strip()
            
            definitions = get_unique_facts(
                [f for f in facts_metadata if f.contains_definition], 3, used_texts
            )
            data['statement'] = '\n'.join(definitions) if definitions else 'Формулировка не найдена.'
            
            named_facts = get_unique_facts(
                [f for f in facts_metadata if 'PER' in f.ner_types or 'DATE' in f.ner_types], 2, used_texts
            )
            data['historical_context'] = ' '.join(named_facts) if named_facts else 'Исторический контекст не найден.'
            
            formula_facts = get_unique_facts(
                [f for f in facts_metadata if re.search(r'[=+\-*/^(){}]', f.text)], 1, used_texts
            )
            data['formulation'] = formula_facts[0] if formula_facts else ''
            
            proof_facts = get_unique_facts(facts_metadata, 1, used_texts)
            data['proof_summary'] = proof_facts[0] if proof_facts else ''
            
            applications = get_unique_facts(facts_metadata, 2, used_texts)
            data['applications'] = '\n'.join(applications) if applications else ''
            
            related = get_unique_facts(facts_metadata, 2, used_texts)
            data['related_concepts'] = '\n'.join(related) if related else ''

        elif profile == 'historical_event':
            data['event_name'] = query.strip().rstrip('?')
            
            dates_facts = get_unique_facts(
                [f for f in facts_metadata if 'DATE' in f.ner_types], 5, used_texts
            )
            data['timeline'] = '\n'.join(dates_facts) if dates_facts else 'Хронология не найдена.'
            data['key_dates'] = '\n'.join(dates_facts[:3]) if dates_facts else 'Не указаны.'
            
            people_facts = get_unique_facts(
                [f for f in facts_metadata if 'PER' in f.ner_types], 5, used_texts
            )
            data['key_figures'] = '\n'.join(people_facts) if people_facts else 'Участники не указаны.'
            
            causes_facts = get_unique_facts(
                [f for f in facts_metadata if f.contains_causal], 3, used_texts
            )
            data['causes'] = '\n'.join(causes_facts) if causes_facts else 'Причины не указаны.'
            
            other_facts = get_unique_facts(
                [f for f in facts_metadata if not ('DATE' in f.ner_types or 'PER' in f.ner_types or f.contains_causal)], 5, used_texts
            )
            data['consequences'] = other_facts[0] if other_facts else 'Последствия не указаны.'
            data['interesting_facts'] = '\n'.join(other_facts[1:4]) if len(other_facts) > 1 else ''

        elif profile == 'programming_concept':
            data['concept_name'] = query.strip().rstrip('?')
            
            lang = 'python'
            for f in facts_metadata:
                if 'python' in f.text.lower():
                    lang = 'python'
                elif 'java' in f.text.lower():
                    lang = 'java'
                elif 'c++' in f.text.lower() or 'cpp' in f.text.lower():
                    lang = 'cpp'
            data['language'] = lang
            
            def_fact = get_unique_facts(
                [f for f in facts_metadata if f.contains_definition], 1, used_texts
            )
            data['definition'] = def_fact[0] if def_fact else 'Определение не найдено.'
            
            code_candidates = get_unique_facts(
                [f for f in facts_metadata if re.search(r'[=;{}\[\]()]', f.text)], 2, used_texts
            )
            data['syntax_example'] = code_candidates[0] if code_candidates else 'Пример не найден.'
            data['practical_example'] = code_candidates[1] if len(code_candidates) > 1 else data['syntax_example']
            
            remaining = get_unique_facts(facts_metadata, 4, used_texts)
            data['use_cases'] = remaining[0] if len(remaining) > 0 else ''
            data['advantages'] = remaining[1] if len(remaining) > 1 else ''
            data['disadvantages'] = remaining[2] if len(remaining) > 2 else ''
            data['alternatives'] = remaining[3] if len(remaining) > 3 else ''

        elif profile == 'scientific_concept':
            data['concept_name'] = query.strip().rstrip('?')
            
            def_facts = get_unique_facts(
                [f for f in facts_metadata if f.contains_definition], 2, used_texts
            )
            data['scientific_definition'] = '\n'.join(def_facts) if def_facts else 'Определение не найдено.'
            
            principles = get_unique_facts(
                [f for f in facts_metadata if f.contains_causal or 'PER' in f.ner_types], 3, used_texts
            )
            data['principles'] = '\n'.join(principles) if principles else 'Принципы не указаны.'
            
            formula_facts = get_unique_facts(
                [f for f in facts_metadata if re.search(r'[=+\-*/^(){}]', f.text)], 2, used_texts
            )
            data['mathematical_description'] = '\n'.join(formula_facts) if formula_facts else 'Не найдено.'
            
            exp_facts = get_unique_facts(
                [f for f in facts_metadata if 'DATE' in f.ner_types or 'PERCENT' in f.ner_types], 2, used_texts
            )
            data['experimental_evidence'] = '\n'.join(exp_facts) if exp_facts else 'Не указаны.'
            
            remaining = get_unique_facts(facts_metadata, 4, used_texts)
            data['application_domains'] = remaining[0] if len(remaining) > 0 else ''
            data['current_state'] = remaining[1] if len(remaining) > 1 else ''

        elif profile == 'factoid':
            short_answer_facts = get_unique_facts(facts_metadata, 1, used_texts)
            data['short_answer'] = short_answer_facts[0] if short_answer_facts else 'Информация не найдена.'
            
            bullet_facts = get_unique_facts(facts_metadata, 10, used_texts)
            data['bullet_points'] = bullet_facts
            
            sources = set()
            for f in facts_metadata[:15]:
                if f.source_domain:
                    sources.add(f.source_domain)
            data['sources'] = list(sources)[:5]

        elif profile == 'how_why':
            causal_facts = get_unique_facts(
                [f for f in facts_metadata if f.contains_causal], 5, used_texts
            )
            data['explanations'] = '\n'.join(causal_facts) if causal_facts else 'Объяснения не найдены.'
            
            mechanism_keywords = ['работает', 'процесс', 'этап', 'шаг', 'функционирует']
            mechanism = get_unique_facts(
                [f for f in facts_metadata if any(kw in f.text.lower() for kw in mechanism_keywords)], 3, used_texts
            )
            data['mechanism'] = '\n'.join(mechanism) if mechanism else 'Механизм не описан.'
            
            factors = get_unique_facts(
                [f for f in facts_metadata if 'влия' in f.text.lower() or 'фактор' in f.text.lower() or 'причин' in f.text.lower()], 3, used_texts
            )
            data['factors'] = '\n'.join(factors) if factors else 'Факторы не указаны.'
            
            other = get_unique_facts(facts_metadata, 3, used_texts)
            data['additional_info'] = '\n'.join(other)

        elif profile == 'evaluation':
            comparative = get_unique_facts(
                [f for f in facts_metadata if 'лучше' in f.text.lower() or 'хуже' in f.text.lower() or 'отличается' in f.text.lower() or 'сравн' in f.text.lower()], 3, used_texts
            )
            data['comparison'] = '\n'.join(comparative) if comparative else 'Сравнение не найдено.'
            
            advantages = get_unique_facts(
                [f for f in facts_metadata if 'преимуществ' in f.text.lower() or 'достоинств' in f.text.lower() or 'плюс' in f.text.lower()], 3, used_texts
            )
            data['advantages'] = '\n'.join(advantages) if advantages else 'Преимущества не указаны.'
            
            disadvantages = get_unique_facts(
                [f for f in facts_metadata if 'недостат' in f.text.lower() or 'минус' in f.text.lower() or 'проблем' in f.text.lower()], 3, used_texts
            )
            data['disadvantages'] = '\n'.join(disadvantages) if disadvantages else 'Недостатки не указаны.'
            
            recommendations = get_unique_facts(
                [f for f in facts_metadata if 'рекоменд' in f.text.lower() or 'совет' in f.text.lower() or 'следует' in f.text.lower()], 2, used_texts
            )
            data['recommendations'] = '\n'.join(recommendations) if recommendations else ''

        else:  # profile == 'default' или любой другой
            summary_facts = get_unique_facts(facts_metadata, 1, used_texts)
            data['summary'] = summary_facts[0] if summary_facts else 'Нет информации.'
            
            details_facts = get_unique_facts(facts_metadata, 5, used_texts)
            data['details'] = '\n'.join(details_facts) if details_facts else 'Дополнительная информация отсутствует.'
            
            extra_facts = get_unique_facts(facts_metadata, 4, used_texts)
            data['extra'] = '\n'.join(extra_facts) if extra_facts else ''

        return data

    # ----------------------------------------------------------------------
    # NER ПРИЗНАКИ (с использованием GLiNER2)
    # ----------------------------------------------------------------------
    async def _compute_ner_features(self, sent: str, query: str) -> Tuple[float, List[str]]:
        """
        Умный NER: подстраивается под смысл вопроса.
        """
        if not self.ner_enabled or not hasattr(self, 'gliner_model'):
            return 0.0, []

        try:
            # 1. Определяем профиль через детектор
            profile = self._detect_query_profile(query)
            
            # 2. Формируем список меток под этот конкретный запрос
            competency = self.ner_competencies.get(profile, {})
            specific_labels = competency.get('labels', ["Concept", "Object"])
            
            # Собираем финальный список (База + Специфика)
            # Не перегружаем модель: 5-7 меток — это идеал для точности
            current_labels = list(self.base_mapping.keys())[:3] + specific_labels
            
            # 3. Извлекаем сущности (используем asyncio.to_thread для GLiNER2)
            entities = await asyncio.to_thread(
                self.gliner_model.extract_entities, 
                sent, 
                current_labels, 
                threshold=0.45
            )

            ner_score = 0.0
            ner_types = []

            for ent in entities:
                label = ent['label']
                score = ent['score']
                
                # Считаем вес: если метка из "умного" списка — даем спец. вес, иначе базу
                weight = competency.get('weight', 10.0) if label in specific_labels else self.base_mapping.get(label, 4.0)
                
                ner_score += weight * score
                ner_types.append(label)

            self.stats['ner_used'] += 1
            return ner_score, list(set(ner_types))

        except Exception as e:
            logger.debug(f"GLiNER2 error: {e}")
            return 0.0, []

    # ----------------------------------------------------------------------
    # ОПРЕДЕЛЕНИЕ ТИПА ЗАПРОСА (ПРОФИЛЬ ДЛЯ ШАБЛОНОВ)
    # ----------------------------------------------------------------------
    def _detect_query_profile(self, query: str) -> str:
        """
        Определяет тип вопроса для выбора динамических меток NER и шаблона ответа.
        Разделяет смежные области (биология/физика), чтобы GLiNER2 не путался.
        """
        q = query.lower()

        # 1. Философия и Сознание (Квалиа, Онтология)
        if re.search(r'философ|сознан|квалиа|субъективн|онтолог|эпистем|феноменолог|бытие|смысл|этик[а]|интенциональност|ницше|кант|хайдеггер|априорн', q):
            return 'philosophical_concept'

        # 2. Глубокая Квантовая Физика и Механика
        if re.search(r'квантов[аяи]|декогеренц|суперпозиц|шредер|вигнер|волновой\s+функц|корпускуляр|неопределенност|гамильтониан|кхд|струн|эйнштейн', q):
            return 'quantum_physics_deep'

        # 3. Математика и Формализм
        if re.search(r'теорем[аы]|пифагор|эйлер|ферма|гильберт|аксиом[аы]|формул[аы]|интеграл|производн|дифференц|логарифм|число\s+пи', q):
            return 'mathematical_theorem'

        # 4. Биология, Генетика и Сложные Системы
        if re.search(r'биологи[яи]|генет|синап|нейрон|эволюц|днк|рнк|симбиоз|транскрипц|белок|фермент|митохондр', q):
            return 'biological_system'

        # 5. Программирование / IT / ИИ
        if re.search(r'программир|язык\s+программир|функц[ия]|класс|алгоритм|библиотек[аи]|python|java|c\+\+|javascript|фреймворк|нейросеть|обучение\s+модели', q):
            return 'programming_concept'

        # 6. Общие научные концепции (Физика, Химия, Космос)
        if re.search(r'физик[а]|хими[я]|космос|астрон|теори[я]|закон|принцип|гипотез[а]|парадокс|эффект|реакция', q):
            return 'scientific_concept'

        # 7. Исторические события / Личности
        if re.search(r'истори[яи]|событи[ея]|войн[аы]|революц[ия]|биографи[яи]|родился|умер|году|век[а]|империя|царь|король', q):
            return 'historical_event'

        # 8. Типовые вопросы (Factoid / How-Why / Evaluation)
        if re.search(r'\bкто\b|\bгде\b|\bкогда\b|\bчто такое\b', q):
            return 'factoid'
        
        if re.search(r'\bкак\b|\bпочему\b|\bзачем\b|\bпричина\b', q):
            return 'how_why'

        if re.search(r'\bлучше\b|\bхуже\b|\bстоит ли\b|\bсравнить\b|\bотличие\b', q):
            return 'evaluation'

        return 'default'

    # ----------------------------------------------------------------------
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ----------------------------------------------------------------------
    def _extract_domain(self, url: str) -> str:
        """Извлекает домен из URL."""
        from urllib.parse import urlparse
        try:
            return urlparse(url).netloc.lower()
        except:
            return ''

    def _get_junk_phrases(self) -> List[str]:
        """Загружает список мусорных фраз."""
        return [
            'материал из википедии', 'стабильная версия', 'перейти к навигации',
            'перейти к поиску', 'категория:', 'шаблон:', 'источник —', 'дата обращения:',
            'архивировано', 'автор оригинала:', 'лицензия creative commons',
            'эта страница в последний раз', 'у этого термина существуют и другие значения',
            'см. также', 'примечания', 'ссылки', 'литература', 'фото:', '©',
            'getty images', 'reuters', 'ap', '↑', '↓', '←', '→'
        ]

    def _deduplicate_facts_by_text(self, facts: List[Dict]) -> List[Dict]:
        """Удаляет точные дубликаты текста."""
        seen = set()
        unique = []
        for f in facts:
            sig = f['text'][:100].lower()
            if sig not in seen:
                seen.add(sig)
                unique.append(f)
        return unique

    async def _get_query_embedding(self, query: str):
        """Получает эмбеддинг запроса с таймаутом."""
        try:
            from appp.services.embedding.bge_m3 import embedder
            if embedder is None or embedder.model is None:
                return None
            return await asyncio.wait_for(embedder.embed(query), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("⚠️ Таймаут получения эмбеддинга запроса")
            return None
        except Exception as e:
            logger.error(f"⚠️ Ошибка получения эмбеддинга запроса: {e}")
            return None

    def _generate_extractive_summary(self, text: str, sentences_count: int = 3) -> str:
        """Экстрактивное резюме через sumy (если установлено)."""
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

    def _calculate_confidence(self, chunks: List[Dict], facts: List[Dict] = None, query: str = "") -> float:
        """
        Улучшенный расчёт уверенности на основе:
        - качества чанков
        - авторитетности доменов
        - количества уникальных источников
        - согласованности фактов
        - наличия NER-сущностей
        """
        if not chunks:
            return 0.0
        
        avg_chunk_quality = sum(c.get('quality_score', 0) for c in chunks) / len(chunks)
        
        domain_scores = []
        for chunk in chunks:
            url = chunk.get('source_url', '')
            domain = self._extract_domain(url)
            if any(pd in domain for pd in self.priority_domains):
                domain_scores.append(1.0)
            elif any(ld in domain for ld in self.low_trust_domains):
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

    async def _process_document(self, doc: Dict) -> Dict:
        """Обработка одного документа."""
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

        chunks = [c for c in chunks if c['quality_score'] >= 0.5]
        chunks = chunks[:self.max_chunks_per_document]

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