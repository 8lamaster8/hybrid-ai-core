"""
📊 РАНЖИРОВЩИК - Центральный сервис оценки релевантности фактов
"""

import os
import yaml
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from sentence_transformers import util
import asyncio
from datetime import datetime

from appp.core.logging import logger

from appp.services.embedding.bge_m3 import embedder


@dataclass
class Fact:
    """Структура факта для ранжирования."""
    text: str
    embedding: Optional[np.ndarray] = None
    source_domain: str = ""
    position_ratio: float = 0.5
    ner_score: float = 0.0
    ner_types: List[str] = field(default_factory=list)
    length: int = 0
    contains_definition: bool = False
    contains_causal: bool = False


class RankingService:
    """
    Сервис ранжирования фактов.
    Веса и профили загружаются из configs/ranking.yaml.
    """
    
    def __init__(self, config_path: str = "./configs/ranking.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.domain_credit_cache = {}
        logger.info("📊 RankingService инициализирован")
    
    def _load_config(self) -> Dict:
        default_config = {
            'profiles': {
                'default': {
                    'weights': {
                        'relevance': 0.4,
                        'domain_credit': 0.15,
                        'position': 0.1,
                        'ner': 0.2,
                        'length': 0.05,
                        'definition_causal': 0.1,
                        'uniqueness_penalty': -0.2
                    },
                    'length_optimal': 100,
                    'length_steepness': 0.1,
                    'ner_max_norm': 20.0,
                    'uniqueness_threshold': 0.85
                }
            },
            'domain_credit': {
                'priority': 1.0,
                'neutral': 0.6,
                'low': 0.3
            }
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                if user_config:
                    if 'profiles' in user_config:
                        default_config['profiles'].update(user_config['profiles'])
                    if 'domain_credit' in user_config:
                        default_config['domain_credit'].update(user_config['domain_credit'])
                logger.info(f"📁 Загружен конфиг ранжирования из {self.config_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки ranking.yaml: {e}, используются defaults")
        else:
            logger.warning(f"⚠️ {self.config_path} не найден, используются значения по умолчанию")
        return default_config
    
    def get_domain_credit(self, domain: str, priority_domains: set, low_trust_domains: set) -> float:
        if domain in self.domain_credit_cache:
            return self.domain_credit_cache[domain]
        credit = self.config['domain_credit']['neutral']
        if any(pd in domain for pd in priority_domains):
            credit = self.config['domain_credit']['priority']
        elif any(ld in domain for ld in low_trust_domains):
            credit = self.config['domain_credit']['low']
        self.domain_credit_cache[domain] = credit
        return credit
    
    def _signal_length(self, length: int, profile: Dict) -> float:
        opt = profile.get('length_optimal', 100)
        steep = profile.get('length_steepness', 0.1)
        return 1 / (1 + np.exp(-(length - opt) * steep))
    
    def _signal_ner(self, ner_score: float, profile: Dict) -> float:
        max_norm = profile.get('ner_max_norm', 20.0)
        return min(ner_score / max_norm, 1.0)
    
    def _signal_position(self, position_ratio: float) -> float:
        return 1.0 - position_ratio
    
    def _signal_definition_causal(self, fact: Fact) -> float:
        score = 0.0
        if fact.contains_definition:
            score += 0.3
        if fact.contains_causal:
            score += 0.2
        return min(score, 0.5)
    
    async def rank(
        self,
        query: str,
        facts: List[Fact],
        query_embedding: Optional[np.ndarray] = None,
        priority_domains: set = None,
        low_trust_domains: set = None,
        profile_name: str = 'default'
    ) -> List[Tuple[Fact, float]]:
        """
        Ранжирование фактов с защитой от зависаний и увеличенными таймаутами.
        """
        logger.info(f"📊 RankingService.rank: {len(facts)} фактов, профиль '{profile_name}'")
        start_rank = datetime.now()

        if not facts:
            return []

        profile = self.config['profiles'].get(profile_name, self.config['profiles']['default'])
        weights = profile['weights']

        # ---------- ПРОВЕРКА ДОСТУПНОСТИ ЭМБЕДДЕРА ----------
        from appp.services.embedding.bge_m3 import embedder
        embedder_available = False
        if embedder is not None and embedder.model is not None:
            embedder_available = True
            logger.debug("   ✅ Эмбеддер доступен")
        else:
            logger.warning("   ⚠️ Эмбеддер НЕ доступен, релевантность и MMR отключены")

        # ---------- ЭМБЕДДИНГ ЗАПРОСА ----------
        if query_embedding is None:
            if embedder_available:
                try:
                    # Таймаут 10 секунд на эмбеддинг запроса (было 5)
                    query_embedding = await asyncio.wait_for(
                        embedder.embed(query),
                        timeout=10.0
                    )
                    logger.debug("   ✅ Эмбеддинг запроса получен")
                except asyncio.TimeoutError:
                    logger.warning("   ⚠️ Таймаут эмбеддинга запроса (10 сек)")
                    query_embedding = None
                except Exception as e:
                    logger.warning(f"   ⚠️ Ошибка эмбеддинга запроса: {e}")
                    query_embedding = None
            else:
                query_embedding = None

        # ---------- ЭМБЕДДИНГИ ФАКТОВ ----------
        fact_embeddings = []
        if embedder_available:
            # Собираем тексты, у которых нет эмбеддинга
            need_embed = []
            for i, f in enumerate(facts):
                if f.embedding is not None:
                    fact_embeddings.append(f.embedding)
                else:
                    need_embed.append((i, f.text))

            if need_embed:
                try:
                    texts = [t for _, t in need_embed]
                    # Таймаут 30 секунд на батч эмбеддингов (было 10)
                    embs = await asyncio.wait_for(
                        embedder.embed(texts),
                        timeout=30.0
                    )
                    for (i, _), emb in zip(need_embed, embs):
                        facts[i].embedding = emb
                        fact_embeddings.append(emb)
                    logger.debug(f"   ✅ Эмбеддинги для {len(need_embed)} фактов получены")
                except asyncio.TimeoutError:
                    logger.warning("   ⚠️ Таймаут батч-эмбеддинга фактов (30 сек) - продолжаем без эмбеддингов")
                    fact_embeddings = []  # отключаем эмбеддинги полностью для этого запроса
                except Exception as e:
                    logger.warning(f"   ⚠️ Ошибка батч-эмбеддинга: {e}")
                    fact_embeddings = []
        else:
            fact_embeddings = []

        # ---------- РЕЛЕВАНТНОСТЬ (косинус) ----------
        if query_embedding is not None and fact_embeddings:
            try:
                similarities = util.cos_sim(query_embedding, np.vstack(fact_embeddings))[0].cpu().numpy()
                logger.debug(f"   ✅ Косинусная близость вычислена")
            except Exception as e:
                logger.warning(f"   ⚠️ Ошибка вычисления косинуса: {e}")
                similarities = np.zeros(len(facts))
        else:
            similarities = np.zeros(len(facts))

        # ---------- ОСТАЛЬНЫЕ СИГНАЛЫ ----------
        priority_domains = priority_domains or set()
        low_trust_domains = low_trust_domains or set()
        domain_credits = [self.get_domain_credit(f.source_domain, priority_domains, low_trust_domains) for f in facts]
        position_scores = [self._signal_position(f.position_ratio) for f in facts]
        ner_scores = [self._signal_ner(f.ner_score, profile) for f in facts]
        length_scores = [self._signal_length(f.length, profile) for f in facts]
        defcausal_scores = [self._signal_definition_causal(f) for f in facts]

        # ---------- БАЗОВЫЙ СЧЁТ ----------
        final_scores = (
            weights.get('relevance', 0.4) * similarities +
            weights.get('domain_credit', 0.15) * np.array(domain_credits) +
            weights.get('position', 0.1) * np.array(position_scores) +
            weights.get('ner', 0.2) * np.array(ner_scores) +
            weights.get('length', 0.05) * np.array(length_scores) +
            weights.get('definition_causal', 0.1) * np.array(defcausal_scores)
        )
        logger.debug(f"   ✅ Базовые веса рассчитаны")

        # ---------- MMR-ОТБОР (с защитой) ----------
        uniqueness_penalty_weight = weights.get('uniqueness_penalty', -0.2)
        threshold = profile.get('uniqueness_threshold', 0.85)

        selected = []
        selected_indices = set()

        for iteration in range(len(facts)):
            best_score = -np.inf
            best_idx = -1
            for i in range(len(facts)):
                if i in selected_indices:
                    continue
                score = final_scores[i]
                if selected_indices and embedder_available and fact_embeddings:
                    try:
                        # косинус только с уже выбранными
                        max_sim = max(
                            util.cos_sim(fact_embeddings[i], fact_embeddings[j]).item()
                            for j in selected_indices
                        )
                        score += uniqueness_penalty_weight * max_sim
                    except Exception as e:
                        logger.debug(f"   ⚠️ MMR: ошибка сходства, пропускаем штраф")
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx != -1:
                selected_indices.add(best_idx)
                selected.append((facts[best_idx], best_score))
            else:
                break

        elapsed = (datetime.now() - start_rank).total_seconds()
        logger.info(f"📊 RankingService.rank завершён: {len(selected)} фактов, время {elapsed:.2f} сек")
        return selected