#!/usr/bin/env python3
"""
🤖 AUTONOMOUS AI PRODUCTION READY SYSTEM v3.1
Полностью локальная система, с эмбеддингами, графом, комитетом, интервьюером, ENGRAM
Расширенная: NER, RankingService, умное ранжирование фактов
"""

import os
import sys
import asyncio
import traceback
import yaml
from datetime import datetime
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Core
from appp.core.config import Config
from appp.core.logging import setup_logging, get_logger

# Services
from appp.services.detective.detective import Detective
from appp.services.committee.quality_committee import QualityCommittee
from appp.services.analyst.knowledge_analyst import KnowledgeAnalyst
from appp.services.interviewer.question_generator import QuestionGenerator
from appp.services.storage.chroma_db import ChromaDBService
from appp.services.storage.networkx_graph import NetworkXGraphService
from appp.services.storage.engram_db import EngramService
from appp.services.embedding.bge_m3 import BGE_M3_Embedder
from appp.services.embedding.bge_m3 import embedder as global_embedder
from appp.core.logging import logger

# Coordination
from appp.coordination.service_coordinator import ServiceCoordinator, get_coordinator
from appp.utils.response_templates import RESPONSE_TEMPLATES, format_rich_response


logger = get_logger('AutonomousAI')


class AutonomousAIPro:
    def __init__(self, config_path: str = None):
        self.config = Config.load(config_path)

        self.coordinator = None
        self.detective = None
        self.committee = None
        self.analyst = None
        self.interviewer = None
        self.chroma_db = None
        self.graph_db = None
        self.engram = None
        self.embedder = None

        self.is_initialized = False
        self.is_running = False

        self.session_stats = {
            'start_time': None,
            'questions_asked': 0,
            'research_done': 0,
            'learning_cycles': 0,
            'errors': 0
        }

        logger.info("🤖 AutonomousAIPro инициализирован")

    async def initialize(self):
        """Инициализация всех компонентов"""
        logger.info("🔄 Инициализация системы...")

        try:
            # 1. Эмбеддинги
            logger.info("1/8 🧠 Загрузка модели эмбеддингов...")
            self.embedder = BGE_M3_Embedder(
                model_name=self.config.embedding.model_name,
                model_path=self.config.embedding.model_path,
                device=self.config.embedding.device,
                normalize_embeddings=self.config.embedding.normalize_embeddings,
                cache_dir=self.config.embedding.cache_dir,
                max_cache_size=self.config.embedding.max_cache_size,
                batch_size=self.config.embedding.batch_size,
                embedding_dimension=self.config.embedding.embedding_dimension
            )
            embedder_ok = await self.embedder.initialize()

            # !!! ВАЖНО: переназначаем глобальный embedder в модуле !!!
            import appp.services.embedding.bge_m3 as bge_module
            bge_module.embedder = self.embedder
            logger.info("   ✅ Глобальный embedder модуля bge_m3 заменён")

            if not embedder_ok:
                logger.warning("⚠️ Эмбеддинги не загружены, ChromaDB и ранжирование по эмбеддингам будут недоступны")
            else:
                # Переназначаем глобальный embedder для RankingService и других модулей
                global_embedder = self.embedder
                logger.info("   ✅ Глобальный embedder переназначен")

            # 2. Хранилища
            logger.info("2/8 💾 Инициализация хранилищ...")

            # ChromaDB (только если эмбеддер загружен)
            if embedder_ok and self.embedder.model is not None:
                self.chroma_db = ChromaDBService(
                    persist_directory=self.config.storage.chroma_path,
                    embedding_function=self.embedder.get_embedding_function(),
                    collection_name="knowledge_embeddings",
                    max_collection_size=self.config.storage.max_chroma_records
                )
                await self.chroma_db.initialize()
                logger.info("   ✅ ChromaDB инициализирована")
            else:
                self.chroma_db = None
                logger.warning("   ⚠️ ChromaDB отключена (нет эмбеддингов)")

            # NetworkX граф
            self.graph_db = NetworkXGraphService(
                db_path=self.config.storage.graph_path,
                auto_save=self.config.storage.auto_save,
                save_interval=self.config.storage.save_interval
            )
            await self.graph_db.initialize()

            # 🧠 Engram память
            logger.info("   🧠 Загрузка Engram памяти...")
            try:
                self.engram = EngramService(
                    db_path=self.config.storage.engram_path,
                    max_records=self.config.storage.max_engram_records
                )
                await self.engram.initialize()
                logger.info("   ✅ Engram память загружена")
            except Exception as e:
                logger.warning(f"   ⚠️ EngramService не доступен: {e}")
                self.engram = None

            # 3. Детектив
            logger.info("3/8 🔍 Инициализация CURL детектива...")
            detective_config = {
                'search_engine': self.config.detective.search_engine,
                'max_pages_per_topic': self.config.detective.max_pages_per_topic,
                'max_results_per_page': self.config.detective.max_results_per_page,
                'min_content_length': self.config.detective.min_content_length,
                'max_content_length': self.config.detective.max_content_length,
                'timeout': self.config.detective.timeout,
                'user_agent': self.config.detective.user_agent,
                'proxies': self.config.detective.proxies,
                'retry_attempts': self.config.detective.retry_attempts,
                'blacklist_domains': self.config.detective.blacklist_domains,
                'priority_domains': getattr(self.config.detective, 'priority_domains', [])
            }
            self.detective = Detective(detective_config)
            await self.detective.initialize()

            # 4. Комитет качества
            logger.info("4/8 ⚖️ Инициализация комитета качества...")
            committee_config = {
                'min_relevance_score': self.config.committee.min_relevance_score,
                'min_quality_score': self.config.committee.min_quality_score,
                'min_uniqueness_score': self.config.committee.min_uniqueness_score,
                'blocked_keywords': self.config.committee.blocked_keywords,
                'enable_embedding_check': self.config.committee.enable_embedding_check,
                'embedding_threshold': self.config.committee.embedding_threshold,
                'min_sentences': self.config.committee.min_sentences,
                'language': self.config.committee.language
            }
            self.committee = QualityCommittee(committee_config)
            await self.committee.initialize()

            # --- Загрузка списков доменов для аналитика из quality.yaml ---
            priority_domains = []
            low_trust_domains = []
            quality_config_path = os.path.join(BASE_DIR, 'configs', 'quality.yaml')
            if os.path.exists(quality_config_path):
                try:
                    with open(quality_config_path, 'r', encoding='utf-8') as f:
                        qc = yaml.safe_load(f)
                        priority_domains = qc.get('priority_domains', [])
                        low_trust_domains = qc.get('low_trust_domains', [])
                    logger.info(f"📁 Загружены домены для аналитика: приоритетных {len(priority_domains)}, низкого доверия {len(low_trust_domains)}")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось загрузить quality.yaml: {e}")

            # 5. Аналитик знаний (с поддержкой NER и ранжирования)
            logger.info("5/8 📚 Инициализация аналитика знаний...")
            analyst_config = {
                'chunk_size': self.config.analyst.chunk_size,
                'chunk_overlap': self.config.analyst.chunk_overlap,
                'min_chunk_length': self.config.analyst.min_chunk_length,
                'max_chunks_per_document': self.config.analyst.max_chunks_per_document,
                'extraction_strategy': self.config.analyst.extraction_strategy,
                'enable_summarization': self.config.analyst.enable_summarization,
                'summary_length': self.config.analyst.summary_length,
                'enable_entity_extraction': self.config.analyst.enable_entity_extraction,
                'enable_relation_extraction': self.config.analyst.enable_relation_extraction,
                'language': self.config.analyst.language,
                'min_confidence': self.config.analyst.min_confidence,
                # Добавляем параметры для ранжирования и NER
                'enable_ner': True,
                'priority_domains': priority_domains,
                'low_trust_domains': low_trust_domains
            }
            self.analyst = KnowledgeAnalyst(analyst_config)
            await self.analyst.initialize()

            # 6. Интервьюер
            logger.info("6/8 🎤 Инициализация интервьюера...")
            interviewer_config = {
                'max_questions_per_topic': getattr(self.config, 'max_questions_per_topic', 15),
                'question_depth_levels': getattr(self.config, 'question_depth_levels', 3),
                'enable_followup_questions': getattr(self.config, 'enable_followup_questions', True),
                'question_types': getattr(self.config, 'question_types', ['factual', 'comparative', 'causal', 'procedural']),
                'min_question_quality': getattr(self.config, 'min_question_quality', 0.6),
                'language': self.config.committee.language
            }
            self.interviewer = QuestionGenerator(interviewer_config)
            await self.interviewer.initialize()

            # 7. Координатор сервисов
            logger.info("7/8 🎯 Инициализация координатора сервисов...")
            self.coordinator = ServiceCoordinator(
                self.config,
                detective=self.detective,
                committee=self.committee,
                analyst=self.analyst,
                interviewer=self.interviewer,
                chroma_db=self.chroma_db,
                graph_db=self.graph_db,
                engram=self.engram,
                embedder=self.embedder
            )
            await self.coordinator.initialize()

            self.is_initialized = True
            self.session_stats['start_time'] = datetime.now()

            logger.info("✅ Система полностью инициализирована и готова к работе")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}", exc_info=True)
            return False

    # ---------- Публичные методы ----------
    async def ask_question(self, question: str) -> dict:
        if not self.is_initialized:
            return {'error': 'Система не инициализирована'}
        logger.info(f"❓ Вопрос: {question}")
        self.session_stats['questions_asked'] += 1

        task_data = {
            'type': 'simple_question',
            'question': question,
            'priority': 1
        }
        task_id = await self.coordinator.submit_task(task_data)

        timeout = self.config.coordinator.task_timeout  # 300 секунд по умолчанию
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            await asyncio.sleep(0.5)
            task_status = await self.coordinator.get_task_status(task_id)
            if task_status['status'] in ['completed', 'failed']:
                return task_status.get('result', {})
        return {'error': f'Таймаут обработки вопроса (>{timeout} сек)'}

    async def research_topic(self, topic: str, depth: int = 2) -> dict:
        self.session_stats['research_done'] += 1
        task_data = {
            'type': 'deep_research',
            'topic': topic,
            'depth': depth,
            'priority': 1
        }
        task_id = await self.coordinator.submit_task(task_data)
        for _ in range(60):
            await asyncio.sleep(1)
            task_status = await self.coordinator.get_task_status(task_id)
            if task_status['status'] in ['completed', 'failed']:
                return task_status.get('result', {})
        return {'error': 'Таймаут исследования'}

    async def explore_topic(self, topic: str) -> dict:
        task_data = {
            'type': 'topic_exploration',
            'topic': topic,
            'max_cycles': 3,
            'priority': 1
        }
        task_id = await self.coordinator.submit_task(task_data)
        for _ in range(120):
            await asyncio.sleep(1)
            task_status = await self.coordinator.get_task_status(task_id)
            if task_status['status'] in ['completed', 'failed']:
                return task_status.get('result', {})
        return {'error': 'Таймаут исследования'}

    async def self_learn(self) -> dict:
        self.session_stats['learning_cycles'] += 1
        task_data = {
            'type': 'self_learning',
            'priority': 2
        }
        task_id = await self.coordinator.submit_task(task_data)
        return {
            'success': True,
            'message': 'Задача самообучения запущена',
            'task_id': task_id
        }

    async def get_system_status(self) -> dict:
        if not self.is_initialized:
            return {'initialized': False}

        metrics = await self.coordinator.get_system_metrics()
        status = {
            'initialized': self.is_initialized,
            'running': self.is_running,
            'session': self.session_stats,
            'uptime_seconds': (datetime.now() - self.session_stats['start_time']).total_seconds() if self.session_stats['start_time'] else 0,
            'coordinator': metrics.get('coordinator', {}),
            'detective': await self.detective.get_stats() if self.detective else {},
            'committee': await self.committee.get_stats() if self.committee else {},
            'analyst': await self.analyst.get_analyst_stats() if self.analyst else {},
            'interviewer': await self.interviewer.get_stats() if self.interviewer else {},
            'chroma': await self.chroma_db.get_detailed_stats() if self.chroma_db else {},
            'graph': await self.graph_db.get_stats() if self.graph_db else {},
            'engram': await self.engram.get_stats() if self.engram else {},
            'embedder': await self.embedder.get_metrics() if self.embedder else {}
        }
        return status

    # ---------- Интерактивный режим ----------
    async def interactive_mode(self):
        print("\n" + "=" * 80)
        print("🤖 AUTONOMOUS AI PRO - ИНТЕРАКТИВНЫЙ РЕЖИМ")
        print("=" * 80)
        print("\n📋 Доступные команды:")
        print("   • вопрос <текст>     - задать вопрос")
        print("   • исследовать <тема> - глубокое исследование")
        print("   • цикл <тема>        - исследование с циклами координат")
        print("   • обучиться          - запустить самообучение")
        print("   • статус             - состояние системы")
        print("   • статистика         - подробная статистика")
        print("   • выход              - завершение работы")
        print("\n" + "=" * 80)

        while True:
            try:
                user_input = input("\n🎯 > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                    print("👋 Завершение работы...")
                    break
                elif user_input.lower() in ['статус', 'status']:
                    status = await self.get_system_status()
                    self._display_status(status)
                elif user_input.lower() in ['статистика', 'stats']:
                    status = await self.get_system_status()
                    self._display_detailed_stats(status)
                elif user_input.lower() in ['обучиться', 'обучение', 'learn']:
                    print("🧠 Запуск самообучения...")
                    result = await self.self_learn()
                    print(f"✅ Задача создана: {result.get('task_id')}")
                elif user_input.lower().startswith('вопрос '):
                    question = user_input[7:].strip()
                    if question:
                        print(f"❓ Обрабатываю: {question}")
                        result = await self.ask_question(question)
                        self._display_answer(result)
                    else:
                        print("⚠️ Укажите вопрос")
                elif user_input.lower().startswith('исследовать '):
                    topic = user_input[12:].strip()
                    if topic:
                        print(f"🔍 Глубокое исследование: {topic}")
                        result = await self.research_topic(topic, depth=2)
                        self._display_research(result)
                    else:
                        print("⚠️ Укажите тему")
                elif user_input.lower().startswith('цикл '):
                    topic = user_input[5:].strip()
                    if topic:
                        print(f"🌀 Цикл координат: {topic}")
                        result = await self.explore_topic(topic)
                        self._display_exploration(result)
                    else:
                        print("⚠️ Укажите тему")
                else:
                    print(f"❓ Вопрос: {user_input}")
                    result = await self.ask_question(user_input)
                    self._display_answer(result)
            except KeyboardInterrupt:
                print("\n👋 Прервано пользователем")
                break
            except Exception as e:
                logger.error(f"Ошибка в интерактивном режиме: {e}")
                print(f"❌ Ошибка: {e}")

    # ---------- Методы отображения ----------
    def _display_answer(self, result: dict):
        if 'error' in result:
            print(f"\n❌ Ошибка: {result['error']}")
            return

        profile = result.get('profile', '')
        key_facts_metadata = result.get('key_facts_metadata', [])
        query = result.get('query', '')

        # Импортируем шаблоны здесь, чтобы избежать циклического импорта
        from appp.utils.response_templates import RESPONSE_TEMPLATES, format_rich_response

        # Пробуем использовать шаблон, если профиль определён и есть метаданные
        if profile and key_facts_metadata and profile in RESPONSE_TEMPLATES:
            try:
                # Подготавливаем данные через аналитика
                template_data = self.analyst._prepare_template_data(profile, key_facts_metadata, query)
                logger.info(f"PROFILE: {profile}, METADATA COUNT: {len(key_facts_metadata)}")
                formatted = format_rich_response(profile, template_data)
                print("\n" + "✅" * 40)
                print(f"🤖 ОТВЕТ (источник: {result.get('source', 'unknown')})")
                print("✅" * 40)
                print(f"\n{formatted}\n")
            except Exception as e:
                logger.error(f"Ошибка шаблонного форматирования: {e}", exc_info=True)
                # fallback на старый формат
                self._display_fallback_answer(result)
        else:
            # --- СТАРЫЙ ФОРМАТ (СПИСОК ФАКТОВ) ---
            self._display_fallback_answer(result)

        # --- ОБЩАЯ ИНФОРМАЦИЯ (уверенность, время, источники) ---
        if 'confidence' in result:
            print(f"📊 Уверенность: {result['confidence']:.1%}")
        if 'processing_time' in result:
            print(f"⏱️  Время: {result['processing_time']:.2f} сек")
        if 'sources' in result and result['sources']:
            print("\n🔗 Источники:")
            for i, src in enumerate(result['sources'][:3], 1):
                print(f"   {i}. {src}")
        print("\n" + "✅" * 40)

    def _display_fallback_answer(self, result: dict):
        """Старый формат — список фактов"""
        print("\n" + "✅" * 40)
        print(f"🤖 ОТВЕТ (источник: {result.get('source', 'unknown')})")
        print("✅" * 40)
        answer = result.get('answer', result.get('synthesis', 'Нет ответа'))
        print(f"\n{answer}\n")

    def _display_research(self, result: dict):
        if 'error' in result:
            print(f"\n❌ Ошибка: {result['error']}")
            return
        print("\n" + "🔍" * 40)
        print(f"📚 РЕЗУЛЬТАТ ИССЛЕДОВАНИЯ: {result.get('topic', '')}")
        print("🔍" * 40)
        if 'synthesis' in result:
            print(f"\n{result['synthesis']}\n")
        if 'key_findings' in result:
            print("🎯 Ключевые находки:")
            for i, finding in enumerate(result['key_findings'][:5], 1):
                print(f"   {i}. {finding}")
        if 'sources_used' in result:
            print(f"\n📊 Использовано источников: {result['sources_used']}")
            print(f"📈 Глубина: {result.get('depth_achieved', 'N/A')}")
        print("\n" + "🔍" * 40)

    def _display_exploration(self, result: dict):
        if 'error' in result:
            print(f"\n❌ Ошибка: {result['error']}")
            return
        print("\n" + "🌀" * 40)
        print(f"🔄 ЦИКЛ КООРДИНАТ: {result.get('topic', '')}")
        print("🌀" * 40)
        print(f"\n✅ Завершено циклов: {result.get('cycles_completed', 0)}")
        print(f"📦 Обработано чанков: {result.get('knowledge_chunks', 0)}")
        if 'synthesis' in result:
            print(f"\n📋 Синтез:\n{result['synthesis']}\n")
        if 'key_insights' in result:
            print("💡 Ключевые инсайты:")
            for i, insight in enumerate(result['key_insights'][:5], 1):
                print(f"   {i}. {insight}")
        print("\n" + "🌀" * 40)

    def _display_status(self, status: dict):
        print("\n" + "📊" * 40)
        print("СОСТОЯНИЕ СИСТЕМЫ")
        print("📊" * 40)
        if not status.get('initialized'):
            print("❌ Система не инициализирована")
            return
        uptime = status.get('uptime_seconds', 0)
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        print(f"⏱️  Время работы: {hours}ч {minutes}мин")
        print(f"❓ Вопросов: {status['session']['questions_asked']}")
        print(f"🔍 Исследований: {status['session']['research_done']}")
        print(f"🧠 Циклов обучения: {status['session']['learning_cycles']}")
        coord = status.get('coordinator', {})
        print(f"\n🎯 Координатор:")
        print(f"   • Задач обработано: {coord.get('tasks_processed', 0)}")
        print(f"   • Очередь: {coord.get('queue_size', 0)}")
        print(f"   • Утилизация: {coord.get('worker_utilization', 0):.1%}")
        chroma = status.get('chroma', {})
        print(f"\n💾 ChromaDB:")
        print(f"   • Записей: {chroma.get('collection_size', 0)}")
        print(f"   • Хиты кэша: {chroma.get('cache_hit_rate', 0):.1%}")
        graph = status.get('graph', {})
        print(f"\n🕸️  Граф знаний:")
        print(f"   • Узлов: {graph.get('nodes', 0)}")
        print(f"   • Связей: {graph.get('edges', 0)}")
        engram = status.get('engram', {})
        print(f"\n🧠 Engram память:")
        print(f"   • Записей: {engram.get('total_records', 0)}")
        print(f"   • Хиты: {engram.get('hit_rate', 0):.1%}")
        print("\n" + "📊" * 40)

    def _display_detailed_stats(self, status: dict):
        self._display_status(status)
        committee = status.get('committee', {})
        print(f"\n⚖️  Комитет качества:")
        print(f"   • Проверок: {committee.get('evaluations', 0)}")
        print(f"   • Одобрено: {committee.get('approved', 0)} ({committee.get('approval_rate', 0):.1%})")
        analyst = status.get('analyst', {})
        print(f"\n📚 Аналитик:")
        print(f"   • Документов: {analyst.get('documents_processed', 0)}")
        print(f"   • Чанков: {analyst.get('chunks_created', 0)}")
        embedder = status.get('embedder', {})
        print(f"\n🧠 Эмбеддинги:")
        print(f"   • Кэш: {embedder.get('cache_size', 0)} записей")
        print(f"   • Хиты: {embedder.get('cache_hit_rate', 0):.1%}")
        print(f"   • Среднее время: {embedder.get('avg_embedding_time', 0):.3f} сек")

    async def cleanup(self):
        logger.info("🛑 Завершение работы системы...")
        if self.coordinator:
            await self.coordinator.shutdown()
        if self.chroma_db:
            await self.chroma_db.close()
        if self.graph_db:
            await self.graph_db.close()
        if self.engram:
            await self.engram.close()
        if self.detective:
            await self.detective.cleanup()
        if self.embedder:
            await self.embedder.close()
        logger.info("✅ Система завершила работу")
        print("\n👋 До свидания!")


async def main():
    print("\n" + "🚀" * 80)
    print(" AUTONOMOUS AI PRODUCTION READY SYSTEM v3.1 ")
    print(" Полностью локальная, с Engram памятью, BGE-M3, NER и ранжированием ")
    print("🚀" * 80)

    log_file = './data/logs/autonomous_ai.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_level='INFO', log_file=log_file, use_detailed_format=False)

    ai = AutonomousAIPro(config_path='./configs/production.yaml')

    try:
        success = await ai.initialize()
        if not success:
            logger.error("❌ Не удалось инициализировать систему")
            print("❌ Критическая ошибка инициализации. Проверьте логи.")
            return
        await ai.interactive_mode()
    except KeyboardInterrupt:
        print("\n\n👋 Прерывание пользователя")
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}", exc_info=True)
        print(f"\n❌ Непредвиденная ошибка: {e}")
        traceback.print_exc()
    finally:
        await ai.cleanup()


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        sys.exit(1)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Программа прервана")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        traceback.print_exc()