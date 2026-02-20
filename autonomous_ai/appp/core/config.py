"""
Конфигурация системы с увеличенными лимитами
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DetectiveConfig:
    """Конфигурация детектива"""
    search_engine: str = "google"
    max_pages_per_topic: int = 25  # Увеличено до 25
    max_results_per_page: int = 20
    min_content_length: int = 3000  # Минимум 3000 символов
    max_content_length: int = 15000  # Максимум 15000
    timeout: int = 45
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    proxies: list = field(default_factory=list)
    enable_javascript: bool = False
    verify_ssl: bool = True
    retry_attempts: int = 3
    delay_between_requests: float = 1.0
    concurrent_searches: int = 5
    priority_domains: list = field(default_factory=lambda: [
        'wikipedia.org', 'habr.com', 'vc.ru', 'rbc.ru', 'science.org', 'arxiv.org'
    ])
    blacklist_domains: list = field(default_factory=lambda: [
        "facebook.com", "twitter.com", "instagram.com", 
        "tiktok.com", "pinterest.com", "adult", "porn"
    ])


@dataclass
class CommitteeConfig:
    """Конфигурация комитета"""
    min_relevance_score: float = 0.65
    min_quality_score: float = 0.7
    min_uniqueness_score: float = 0.6
    required_keywords: list = field(default_factory=list)
    blocked_keywords: list = field(default_factory=lambda: [
        "архивная копия", "спам", "реклама", "купить сейчас",
        "акция", "скидка", "рекламная статья"
    ])
    max_text_similarity: float = 0.85
    enable_embedding_check: bool = True
    embedding_threshold: float = 0.75
    min_sentences: int = 3
    max_sentence_length: int = 200
    language: str = "ru"


@dataclass
class AnalystConfig:
    """Конфигурация аналитика"""
    chunk_size: int = 1500  # Увеличенный размер чанка
    chunk_overlap: int = 300
    min_chunk_length: int = 500
    max_chunks_per_document: int = 50
    extraction_strategy: str = "semantic"
    enable_summarization: bool = True
    summary_length: int = 300
    enable_entity_extraction: bool = True
    enable_relation_extraction: bool = True
    language: str = "ru"
    min_confidence: float = 0.6
    max_entities_per_chunk: int = 10
    ner_model: Optional[str] = None


@dataclass
class StorageConfig:
    """Конфигурация хранилищ"""
    chroma_path: str = "./data/chroma"
    graph_path: str = "./data/graphs/knowledge_graph.db"
    engram_path: str = "./data/engram/engram.db"
    cache_path: str = "./data/cache"
    max_chroma_records: int = 1000000
    max_engram_records: int = 500000
    auto_save: bool = True
    save_interval: int = 300  # 5 минут
    backup_enabled: bool = True
    backup_interval: int = 3600  # 1 час


@dataclass
class EmbeddingConfig:
    """Конфигурация эмбеддингов"""
    model_name: str = "BAAI/bge-m3" #dummy
    model_path: str = "./models/bge-m3"
    device: str = "cpu"  # или "cuda"
    normalize_embeddings: bool = True
    cache_dir: str = "./data/cache/embeddings"
    max_cache_size: int = 10000
    batch_size: int = 32
    embedding_dimension: int = 1024


@dataclass
class CoordinatorConfig:
    """Конфигурация координатора"""
    max_concurrent_tasks: int = 10
    task_timeout: int = 300  # 5 минут
    retry_failed_tasks: bool = True
    max_retries: int = 2
    enable_priority_queue: bool = True
    monitoring_interval: int = 30
    auto_save_interval: int = 300  # 5 минут
    max_queue_size: int = 1000
    num_workers: int = 5


@dataclass
class LearningConfig:
    """Конфигурация самообучения"""
    enabled: bool = True
    check_interval: int = 60
    priorities: dict = field(default_factory=lambda: {
        'discovery': 0.4,
        'deepening': 0.3,
        'expansion': 0.2,
        'meta_analysis': 0.07,
        'maintenance': 0.03
    })
    intervals: dict = field(default_factory=lambda: {
        'discovery': 3600,
        'deepening': 7200,
        'expansion': 14400,
        'meta_analysis': 86400,
        'maintenance': 43200
    })
    cycles: dict = field(default_factory=lambda: {
        'discovery': {'min_topics': 1, 'max_topics': 3},
        'deepening': {'depth': 2},
        'expansion': {},
        'meta': {},
        'maintenance': {}
    })



@dataclass
class SystemConfig:
    """Основная конфигурация системы"""
    detective: DetectiveConfig = field(default_factory=DetectiveConfig)
    committee: CommitteeConfig = field(default_factory=CommitteeConfig)
    analyst: AnalystConfig = field(default_factory=AnalystConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    
    # Настройки системы
    log_level: str = "INFO"
    log_file: str = "./data/logs/autonomous_ai.log"
    max_log_size: int = 10485760  # 10 MB
    backup_count: int = 5
    
    # Пути
    data_dir: str = "./data"
    models_dir: str = "./models"
    configs_dir: str = "./configs"
    
    # Производительность
    enable_profiling: bool = False
    profile_output: str = "./data/profiling"
    memory_limit_mb: int = 4096
    cpu_limit: int = 4
    
    # Безопасность
    enable_rate_limiting: bool = True
    requests_per_minute: int = 60
    max_content_size_mb: int = 10


class Config:
    """Менеджер конфигурации"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> SystemConfig:
        """Загрузка конфигурации из файла или значений по умолчанию"""
        if cls._config is not None:
            return cls._config
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                cls._config = cls._create_from_dict(yaml_config)
                print(f"✅ Конфигурация загружена из {config_path}")
            except Exception as e:
                print(f"⚠️  Ошибка загрузки конфигурации: {e}, используются значения по умолчанию")
                cls._config = SystemConfig()
        else:
            print("ℹ️  Конфигурационный файл не найден, используются значения по умолчанию")
            cls._config = SystemConfig()
        
        return cls._config
    
    @classmethod
    def _create_from_dict(cls, config_dict: Dict) -> SystemConfig:
        """Создание конфигурации из словаря"""
        return SystemConfig(
            detective=DetectiveConfig(**config_dict.get('detective', {})),
            committee=CommitteeConfig(**config_dict.get('committee', {})),
            analyst=AnalystConfig(**config_dict.get('analyst', {})),
            storage=StorageConfig(**config_dict.get('storage', {})),
            embedding=EmbeddingConfig(**config_dict.get('embedding', {})),
            coordinator=CoordinatorConfig(**config_dict.get('coordinator', {})),

            learning=LearningConfig(**config_dict.get('learning', {})),

            log_level=config_dict.get('log_level', 'INFO'),
            log_file=config_dict.get('log_file', './data/logs/autonomous_ai.log'),
            max_log_size=config_dict.get('max_log_size', 10485760),
            backup_count=config_dict.get('backup_count', 5),
            data_dir=config_dict.get('data_dir', './data'),
            models_dir=config_dict.get('models_dir', './models'),
            configs_dir=config_dict.get('configs_dir', './configs'),
            enable_profiling=config_dict.get('enable_profiling', False),
            profile_output=config_dict.get('profile_output', './data/profiling'),
            memory_limit_mb=config_dict.get('memory_limit_mb', 4096),
            cpu_limit=config_dict.get('cpu_limit', 4),
            enable_rate_limiting=config_dict.get('enable_rate_limiting', True),
            requests_per_minute=config_dict.get('requests_per_minute', 60),
            max_content_size_mb=config_dict.get('max_content_size_mb', 10)
        )
    
    @classmethod
    def save(cls, config_path: str):
        """Сохранение конфигурации в файл"""
        if cls._config is None:
            print("⚠️  Конфигурация не загружена")
            return
        
        try:
            config_dict = {
                'detective': cls._config.detective.__dict__,
                'committee': cls._config.committee.__dict__,
                'analyst': cls._config.analyst.__dict__,
                'storage': cls._config.storage.__dict__,
                'embedding': cls._config.embedding.__dict__,
                'coordinator': cls._config.coordinator.__dict__,
                'log_level': cls._config.log_level,
                'log_file': cls._config.log_file,
                'max_log_size': cls._config.max_log_size,
                'backup_count': cls._config.backup_count,
                'data_dir': cls._config.data_dir,
                'models_dir': cls._config.models_dir,
                'configs_dir': cls._config.configs_dir,
                'enable_profiling': cls._config.enable_profiling,
                'profile_output': cls._config.profile_output,
                'memory_limit_mb': cls._config.memory_limit_mb,
                'cpu_limit': cls._config.cpu_limit,
                'enable_rate_limiting': cls._config.enable_rate_limiting,
                'requests_per_minute': cls._config.requests_per_minute,
                'max_content_size_mb': cls._config.max_content_size_mb
            }
            
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            print(f"✅ Конфигурация сохранена в {config_path}")
            
        except Exception as e:
            print(f"❌ Ошибка сохранения конфигурации: {e}")
    
    @classmethod
    def get(cls) -> SystemConfig:
        """Получение текущей конфигурации"""
        if cls._config is None:
            return cls.load()
        return cls._config
    
    @classmethod
    def update(cls, updates: Dict):
        """Обновление конфигурации"""
        if cls._config is None:
            cls.load()
        
        for key, value in updates.items():
            if hasattr(cls._config, key):
                setattr(cls._config, key, value)
            else:
                # Проверяем вложенные конфигурации
                for config_field in ['detective', 'committee', 'analyst', 'storage', 'embedding', 'coordinator']:
                    if hasattr(getattr(cls._config, config_field), key):
                        setattr(getattr(cls._config, config_field), key, value)
                        break



    def get(self, key, default=None):
        """Получение значения по ключу с поддержкой вложенных атрибутов"""
        if hasattr(self, key):
            return getattr(self, key)
        # Проверяем вложенные конфиги
        for sub in ['detective', 'committee', 'analyst', 'storage', 'embedding', 'coordinator']:
            if hasattr(getattr(self, sub), key):
                return getattr(getattr(self, sub), key)
        return default
    
    def update(self, updates: dict):
        """Обновление конфигурации"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                for sub in ['detective', 'committee', 'analyst', 'storage', 'embedding', 'coordinator']:
                    if hasattr(getattr(self, sub), key):
                        setattr(getattr(self, sub), key, value)
                        break