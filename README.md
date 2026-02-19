# hybrid-ai-core
Hybrid AI system with Production Assistant and Self-Learning Core using Engram Graph Memory (NetworkX)

---

# 🤖 Hybrid AI System

[РУС] / [ENG]

---

## 🇷🇺 Русский

### 📦 Состав проекта

#### 🎯 Production Assistant (`/app`) — **ГОТОВО К ИСПОЛЬЗОВАНИЮ**
Продакшен-версия AI ассистента:
- FastAPI + async SQLAlchemy
- Чат с историей диалогов
- Векторная база знаний (ChromaDB)
- Система обратной связи и RL агент
- A/B тестирование ответов
- Prometheus метрики
- Streamlit интерфейс

#### 🧠 Self-Learning Core (`/autonomous_ai`) — **В РАЗРАБОТКЕ**
Экспериментальное ядро с графовой памятью (работает, можно тестировать):
- Engram Graph Memory (NetworkX)
- Долговременная память (Engram DB)
- Knowledge Analyst — анализ знаний
- Quality Committee — оценка качества
- Detective — поиск в интернете
- BGE-M3 эмбеддинги
- Question Generator — генерация вопросов

> ⚡ Базовая функциональность работает, идёт полировка и оптимизация.

### 🚀 Быстрый старт

```bash
# Клонируем
git clone https://github.com/yourname/hybrid-ai-core
cd hybrid-ai-core

# Виртуальное окружение
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate на Windows

# Устанавливаем зависимости
pip install -r app/requirements.txt

# Настройка
cp .env.example .env
# Отредактируйте .env под свои параметры

# Запуск API
cd app
python main.py

📚 API документация

После запуска:

    Swagger UI: http://localhost:8000/docs

    ReDoc: http://localhost:8000/redoc

Основные эндпоинты
Метод	Endpoint	Описание
POST	/api/v1/chat/ask	Задать вопрос
POST	/api/v1/knowledge/add	Добавить знания
GET	/api/v1/knowledge/search	Поиск по знаниям
POST	/api/v1/feedback	Отправить отзыв
GET	/api/v1/system/health	Проверка здоровья
GET	/api/v1/system/metrics	Метрики Prometheus
📊 Мониторинг

    Метрики: http://localhost:8000/api/v1/system/metrics

    Health check: http://localhost:8000/api/v1/system/health

    Streamlit дашборд:
    bash

    streamlit run streamlit_apps/monitor_app.py

    Streamlit чат:
    bash

    streamlit run streamlit_apps/chat_app.py

🐳 Docker
bash

# Запуск всех сервисов
docker-compose -f deployments/production/docker-compose.dev.yml up -d

# Просмотр логов
docker-compose -f deployments/production/docker-compose.dev.yml logs -f

📁 Структура проекта
text

.
├── app/                    # Production Assistant (готово)
│   ├── api/               # Эндпоинты FastAPI
│   ├── core/              # Ядро (brain, config)
│   ├── infrastructure/     # БД, кэш
│   ├── services/          # Бизнес-логика
│   └── monitoring/        # Метрики, хелсчеки
│
├── autonomous_ai/          # Self-Learning Core (в разработке) 🚧
│   ├── appp/
│   │   ├── coordination/  # Координатор сервисов
│   │   ├── services/      # Сервисы в процессе
│   │   └── utils/         # Утилиты
│   └── configs/           # YAML конфиги
│
├── streamlit_apps/         # Фронтенд
├── deployments/            # Docker композы
└── tests/                  # Тесты

⚙️ Конфигурация

Основные переменные в .env:
env

DATABASE_URL=postgresql://user:pass@localhost:5432/ai_core
REDIS_URL=redis://localhost:6379/0
CHROMA_HOST=localhost
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

🇬🇧 English
📦 Project Structure
🎯 Production Assistant (/app) — PRODUCTION READY

Production-ready AI assistant:

    FastAPI + async SQLAlchemy

    Chat with conversation history

    Vector knowledge base (ChromaDB)

    Feedback system with RL agent

    A/B testing for responses

    Prometheus metrics

    Streamlit interface

🧠 Self-Learning Core (/autonomous_ai) — IN DEVELOPMENT

Experimental core with graph memory (working, you can test):

    Engram Graph Memory (NetworkX)

    Long-term memory (Engram DB)

    Knowledge Analyst

    Quality Committee

    Detective (web search)

    BGE-M3 embeddings

    Question Generator

    ⚡ Basic functionality works, currently polishing and optimizing.

🚀 Quick Start
bash

# Clone
git clone https://github.com/yourname/hybrid-ai-core
cd hybrid-ai-core

# Virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r app/requirements.txt

# Configuration
cp .env.example .env
# Edit .env with your settings

# Run API
cd app
python main.py

📚 API Documentation

After starting:

    Swagger UI: http://localhost:8000/docs

    ReDoc: http://localhost:8000/redoc

Main Endpoints
Method	Endpoint	Description
POST	/api/v1/chat/ask	Ask a question
POST	/api/v1/knowledge/add	Add knowledge
GET	/api/v1/knowledge/search	Search knowledge
POST	/api/v1/feedback	Submit feedback
GET	/api/v1/system/health	Health check
GET	/api/v1/system/metrics	Prometheus metrics
📊 Monitoring

    Metrics: http://localhost:8000/api/v1/system/metrics

    Health check: http://localhost:8000/api/v1/system/health

    Streamlit dashboard:
    bash

    streamlit run streamlit_apps/monitor_app.py

    Streamlit chat:
    bash

    streamlit run streamlit_apps/chat_app.py

🐳 Docker
bash

# Start all services
docker-compose -f deployments/production/docker-compose.dev.yml up -d

# View logs
docker-compose -f deployments/production/docker-compose.dev.yml logs -f

📁 Project Structure
text

.
├── app/                    # Production Assistant (ready)
│   ├── api/               # FastAPI endpoints
│   ├── core/              # Brain, config
│   ├── infrastructure/     # DB, cache
│   ├── services/          # Business logic
│   └── monitoring/        # Metrics, health checks
│
├── autonomous_ai/          # Self-Learning Core (WIP) 🚧
│   ├── appp/
│   │   ├── coordination/  # Service coordinator
│   │   ├── services/      # Core services
│   │   └── utils/         # Utilities
│   └── configs/           # YAML configs
│
├── streamlit_apps/         # Frontend
├── deployments/            # Docker compose
└── tests/                  # Tests

⚙️ Configuration

Main environment variables in .env:
env

DATABASE_URL=postgresql://user:pass@localhost:5432/ai_core
REDIS_URL=redis://localhost:6379/0
CHROMA_HOST=localhost
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

📄 License

Apache License 2.0
👨💻 Author

Jin V - initial work

⭐ Star this repo if you find it useful!
💻 Dev Environment Note

The entire stack (Assistant + Self-Learning Core) is currently being developed and tested on an old laptop inside a virtual machine. This seriously limits the speed of graph memory (Engram) training and RL agent performance.

If you like the architecture and want to help move the project from an old laptop to proper hardware for full-scale neural network experiments — I'd be truly grateful for your support!

Support the transition to Bare Metal:

BTC: 13qEwAA1JK3f5zkt51DpM63DmgPwznUkom
TON: UQBIr6VL-S6o5pNr7JcsyhYH0SNOUilLIV2kBaqb3EifupPp
**MEMO (REQUIRED):** 3EifupPp
