.PHONY: help install test run docker-up docker-down docker-build deploy

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  run         - Run application in development mode"
	@echo "  docker-up   - Start docker-compose"
	@echo "  docker-down - Stop docker-compose"
	@echo "  docker-build - Build docker images"
	@echo "  deploy      - Deploy to production"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=app --cov-report=html

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker-compose -f docker-compose.yml up -d

docker-down:
	docker-compose -f docker-compose.yml down

docker-build:
	docker-compose -f docker-compose.yml build

docker-logs:
	docker-compose -f docker-compose.yml logs -f

deploy:
	@echo "Deploying to production..."
	# Add your deployment commands here