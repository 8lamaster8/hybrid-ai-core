"""
Настройка Grafana dashboard и Prometheus
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class DashboardGenerator:
    """Генератор конфигурации Grafana dashboard"""
    
    def __init__(self, dashboard_title: str = "AI Assistant Monitoring"):
        self.dashboard_title = dashboard_title
        self.panels = []
        
        # Определяем панели по умолчанию
        self._create_default_panels()
    
    def _create_default_panels(self):
        """Создание панелей по умолчанию"""
        # 1. Questions Processing Rate
        self.panels.append({
            "title": "Questions Processing Rate",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(ai_questions_processed_total[5m])",
                    "legendFormat": "Questions/sec"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
        })
        
        # 2. Average Processing Time
        self.panels.append({
            "title": "Average Processing Time",
            "type": "graph", 
            "targets": [
                {
                    "expr": "rate(ai_question_processing_seconds_sum[5m]) / rate(ai_question_processing_seconds_count[5m])",
                    "legendFormat": "Seconds"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
        })
        
        # 3. Cache Hit Ratio
        self.panels.append({
            "title": "Cache Hit Ratio",
            "type": "singlestat",
            "targets": [
                {
                    "expr": "ai_cache_hits_total / (ai_cache_hits_total + ai_cache_misses_total)",
                    "format": "percent"
                }
            ],
            "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
        })
        
        # 4. Active Sessions
        self.panels.append({
            "title": "Active Sessions",
            "type": "singlestat",
            "targets": [{"expr": "ai_active_sessions"}],
            "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8}
        })
        
        # 5. Knowledge Base Size
        self.panels.append({
            "title": "Knowledge Base Size",
            "type": "singlestat",
            "targets": [{"expr": "ai_knowledge_chunks_total"}],
            "gridPos": {"h": 4, "w": 6, "x": 12, "y": 8}
        })
        
        # 6. Error Rate
        self.panels.append({
            "title": "Error Rate",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(ai_errors_total[5m])",
                    "legendFormat": "Errors/sec"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12}
        })
        
        # 7. Memory Usage
        self.panels.append({
            "title": "Memory Usage",
            "type": "graph",
            "targets": [
                {
                    "expr": "ai_memory_usage_bytes / 1024 / 1024",
                    "legendFormat": "MB"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12}
        })
        
        # 8. Request Duration
        self.panels.append({
            "title": "Request Duration",
            "type": "heatmap",
            "targets": [
                {
                    "expr": "ai_request_duration_seconds_bucket",
                    "legendFormat": "Duration"
                }
            ],
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 20}
        })
    
    def add_custom_panel(self, panel: Dict[str, Any]):
        """
        Добавление пользовательской панели
        
        Args:
            panel: Конфигурация панели
        """
        self.panels.append(panel)
    
    def generate_dashboard_json(self) -> Dict[str, Any]:
        """Генерация JSON конфигурации для Grafana"""
        dashboard = {
            "dashboard": {
                "title": self.dashboard_title,
                "tags": ["ai", "assistant", "monitoring"],
                "timezone": "browser",
                "panels": self.panels,
                "time": {
                    "from": "now-6h",
                    "to": "now"
                },
                "refresh": "30s"
            },
            "folderId": 0,
            "overwrite": True
        }
        
        return dashboard
    
    def save_to_file(self, output_path: Path):
        """
        Сохранение конфигурации в файл
        
        Args:
            output_path: Путь для сохранения
        """
        dashboard_json = self.generate_dashboard_json()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard_json, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dashboard configuration saved to {output_path}")


def generate_prometheus_config() -> str:
    """Генерация конфигурации Prometheus"""
    config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093

# Load rules once and periodically evaluate them according to the global 'evaluation_interval'.
rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'ai_assistant'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/api/v1/system/metrics'
    scrape_interval: 15s
    
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
"""
    return config


def create_datasource_config() -> Dict[str, Any]:
    """Создание конфигурации источника данных для Grafana"""
    return {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "Prometheus",
                "type": "prometheus",
                "access": "proxy",
                "url": "http://prometheus:9090",
                "isDefault": True,
                "jsonData": {
                    "timeInterval": "15s"
                }
            }
        ]
    }


def create_alert_rules() -> Dict[str, Any]:
    """Создание правил алертов"""
    return {
        "groups": [
            {
                "name": "ai_assistant_alerts",
                "rules": [
                    {
                        "alert": "HighErrorRate",
                        "expr": "rate(ai_errors_total[5m]) > 0.1",
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "High error rate detected",
                            "description": "Error rate is above 0.1 per second for 5 minutes"
                        }
                    },
                    {
                        "alert": "HighMemoryUsage",
                        "expr": "ai_memory_usage_bytes > 2e9",  # 2GB
                        "for": "5m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "High memory usage",
                            "description": "Memory usage is above 2GB"
                        }
                    },
                    {
                        "alert": "NoQuestions",
                        "expr": "rate(ai_questions_processed_total[1h]) == 0",
                        "for": "1h",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "No questions processed",
                            "description": "No questions have been processed for 1 hour"
                        }
                    }
                ]
            }
        ]
    }


# Утилиты для быстрого создания конфигураций
def setup_monitoring(output_dir: Path):
    """
    Настройка всей системы мониторинга
    
    Args:
        output_dir: Директория для сохранения конфигураций
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Создаем конфигурацию Prometheus
    prometheus_config = generate_prometheus_config()
    (output_dir / "prometheus.yml").write_text(prometheus_config)
    
    # 2. Создаем конфигурацию Grafana datasource
    datasource_config = create_datasource_config()
    with open(output_dir / "datasource.yml", 'w', encoding='utf-8') as f:
        yaml.dump(datasource_config, f, default_flow_style=False)
    
    # 3. Создаем dashboard
    generator = DashboardGenerator()
    generator.save_to_file(output_dir / "dashboard.json")
    
    # 4. Создаем правила алертов
    alert_rules = create_alert_rules()
    with open(output_dir / "alerts.yml", 'w', encoding='utf-8') as f:
        yaml.dump(alert_rules, f, default_flow_style=False)
    
    print(f"✅ Monitoring configuration generated in {output_dir}")