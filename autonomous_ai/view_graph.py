#!/usr/bin/env python3
"""
Скрипт для просмотра графа знаний из SQLite, созданного NetworkXGraphService.

Использование:
    python view_graph_nx.py --db ./data/graphs/knowledge_graph.db [options]

Опции:
    --db FILE           путь к SQLite базе (по умолчанию ./data/graphs/knowledge_graph.db)
    --info              показать общую статистику графа
    --nodes             показать список узлов
    --edges             показать список связей
    --topic TOPIC       показать подграф для указанной темы (по имени или ключу)
    --relation TYPE     фильтр связей по типу (contains, is_a, has_part, и т.д.)
    --graph             визуализировать граф (требуется matplotlib)
    --search TEXT       поиск узлов по имени/содержимому
    --weak              показать слабо изученные темы (мало связей)
    --old DAYS          показать темы, не обновлявшиеся более N дней (по умолчанию 7)
    --type NODE_TYPE    фильтр узлов по типу (topic, chunk, key_point, insight, entity, update)
    --limit N           ограничить количество выводимых строк (по умолчанию 50)
    --raw               показать метаданные последнего сохранения (без загрузки графа)
    --stats             показать расширенную статистику (центральность, кластеризация)
    --export FILE       экспортировать граф в JSON файл
    --export-edges FILE экспортировать только связи в JSON
    --export-nodes FILE экспортировать только узлы в JSON
    --show-all          показать всё без ограничения лимита
"""

import argparse
import sqlite3
import pickle
import zlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import os

import networkx as nx

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_graph(db_path: str, compression: bool = True):
    """
    Загружает граф и метаданные из SQLite (последняя сохранённая версия).
    Возвращает (graph, metadata, save_info).
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Проверяем наличие таблицы
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_store'")
    if not c.fetchone():
        conn.close()
        raise ValueError(f"Таблица 'graph_store' не найдена в {db_path}")

    # Берём последнюю запись
    c.execute("SELECT graph, metadata, timestamp, version FROM graph_store ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if not row:
        raise ValueError("Нет данных в таблице graph_store")

    graph_blob, meta_blob, timestamp, version = row

    if compression:
        graph_blob = zlib.decompress(graph_blob)
        meta_blob = zlib.decompress(meta_blob)

    graph = pickle.loads(graph_blob)
    node_metadata = pickle.loads(meta_blob)

    save_info = {
        'timestamp': timestamp,
        'version': version
    }
    return graph, node_metadata, save_info


def print_info(graph: nx.Graph, node_metadata: dict, save_info: dict):
    """Выводит общую информацию о графе."""
    print("="*60)
    print("ГРАФ ЗНАНИЙ")
    print("="*60)
    print(f"Последнее сохранение: {save_info['timestamp']} (версия {save_info['version']})")
    print(f"Узлов: {graph.number_of_nodes()}")
    print(f"Рёбер: {graph.number_of_edges()}")
    print(f"Плотность: {nx.density(graph):.6f}")
    print(f"Компонент связности: {nx.number_connected_components(graph)}")

    # Типы узлов
    node_types = defaultdict(int)
    for _, data in graph.nodes(data=True):
        t = data.get('type', 'unknown')
        node_types[t] += 1
    print("\nТипы узлов:")
    for t, cnt in sorted(node_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {cnt}")

    # Типы рёбер
    edge_types = defaultdict(int)
    for _, _, data in graph.edges(data=True):
        t = data.get('relation', 'unknown')
        edge_types[t] += 1
    print("\nТипы связей (relation):")
    for t, cnt in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {cnt}")


def print_nodes(graph: nx.Graph, limit: int = 50, filter_type: str = None):
    """Выводит список узлов."""
    print(f"\n{'='*60}")
    print(f"УЗЛЫ (первые {limit})")
    if filter_type:
        print(f"Фильтр по типу: {filter_type}")
    print(f"{'='*60}")

    nodes = list(graph.nodes(data=True))
    if filter_type:
        nodes = [(n, d) for n, d in nodes if d.get('type') == filter_type]

    for i, (node, data) in enumerate(nodes[:limit]):
        t = data.get('type', '?')
        name = data.get('name') or data.get('topic') or node
        # Для chunk, insight покажем превью содержимого
        preview = ""
        if t in ('chunk', 'insight', 'key_point', 'update'):
            content = data.get('content', '')
            if content:
                preview = content[:50] + "..." if len(content) > 50 else content
        print(f"{i+1:3}. {node[:50]:<50} type={t:<12} name={name[:30]}")
        if preview:
            print(f"     ↳ {preview}")


def print_edges(graph: nx.Graph, limit: int = 50, topic: str = None, relation_type: str = None):
    """Выводит список рёбер, опционально фильтруя по теме и/или типу связи."""
    print(f"\n{'='*60}")
    if topic and relation_type:
        print(f"СВЯЗИ ТИПА '{relation_type}' ДЛЯ ТЕМЫ '{topic}' (первые {limit})")
    elif topic:
        print(f"СВЯЗИ ДЛЯ ТЕМЫ '{topic}' (первые {limit})")
    elif relation_type:
        print(f"СВЯЗИ ТИПА '{relation_type}' (первые {limit})")
    else:
        print(f"ВСЕ СВЯЗИ (первые {limit})")
    print(f"{'='*60}")

    edges = list(graph.edges(data=True))

    # Фильтр по теме
    if topic:
        topic_node = None
        topic_key = f"topic_{topic}"
        if topic_key in graph:
            topic_node = topic_key
        else:
            for n, d in graph.nodes(data=True):
                if d.get('type') == 'topic' and d.get('name') == topic:
                    topic_node = n
                    break
        if topic_node:
            edges = [e for e in edges if topic_node in (e[0], e[1])]
        else:
            print(f"Тема '{topic}' не найдена.")
            return

    # Фильтр по типу связи
    if relation_type:
        edges = [e for e in edges if e[2].get('relation') == relation_type]

    if not edges:
        print("Нет связей, соответствующих фильтрам.")
        return

    for i, (u, v, data) in enumerate(edges[:limit]):
        rel = data.get('relation', '—')
        weight = data.get('weight', '—')
        print(f"{i+1:3}. {u[:40]:<40} --[{rel}]--> {v[:40]:<40}  (w={weight})")


def visualize_graph(graph: nx.Graph, topic: str = None):
    """Визуализирует граф (или подграф для указанной темы)."""
    if not HAS_MPL:
        print("matplotlib не установлен, визуализация недоступна.")
        return

    if topic:
        # Находим узел темы
        topic_node = None
        topic_key = f"topic_{topic}"
        if topic_key in graph:
            topic_node = topic_key
        else:
            for n, d in graph.nodes(data=True):
                if d.get('type') == 'topic' and d.get('name') == topic:
                    topic_node = n
                    break
        if topic_node:
            # Берём эго‑граф темы (сам узел + его соседи)
            neighbors = list(graph.neighbors(topic_node))
            sub_nodes = [topic_node] + neighbors
            subgraph = graph.subgraph(sub_nodes)
            title = f"Граф для темы '{topic}'"
        else:
            print(f"Тема '{topic}' не найдена.")
            return
    else:
        subgraph = graph
        title = "Полный граф знаний"

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

    # Цвета узлов по типу
    color_map = {
        'topic': 'lightcoral',
        'chunk': 'lightblue',
        'key_point': 'lightgreen',
        'insight': 'gold',
        'entity': 'purple',
        'update': 'orange',
        'unknown': 'gray'
    }
    node_colors = []
    for node in subgraph.nodes:
        t = subgraph.nodes[node].get('type', 'unknown')
        node_colors.append(color_map.get(t, 'gray'))

    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight='bold')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def search_nodes(graph: nx.Graph, text: str, limit: int = 50):
    """Ищет узлы по имени/ключу или содержимому (для текстовых полей)."""
    results = []
    text_lower = text.lower()
    for node, data in graph.nodes(data=True):
        # Поиск по ключу узла
        if text_lower in node.lower():
            results.append((node, data))
            continue
        # Поиск по полям name, topic, content
        for field in ['name', 'topic', 'content']:
            val = data.get(field, '')
            if isinstance(val, str) and text_lower in val.lower():
                results.append((node, data))
                break
        if len(results) >= limit:
            break

    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТЫ ПОИСКА: '{text}' (первые {limit})")
    print(f"{'='*60}")
    if not results:
        print("Ничего не найдено.")
        return

    for i, (node, data) in enumerate(results[:limit]):
        t = data.get('type', '?')
        name = data.get('name') or data.get('topic') or node
        print(f"{i+1:3}. {node[:50]:<50} type={t:<12} name={name[:30]}")


def weak_areas(graph: nx.Graph, limit: int = 5):
    """Показывает слабо изученные темы (на основе степени и давности)."""
    topics = []
    for node, data in graph.nodes(data=True):
        if data.get('type') != 'topic':
            continue
        degree = graph.degree(node)
        # Дата последнего обновления (из атрибута updated_at или created_at из node_metadata?)
        updated = data.get('updated_at')
        if updated:
            try:
                last_date = datetime.fromisoformat(updated)
            except:
                last_date = datetime(2000, 1, 1)
        else:
            last_date = datetime(2000, 1, 1)
        days_old = (datetime.now() - last_date).days
        # Вес слабости: обратная степень + давность
        weakness = (1 / (degree + 1)) * 10 + days_old * 0.1
        topics.append((weakness, node, data.get('name', node), degree, days_old))

    topics.sort(key=lambda x: x[0], reverse=True)
    print(f"\n{'='*60}")
    print(f"СЛАБО ИЗУЧЕННЫЕ ТЕМЫ (первые {limit})")
    print(f"{'='*60}")
    for i, (_, node, name, degree, days) in enumerate(topics[:limit]):
        print(f"{i+1:3}. {name[:50]:<50} степень={degree:<3} дней без обновления={days}")


def old_topics(graph: nx.Graph, days_old: int = 7, limit: int = 10):
    """Показывает темы, не обновлявшиеся более N дней."""
    cutoff = datetime.now() - timedelta(days=days_old)
    old = []
    for node, data in graph.nodes(data=True):
        if data.get('type') != 'topic':
            continue
        updated = data.get('updated_at')
        if not updated:
            continue
        try:
            updated_date = datetime.fromisoformat(updated)
        except:
            continue
        if updated_date < cutoff:
            days = (datetime.now() - updated_date).days
            old.append((days, node, data.get('name', node), updated))

    old.sort(key=lambda x: x[0], reverse=True)
    print(f"\n{'='*60}")
    print(f"ТЕМЫ, НЕ ОБНОВЛЯВШИЕСЯ > {days_old} ДНЕЙ (первые {limit})")
    print(f"{'='*60}")
    for i, (days, node, name, updated) in enumerate(old[:limit]):
        print(f"{i+1:3}. {name[:50]:<50} дней={days:<3} последнее обновление={updated}")


def extended_stats(graph: nx.Graph):
    """Расширенная статистика (центральность, кластеризация)."""
    print(f"\n{'='*60}")
    print("РАСШИРЕННАЯ СТАТИСТИКА")
    print(f"{'='*60}")

    # Средний кластерный коэффициент
    try:
        avg_clust = nx.average_clustering(graph)
        print(f"Средний кластерный коэффициент: {avg_clust:.4f}")
    except:
        print("Не удалось вычислить кластеризацию")

    # Центральность по степени (топ-5)
    if graph.number_of_nodes() > 0:
        cent = nx.degree_centrality(graph)
        top = sorted(cent.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nТоп-5 узлов по центральности (степень):")
        for node, c in top:
            t = graph.nodes[node].get('type', '?')
            name = graph.nodes[node].get('name') or node
            print(f"  {name[:40]:<40} type={t:<10} центральность={c:.4f}")

    # Компоненты связности
    comps = list(nx.connected_components(graph))
    print(f"\nКомпонент связности: {len(comps)}")
    if len(comps) > 1:
        sizes = [len(c) for c in comps]
        print(f"  Размеры: {sorted(sizes, reverse=True)[:5]}")


def print_raw_save_info(db_path: str):
    """Показывает информацию о последнем сохранении без загрузки графа."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_store'")
    if not c.fetchone():
        print(f"Таблица 'graph_store' не найдена в {db_path}")
        conn.close()
        return

    c.execute("SELECT id, timestamp, version, length(graph), length(metadata) FROM graph_store ORDER BY id DESC LIMIT 5")
    rows = c.fetchall()
    conn.close()

    print("="*60)
    print("ИНФОРМАЦИЯ О СОХРАНЕНИЯХ")
    print("="*60)
    for row in rows:
        id, ts, ver, g_len, m_len = row
        print(f"ID: {id}, Время: {ts}, Версия: {ver}, Размер графа: {g_len} байт, Метаданные: {m_len} байт")


def export_to_json(graph: nx.Graph, filename: str, export_type: str = 'all'):
    """
    Экспорт графа в JSON.
    export_type: 'all' - всё, 'nodes' - только узлы, 'edges' - только рёбра
    """
    data = {}
    
    if export_type in ['all', 'nodes']:
        nodes = []
        for node, attrs in graph.nodes(data=True):
            node_data = {
                'id': node,
                'type': attrs.get('type', 'unknown'),
                'attributes': {k: v for k, v in attrs.items() if k != 'type'}
            }
            nodes.append(node_data)
        data['nodes'] = nodes
    
    if export_type in ['all', 'edges']:
        edges = []
        for u, v, attrs in graph.edges(data=True):
            edge_data = {
                'source': u,
                'target': v,
                'relation': attrs.get('relation', 'unknown'),
                'attributes': {k: v for k, v in attrs.items() if k != 'relation'}
            }
            edges.append(edge_data)
        data['edges'] = edges
    
    if export_type == 'all':
        data['stats'] = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': nx.density(graph)
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Экспортировано в {filename}:")
    if export_type in ['all', 'nodes']:
        print(f"   - Узлов: {len(data.get('nodes', []))}")
    if export_type in ['all', 'edges']:
        print(f"   - Связей: {len(data.get('edges', []))}")


def main():
    parser = argparse.ArgumentParser(description="Просмотр графа знаний из SQLite (NetworkXGraphService).")
    parser.add_argument('--db', default='./data/graphs/knowledge_graph.db',
                        help='Путь к SQLite базе графа')
    parser.add_argument('--info', action='store_true', help='Показать общую статистику')
    parser.add_argument('--nodes', action='store_true', help='Показать узлы')
    parser.add_argument('--edges', action='store_true', help='Показать связи')
    parser.add_argument('--topic', help='Фильтр по имени темы (без префикса)')
    parser.add_argument('--relation', help='Фильтр связей по типу (contains, is_a, has_part, и т.д.)')
    parser.add_argument('--graph', action='store_true', help='Визуализировать граф')
    parser.add_argument('--search', help='Поиск по ключу/содержимому узлов')
    parser.add_argument('--weak', action='store_true', help='Показать слабо изученные темы')
    parser.add_argument('--old', type=int, nargs='?', const=7, metavar='DAYS',
                        help='Показать темы, не обновлявшиеся более N дней (по умолчанию 7)')
    parser.add_argument('--stats', action='store_true', help='Расширенная статистика')
    parser.add_argument('--type', dest='node_type', help='Тип узлов для --nodes')
    parser.add_argument('--limit', type=int, default=50, help='Лимит строк')
    parser.add_argument('--raw', action='store_true', help='Показать метаданные сохранений (без загрузки графа)')
    parser.add_argument('--export', metavar='FILE', help='Экспортировать весь граф в JSON файл')
    parser.add_argument('--export-nodes', metavar='FILE', help='Экспортировать только узлы в JSON')
    parser.add_argument('--export-edges', metavar='FILE', help='Экспортировать только связи в JSON')
    parser.add_argument('--show-all', action='store_true', help='Показать всё без ограничения лимита')

    args = parser.parse_args()

    # Если ничего не выбрано, показываем info + nodes + edges
    if not any([args.info, args.nodes, args.edges, args.graph, args.search,
                args.weak, args.old is not None, args.stats, args.raw,
                args.export, args.export_nodes, args.export_edges]):
        args.info = True
        args.nodes = True
        args.edges = True

    # Если запрошен raw, не загружаем граф
    if args.raw:
        try:
            print_raw_save_info(args.db)
        except Exception as e:
            print(f"Ошибка: {e}")
        return

    # Загружаем граф
    try:
        graph, node_metadata, save_info = load_graph(args.db)
    except Exception as e:
        print(f"Ошибка загрузки графа: {e}")
        return

    # Экспорт в JSON (если запрошен)
    if args.export:
        export_to_json(graph, args.export, 'all')
    if args.export_nodes:
        export_to_json(graph, args.export_nodes, 'nodes')
    if args.export_edges:
        export_to_json(graph, args.export_edges, 'edges')
    
    # Если только экспорт, выходим
    if args.export or args.export_nodes or args.export_edges:
        return

    # Определяем лимит
    limit = args.limit if not args.show_all else 1000000

    if args.info:
        print_info(graph, node_metadata, save_info)

    if args.stats:
        extended_stats(graph)

    if args.weak:
        weak_areas(graph, limit=limit)

    if args.old is not None:
        old_topics(graph, days_old=args.old, limit=limit)

    if args.search:
        search_nodes(graph, args.search, limit=limit)

    if args.nodes:
        print_nodes(graph, limit=limit, filter_type=args.node_type)

    if args.edges:
        print_edges(graph, limit=limit, topic=args.topic, relation_type=args.relation)

    if args.graph:
        visualize_graph(graph, topic=args.topic)


if __name__ == '__main__':
    main()



# Все связи типа 'contains'
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --edges --relation contains

# Все связи типа 'has_part' для темы "ии"
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --edges --topic ии --relation has_part

# Статистика по типам связей
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --info

# Экспорт всего графа (узлы + связи + статистика)
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --export graph.json

# Экспорт только узлов
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --export-nodes nodes.json

# Экспорт только связей
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --export-edges edges.json

# Экспорт с фильтром по теме (сначала экспорт, потом можно отфильтровать)
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --export all.json
#jq '.edges[] | select(.relation=="contains")' all.json  # фильтр через jq

# Все связи без ограничения
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --edges --show-all

# Все узлы типа topic
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --nodes --type topic --show-all


# Посмотреть все связи типа 'contains'
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --edges --relation contains --limit 100

# Посмотреть все связи типа 'has_part' для темы "ии"
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --edges --topic ии --relation has_part

# Экспортировать все связи в JSON для анализа в другой программе
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --export-edges edges.json

# Экспортировать всё и потом анализировать
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --export full.json
# Теперь можно открыть full.json в любом редакторе или программе

# Узнать все типы связей, которые есть в графе
#python view_graph_nx.py --db ./data/graphs/knowledge_graph.db --info | grep "Типы связей" -A 10