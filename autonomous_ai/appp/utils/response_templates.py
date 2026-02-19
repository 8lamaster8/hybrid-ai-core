"""
Шаблоны для богатых, персонализированных ответов
"""

RESPONSE_TEMPLATES = {
    'mathematical_theorem': {
        'structure': [
            "## 🧮 Теорема: {theorem_name}",
            "",
            "### 📋 Формулировка",
            "{statement}",
            "",
            "### 📐 Математическая запись",
            "```latex",
            "{formulation}",
            "```",
            "",
            "### 📝 Краткое доказательство",
            "{proof_summary}",
            "",
            "### 📜 Историческая справка",
            "{historical_context}",
            "",
            "### 💡 Практическое применение",
            "{applications}",
            "",
            "### 🔗 Связанные понятия",
            "{related_concepts}"
        ],
        'fallback': "К сожалению, не удалось найти полную информацию о данной теореме."
    },
    'historical_event': {
        'structure': [
            "## 📜 Событие: {event_name}",
            "",
            "### ⏳ Хронология",
            "{timeline}",
            "",
            "### 📅 Ключевые даты",
            "{key_dates}",
            "",
            "### 👥 Участники и фигуры",
            "{key_figures}",
            "",
            "### 🔍 Причины и предпосылки",
            "{causes}",
            "",
            "### 🌍 Последствия и значение",
            "{consequences}",
            "",
            "### 💫 Интересные факты",
            "{interesting_facts}"
        ]
    },
    'programming_concept': {
        'structure': [
            "## 💻 {concept_name}",
            "",
            "### 📖 Определение",
            "{definition}",
            "",
            "### 🔧 Синтаксис/Использование",
            "```{language}",
            "{syntax_example}",
            "```",
            "",
            "### 🚀 Практический пример",
            "```{language}",
            "{practical_example}",
            "```",
            "",
            "### 🎯 Когда использовать",
            "{use_cases}",
            "",
            "### ⚖️ Преимущества и недостатки",
            "**✅ Преимущества:** {advantages}",
            "**❌ Недостатки:** {disadvantages}",
            "",
            "### 🔄 Альтернативы",
            "{alternatives}"
        ]
    },
    'scientific_concept': {
        'structure': [
            "## 🔬 {concept_name}",
            "",
            "### 📋 Научное определение",
            "{scientific_definition}",
            "",
            "### ⚙️ Основные принципы",
            "{principles}",
            "",
            "### 📐 Математическое описание",
            "```",
            "{mathematical_description}",
            "```",
            "",
            "### 🔬 Экспериментальные подтверждения",
            "{experimental_evidence}",
            "",
            "### 💡 Области применения",
            "{application_domains}",
            "",
            "### 📊 Современное состояние",
            "{current_state}"
        ]
    },
    'factoid': {
        'structure': [
            "## 📌 {query}",
            "",
            "### ✨ Краткий ответ",
            "{short_answer}",
            "",
            "### 📋 Основные факты",
            "{bullet_points}",
            "",
            #"### 🔗 Источники",
            #"{sources}"
        ]
    },
    'how_why': {
        'structure': [
            "## 🔍 {query}",
            "",
            "### ❓ Причины и объяснения",
            "{explanations}",
            "",
            "### ⚙️ Механизм / Процесс",
            "{mechanism}",
            "",
            "### 📊 Ключевые факторы",
            "{factors}",
            "",
            "### ℹ️ Дополнительная информация",
            "{additional_info}"
        ]
    },
    'evaluation': {
        'structure': [
            "## ⚖️ {query}",
            "",
            "### 🔄 Сравнение",
            "{comparison}",
            "",
            "### ✅ Преимущества",
            "{advantages}",
            "",
            "### ❌ Недостатки",
            "{disadvantages}",
            "",
            "### 💡 Рекомендации",
            "{recommendations}"
        ]
    },
    'default': {
        'structure': [
            "## 📖 {query}",
            "",
            "### 📝 Что мы знаем об этом",
            "{summary}",
            "",
            "### 🔍 Подробности",
            "{details}",
            "",
            "### 📌 Дополнительно",
            "{extra}"
        ]
    }
}


def format_rich_response(template_type: str, data: dict) -> str:
    """
    Форматирует ответ по шаблону с обработкой списков.
    """
    template = RESPONSE_TEMPLATES.get(template_type)
    if not template:
        return data.get('default_answer', '')

    response_lines = []
    
    for line in template['structure']:
        import re
        placeholders = re.findall(r'\{(\w+)\}', line)
        
        if placeholders:
            formatted_line = line
            for placeholder in placeholders:
                value = data.get(placeholder, '')
                
                if isinstance(value, list):
                    # Если список — делаем маркированный список с отступами
                    if value:
                        bullet_items = []
                        for i, item in enumerate(value, 1):
                            if item and isinstance(item, str):
                                # Убираем номера в начале, если они есть
                                clean_item = re.sub(r'^\d+\.\s*', '', item)
                                bullet_items.append(f"  • {clean_item}")
                        value = '\n'.join(bullet_items) if bullet_items else ''
                    else:
                        value = '  • Информация отсутствует'
                        
                elif isinstance(value, str):
                    # Если строка пустая
                    if not value.strip():
                        value = 'Информация отсутствует'
                
                formatted_line = formatted_line.replace(f'{{{placeholder}}}', value)
            
            response_lines.append(formatted_line)
        else:
            response_lines.append(line)

    # Добавляем источники в конец, если есть и не пустые
    #sources = data.get('sources')
    #if sources and isinstance(sources, list) and sources:
    #    response_lines.append("")
    #    response_lines.append("### 🔗 Источники")
    #    for i, src in enumerate(sources[:3], 1):
    #        response_lines.append(f"{i}. {src}")

    return '\n'.join(response_lines)