# AIнура

Локальный кулинарный ассистент для подбора и адаптации рецептов по имеющимся продуктам, порциям и ограничениям пользователя.

Проект команды №52 «Нейрохрючево», НИЯУ МИФИ.

## Возможности

- Генерация рецепта без внешних источников — baseline.
- Поиск рецептов в книжном корпусе и генерация с контекстом — RAG.
- Агентный сценарий с разбором требований, отбором источников, созданием рецепта и проверкой результата.
- Веб-интерфейс с историей запросов, уточнениями и обсуждением рецепта.

Агентный сценарий объединяет пять ролей: координатор, агент требований, агент поиска и отбора, агент рецепта и критик. Роли используют одну локальную модель последовательно.

## Технологии

Python, NumPy, PyTorch, Transformers, Ollama, HTML и JavaScript.

- Генерация: `qwen3:4b-instruct-2507-q4_K_M`.
- Эмбеддинги: `BAAI/bge-m3`.
- Переранжирование: `BAAI/bge-reranker-v2-m3`.
- Корпус: 5943 рецепта из пяти книг.

## Структура

```text
version_1/
  ainura_local/    Поиск, baseline и RAG
  recipe_chunks/  Корпус и метаданные
  webui/          HTTP-сервер и интерфейс
version_2/
  ainura_local/
    agents/       Модули агентов
    agent_runtime.py
    ...           Общие инструменты и тесты
tools/            Проверки кода и корпуса
```

В базовой версии доступны общие инструменты и RAG. Агентный режим подключается отдельными модулями в `version_2/ainura_local/agents/` и контроллером `agent_recipe.py`.

## Установка

Нужны Python 3.12+, Ollama и Node.js 22+ для проверки JavaScript.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Служба Ollama должна быть запущена. На Linux и macOS используйте `.venv/bin/python` вместо `.\.venv\Scripts\python.exe`.

## Подготовка поиска

Построение индекса:

```powershell
.\.venv\Scripts\python.exe -X utf8 version_1/ainura_local/retrieval.py build --chunks version_1/recipe_chunks/chunks.jsonl --index version_1/ainura_local/recipe_index
```

Первый поиск загружает модели поиска в локальный кеш:

```powershell
.\.venv\Scripts\python.exe -X utf8 version_1/ainura_local/retrieval.py search --index version_1/ainura_local/recipe_index --query "Ужин из курицы и риса без молока" --output version_1/ainura_local/search.json
.\.venv\Scripts\python.exe -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')"
```

Индекс, веса моделей и история запросов создаются локально и не хранятся в Git.

## Запуск

```powershell
.\.venv\Scripts\python.exe -X utf8 version_1/webui/server.py
```

Интерфейс: http://127.0.0.1:8765. Он использует заранее загруженные модели и токенизатор. Агентный режим доступен после подключения всех его модулей.

Пример baseline через CLI:

```powershell
.\.venv\Scripts\python.exe -X utf8 version_1/ainura_local/answer_recipe.py --mode baseline --query "Ужин из курицы и риса без молока" --output artifacts/baseline.json
```

## Проверки

Для тестов без моделей и GPU достаточно NumPy и Node.js:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -X utf8 tools/check.py
.\.venv\Scripts\python.exe -X utf8 tools/validate_data.py
```

Проверяются поиск, сборка контекста, структура ответов, сервер и интерфейс. Тесты подключённых агентов запускаются автоматически. Результаты сохраняются в `artifacts/`.

## Ограничения

Внутреннее одобрение критика не заменяет независимую оценку качества. Точные КБЖУ, стоимость блюда и безопасность при аллергиях требуют отдельного справочника и проверки.
