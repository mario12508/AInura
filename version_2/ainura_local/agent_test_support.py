"""Small model-free fixtures shared by the five agent contract tests."""
from types import SimpleNamespace
from agent_runtime import validate_shape


class RecordingClient:
    def __init__(self, output, fits=True):
        self.output, self.fits_result = output, fits
        self.requests, self.calls = [], 0

    def fits(self, messages, output_tokens):
        return self.fits_result

    def call(self, role, system, data, schema, output_tokens, validate=None):
        self.requests.append({'role': role, 'data': data, 'schema': schema})
        self.calls += 1
        validate_shape(self.output, schema)
        if validate:
            validate(self.output)
        return self.output


def settings():
    return SimpleNamespace(max_sources=2, candidates=10, num_predict=1536,
                           max_llm_calls=30, max_searches=4, max_repairs=2, index=None)


def state():
    source = {'id': 'a', 'title': 'Рис', 'text': 'Книжный рецепт', 'book': 'Книга'}
    return {'query': 'Рис без молока', 'requirements': {'constraints': [
        {'kind': 'forbidden', 'description': 'Без молока', 'evidence': 'без молока'}],
        'conflicts': [], 'questions': []},
        'candidates': [source], 'selected': [source], 'selection': None,
        'draft': None, 'draft_count': 0, 'draft_version': 0, 'checked_version': -1,
        'context_version': 0, 'draft_context_version': -1, 'critique': None,
        'search_rounds': 1, 'search_queries': ['Рис без молока'], 'last_error': None}
