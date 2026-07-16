from agmem.llm.structured import coerce_to_schema, extract_json

ITEMS_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array"}},
    "required": ["items"],
}


def test_extract_json_from_code_fence():
    text = 'Here you go:\n```json\n{"a": 1}\n```'
    assert extract_json(text) == {"a": 1}


def test_extract_json_top_level_array():
    text = '```json\n[{"title": "x"}]\n```'
    assert extract_json(text) == [{"title": "x"}]


def test_coerce_bare_array_wrapped_into_single_array_field():
    # observed Qwen3-0.6B failure: bare array instead of {"items": [...]}
    parsed = extract_json('[{"title": "t", "description": "d", "content": "c"}]')
    coerced = coerce_to_schema(parsed, ITEMS_SCHEMA)
    assert coerced == {"items": [{"title": "t", "description": "d", "content": "c"}]}


def test_coerce_ambiguous_array_schema_refused():
    two_arrays = {"type": "object", "properties": {
        "a": {"type": "array"}, "b": {"type": "array"}}}
    assert coerce_to_schema([1, 2], two_arrays) is None


def test_coerce_dict_passthrough():
    assert coerce_to_schema({"x": 1}, ITEMS_SCHEMA) == {"x": 1}
