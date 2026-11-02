from app.llm.output_parser import parse_generated_response_result


def _json(title: str = "标题") -> str:
    return (
        '{"title":"%s","summary":"摘要","sections":[{"heading":"段落","content":"内容","source_ids":["SRC"]}],'
        '"warnings":[],"next_actions":[],"disclaimer":"声明"}'
    ) % title


def test_parser_accepts_direct_json():
    result = parse_generated_response_result(_json())

    assert result.success is True
    assert result.schema_valid is True
    assert result.structured_output["title"] == "标题"


def test_parser_accepts_markdown_json_code_block():
    result = parse_generated_response_result(f"```json\n{_json('代码块')}\n```")

    assert result.success is True
    assert result.structured_output["title"] == "代码块"


def test_parser_extracts_json_from_wrapped_text():
    result = parse_generated_response_result(f"以下是结果：\n{_json('混合文本')}\n请查收。")

    assert result.success is True
    assert result.structured_output["title"] == "混合文本"


def test_parser_reports_non_json_failure():
    result = parse_generated_response_result("这是一段没有 JSON 的自然语言。")

    assert result.success is False
    assert result.schema_valid is False
    assert result.parse_error


def test_parser_reports_schema_failure():
    result = parse_generated_response_result('{"title": 123, "sections": "bad"}')

    assert result.success is False
    assert result.schema_valid is False
    assert result.parse_error.startswith("schema_invalid")


def test_parser_continues_after_an_earlier_schema_invalid_json_object():
    content = '{"note":"explanation"}\n```json\n' + _json("后续合格对象") + "\n```"

    result = parse_generated_response_result(content)

    assert result.success is True
    assert result.schema_valid is True
    assert result.structured_output["title"] == "后续合格对象"


def test_parser_continues_to_later_unfenced_json_object():
    content = '{"title":123,"sections":"bad"}\n' + _json("第二个合格对象")

    result = parse_generated_response_result(content)

    assert result.success is True
    assert result.schema_valid is True
    assert result.structured_output["title"] == "第二个合格对象"


def test_parser_accepts_literal_newline_inside_json_string():
    content = _json("含换行").replace("摘要", "第一行\n第二行")

    result = parse_generated_response_result(content)

    assert result.success is True
    assert result.structured_output["summary"] == "第一行\n第二行"


def test_parser_removes_only_trailing_collection_commas():
    content = _json("尾逗号").replace('"source_ids":["SRC"]}', '"source_ids":["SRC",],}')

    result = parse_generated_response_result(content)

    assert result.success is True
    assert result.structured_output["title"] == "尾逗号"


def test_parser_closes_only_cleanly_unterminated_containers():
    content = _json("缺少尾括号")[:-1]

    result = parse_generated_response_result(content)

    assert result.success is True
    assert result.structured_output["title"] == "缺少尾括号"


def test_parser_does_not_repair_output_ending_inside_string():
    content = _json("字符串截断").split('"disclaimer"', 1)[0] + '"disclaimer":"未完成'

    result = parse_generated_response_result(content)

    assert result.success is False
    assert result.schema_valid is False
