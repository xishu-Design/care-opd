SENSITIVE_FIELDS = ["姓名", "手机号", "地址", "身份证", "精确就诊时间", "医院内部编号"]


def contains_sensitive_value(payload: dict, sensitive_values: list[str]) -> bool:
    text = str(payload)
    return any(value for value in sensitive_values if value and value in text)
