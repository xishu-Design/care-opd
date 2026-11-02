from typing import Any


def success_response(data: Any = None, message: str = "") -> dict[str, Any]:
    return {"success": True, "data": data if data is not None else {}, "message": message}


def error_response(error_code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "success": False,
        "error_code": error_code,
        "message": message,
        "details": details or {},
    }
