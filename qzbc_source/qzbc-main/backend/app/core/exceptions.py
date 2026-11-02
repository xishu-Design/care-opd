from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.core.responses import error_response


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            return JSONResponse(
                status_code=exc.status_code,
                content=error_response(
                    exc.detail.get("error_code", "http_error"),
                    exc.detail.get("message", str(exc.detail)),
                    exc.detail.get("details", {}),
                ),
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response("http_error", str(exc.detail)),
        )

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(_: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_response("validation_error", "Request validation failed", {"errors": exc.errors()}),
        )
