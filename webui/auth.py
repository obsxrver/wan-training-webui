import os
import secrets
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .config import (
    AUTH_COOKIE_MAX_AGE,
    AUTH_COOKIE_NAME,
    AUTH_QUERY_PARAM,
    JUPYTER_PORT_ENV_VAR,
    PUBLIC_IP_ENV_VAR,
    TOKEN_ENV_VAR,
    VAST_ENV_VARS,
)


def _load_auth_token() -> str:
    token = os.environ.get(TOKEN_ENV_VAR)
    if token:
        return token
    generated = secrets.token_hex(32)
    print(
        "[webui] JUPYTER_TOKEN environment variable not set. "
        "Generated temporary token for this process: %s" % generated
    )
    return generated


AUTH_TOKEN = _load_auth_token()


def is_vast_instance() -> bool:
    return any(os.environ.get(var) for var in VAST_ENV_VARS)


def build_jupyter_base_url() -> Optional[str]:
    public_ip = os.environ.get(PUBLIC_IP_ENV_VAR)
    port = os.environ.get(JUPYTER_PORT_ENV_VAR)
    if not public_ip or not port:
        return None
    return f"https://{public_ip}:{port}"


class TokenAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, token: str) -> None:
        super().__init__(app)
        self._token = token

    def _set_auth_cookie(self, response, scheme: str) -> None:
        response.set_cookie(
            AUTH_COOKIE_NAME,
            self._token,
            httponly=True,
            secure=scheme == "https",
            samesite="lax",
            path="/",
            max_age=AUTH_COOKIE_MAX_AGE,
            expires=AUTH_COOKIE_MAX_AGE,
        )

    async def dispatch(self, request: Request, call_next):
        if not self._token:
            return await call_next(request)

        cookie_token = request.cookies.get(AUTH_COOKIE_NAME)
        query_token = request.query_params.get(AUTH_QUERY_PARAM)
        token_source = None

        if cookie_token == self._token:
            token_source = "cookie"
        elif query_token == self._token:
            token_source = "query"

        if token_source is None:
            if request.method == "OPTIONS":
                return await call_next(request)
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)

        if token_source == "query" and request.method in {"GET", "HEAD"}:
            redirect_url = request.url.remove_query_params(AUTH_QUERY_PARAM)
            response = RedirectResponse(url=str(redirect_url), status_code=303)
            self._set_auth_cookie(response, redirect_url.scheme)
            return response

        response = await call_next(request)

        if token_source == "query" and cookie_token != self._token:
            self._set_auth_cookie(response, request.url.scheme)
        elif token_source == "cookie":
            self._set_auth_cookie(response, request.url.scheme)

        return response
