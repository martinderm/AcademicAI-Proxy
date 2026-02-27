"""
AcademicAI Provider — direkter httpx-Aufruf, kein LiteLLM
"""

import os
import json
import time
import uuid
import logging
from dataclasses import dataclass

import httpx

from .auth import get_base_url, get_headers
from .errors import map_error
from .transformation import build_request_body, build_models_response

log = logging.getLogger("academicai-proxy")

DEBUG_DUMPS = os.environ.get("ACADEMICAI_DEBUG_DUMPS", "false").lower() in ("1", "true", "yes", "on")
MAX_RETRIES = int(os.environ.get("ACADEMICAI_RETRY_MAX", 2))
RETRY_BASE_MS = int(os.environ.get("ACADEMICAI_RETRY_BASE_MS", 300))


# --- Response-Datentypen ---

@dataclass
class Message:
    role: str
    content: str


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: str


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class CompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: list
    usage: Usage


# --- Provider ---

class AcademicAIProvider:
    """
    Direkter Provider für AcademicAI.
    Kein LiteLLM — plain httpx.
    """

    def completion(
        self,
        model: str,
        messages: list,
        optional_params: dict = None,
        timeout: float = 60.0,
    ) -> CompletionResponse:

        optional_params = optional_params or {}

        base_url = get_base_url()
        auth_headers = get_headers()
        endpoint = f"{base_url}/api/v1/llm/chat"

        request_body = build_request_body(model, messages, optional_params)

        log.info(
            f"backend request roles={[m.get('role') for m in request_body.get('messages', [])]} "
            f"body_keys={list(request_body.keys())}"
        )

        # Debug-Dump für Entwicklung (optional)
        if DEBUG_DUMPS:
            debug_path = os.path.join(os.path.dirname(__file__), "..", "last_backend_request.json")
            try:
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(request_body, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        raw = None
        last_exc = None
        with httpx.Client(timeout=timeout) as http_client:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    raw = http_client.post(endpoint, headers=auth_headers, json=request_body)
                except httpx.TimeoutException as e:
                    last_exc = e
                    if attempt < MAX_RETRIES:
                        sleep_s = (RETRY_BASE_MS * (2 ** attempt)) / 1000.0
                        log.warning(f"backend timeout, retrying attempt={attempt + 1}/{MAX_RETRIES} sleep={sleep_s}s")
                        time.sleep(sleep_s)
                        continue
                    raise TimeoutError(f"AcademicAI Request Timeout: {e}") from e
                except httpx.HTTPError as e:
                    last_exc = e
                    if attempt < MAX_RETRIES:
                        sleep_s = (RETRY_BASE_MS * (2 ** attempt)) / 1000.0
                        log.warning(f"backend transport error, retrying attempt={attempt + 1}/{MAX_RETRIES} sleep={sleep_s}s error={e}")
                        time.sleep(sleep_s)
                        continue
                    raise

                # Retry nur bei transients
                if raw.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    sleep_s = (RETRY_BASE_MS * (2 ** attempt)) / 1000.0
                    log.warning(f"backend status={raw.status_code}, retrying attempt={attempt + 1}/{MAX_RETRIES} sleep={sleep_s}s")
                    time.sleep(sleep_s)
                    continue
                break

        if raw is None:
            raise RuntimeError(f"AcademicAI request failed without response: {last_exc}")

        if raw.status_code != 200:
            try:
                err_body = raw.json()
            except Exception:
                err_body = {"message": raw.text}
            log.error(f"backend error {raw.status_code}: {err_body}")
            raise map_error(raw.status_code, err_body)

        data = raw.json().get("data", {})
        content = data.get("content", "")
        finish_reason = data.get("finishReason", "stop")
        usage_data = data.get("usage", {})

        return CompletionResponse(
            id=f"academicai-{uuid.uuid4().hex}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=usage_data.get("promptTokens", 0),
                completion_tokens=usage_data.get("completionTokens", 0),
                total_tokens=usage_data.get("totalTokens", 0),
            ),
        )

    def get_models(self) -> dict:
        """
        Listet verfügbare Modelle vom AcademicAI-Backend.
        """
        base_url = get_base_url()
        auth_headers = get_headers()
        endpoint = f"{base_url}/api/v1/llm/models"

        raw = None
        with httpx.Client(timeout=30.0) as http_client:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    raw = http_client.get(endpoint, headers=auth_headers)
                except httpx.HTTPError as e:
                    if attempt < MAX_RETRIES:
                        sleep_s = (RETRY_BASE_MS * (2 ** attempt)) / 1000.0
                        log.warning(f"models request error, retrying attempt={attempt + 1}/{MAX_RETRIES} sleep={sleep_s}s error={e}")
                        time.sleep(sleep_s)
                        continue
                    raise

                if raw.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    sleep_s = (RETRY_BASE_MS * (2 ** attempt)) / 1000.0
                    log.warning(f"models status={raw.status_code}, retrying attempt={attempt + 1}/{MAX_RETRIES} sleep={sleep_s}s")
                    time.sleep(sleep_s)
                    continue
                break

        if raw is None:
            raise RuntimeError("AcademicAI models request failed without response")

        if raw.status_code != 200:
            try:
                body = raw.json()
            except Exception:
                body = {"message": raw.text}
            raise map_error(raw.status_code, body)

        academic_models = raw.json()
        if isinstance(academic_models, dict):
            academic_models = academic_models.get("data", academic_models)
        if not isinstance(academic_models, list):
            academic_models = [academic_models]

        return build_models_response(academic_models)
