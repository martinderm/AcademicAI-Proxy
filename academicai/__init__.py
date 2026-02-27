"""
AcademicAI — OpenAI-kompatibler Client ohne LiteLLM.

Schnellstart:
    from academicai import completion, get_models

    models = get_models()
    response = completion(model="gpt-4o", messages=[{"role": "user", "content": "Hallo!"}])
    print(response.choices[0].message.content)
"""

from .provider import AcademicAIProvider, CompletionResponse

_provider: AcademicAIProvider | None = None


def _get_provider() -> AcademicAIProvider:
    global _provider
    if _provider is None:
        _provider = AcademicAIProvider()
    return _provider


def completion(model: str, messages: list, **kwargs) -> CompletionResponse:
    """
    AcademicAI Chat Completion.

    Args:
        model:    Modellname, z.B. "gpt-4o", "gpt-5", "Mistral-Large-3"
        messages: Liste von {"role": ..., "content": ...}
        **kwargs: Optional: temperature, max_tokens, max_completion_tokens,
                  frequency_penalty, presence_penalty, response_format,
                  reasoning_effort, verbosity, seed, stop,
                  extra_body={"tailoredAiId": "..."} für RAG

    Returns:
        CompletionResponse mit .choices[0].message.content etc.
    """
    return _get_provider().completion(model=model, messages=messages, optional_params=kwargs)


def get_models() -> dict:
    """
    Listet verfügbare AcademicAI-Modelle.
    Gibt OpenAI-kompatibles /v1/models Format zurück.
    """
    return _get_provider().get_models()


from .tool_emulation import (  # noqa: F401
    inject_tools_into_messages,
    parse_tool_call,
    strip_tool_call_tag,
    build_tool_calls_response,
    build_tool_calls_sse_chunks,
)

__all__ = [
    "AcademicAIProvider", "CompletionResponse", "completion", "get_models",
    "inject_tools_into_messages", "parse_tool_call", "strip_tool_call_tag",
    "build_tool_calls_response", "build_tool_calls_sse_chunks",
]
