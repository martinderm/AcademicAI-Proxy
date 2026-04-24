import httpx
import pytest

from _e2e import e2e_server


BIG_SYSTEM = (
    "Du bist Dagobert, ein KI-Assistent. "
    "Antworte auf Deutsch. Sei präzise und klar."
)


def _call(e2e_server: dict, body: dict) -> httpx.Response:
    return httpx.post(
        f"{e2e_server['base']}/v1/chat/completions",
        headers=e2e_server["headers"],
        json=body,
        timeout=60,
    )


@pytest.mark.e2e
def test_system_in_messages_streaming(e2e_server):
    response = _call(
        e2e_server,
        {
            "model": e2e_server["model"],
            "stream": True,
            "messages": [
                {"role": "system", "content": BIG_SYSTEM},
                {"role": "user", "content": "Sag kurz Hallo auf Deutsch."},
            ],
        },
    )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text


@pytest.mark.e2e
def test_top_level_system_streaming(e2e_server):
    response = _call(
        e2e_server,
        {
            "model": e2e_server["model"],
            "stream": True,
            "system": BIG_SYSTEM,
            "messages": [{"role": "user", "content": "Sag kurz Hallo auf Deutsch."}],
        },
    )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
