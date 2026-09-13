#!/usr/bin/env python3
"""
AcademicAI Connectivity & Model Verification CLI.

Supports:
1. Local Proxy mode (default): tests endpoints on http://127.0.0.1:<PORT> (/v1/models and /v1/chat/completions)
2. Upstream Direct mode (--upstream): tests AcademicAI backend directly (/api/v1/llm/models and /api/v1/llm/chat)
3. Model Listing (--list): lists all registered models with context window and pricing
4. Model Filter (--model <name>): tests only matching models
"""

import argparse
import os
import sys
import time
from typing import Any, Optional, Tuple

import httpx
from dotenv import load_dotenv

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

load_dotenv()


def _extract_error_message(resp: httpx.Response) -> str:
    """Extracts human-readable error from OpenAI, BOKU AcademicAI, or FastAPI error responses."""
    try:
        data = resp.json()
        if isinstance(data, dict):
            # 1. OpenAI-style error: {"error": {"message": "..."}}
            err_obj = data.get("error")
            if isinstance(err_obj, dict) and err_obj.get("message"):
                return str(err_obj["message"])
            if isinstance(err_obj, str):
                return err_obj

            # 2. Upstream BOKU meta error: {"meta": {"error": {"message": "..."}}}
            meta = data.get("meta")
            if isinstance(meta, dict):
                meta_err = meta.get("error")
                if isinstance(meta_err, dict) and meta_err.get("message"):
                    return str(meta_err["message"])

            # 3. FastAPI detail: {"detail": "..."}
            detail = data.get("detail")
            if detail:
                if isinstance(detail, list) and detail:
                    first = detail[0]
                    if isinstance(first, dict) and "msg" in first:
                        return str(first["msg"])
                return str(detail)

            # 4. Generic message: {"message": "..."}
            if data.get("message"):
                return str(data["message"])
    except Exception:
        pass

    text = (resp.text or "").strip()
    if text:
        return text[:80] + ("..." if len(text) > 80 else "")
    return f"HTTP {resp.status_code}"


def _format_costs(costs: list) -> str:
    if not costs or not isinstance(costs, list):
        return "-"
    parts = []
    for c in costs:
        if not isinstance(c, dict):
            continue
        ctype = c.get("costType", "")
        cval = c.get("cost", 0)
        if ctype == "input_tokens":
            parts.append(f"In: {cval * 1000:.2f}€/1M")
        elif ctype == "output_tokens":
            parts.append(f"Out: {cval * 1000:.2f}€/1M")
        elif ctype == "per_request":
            parts.append(f"Req: {cval}€")
    return ", ".join(parts) if parts else "-"


def test_proxy_model(
    completions_url: str,
    headers: dict,
    model_name: str,
    stream: bool,
    timeout: float = 20.0,
) -> Tuple[bool, str]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Antworte mit nur einem Wort: OK"}],
        "temperature": 0.6,
        "stream": stream,
    }
    t0 = time.time()
    try:
        resp = httpx.post(completions_url, headers=headers, json=payload, timeout=timeout)
        dt = round(time.time() - t0, 2)
        if resp.status_code == 200:
            if stream:
                if "data: " in resp.text:
                    return True, f"OK ({dt}s)"
                return False, f"Bad stream: {resp.text[:40]}"
            return True, f"OK ({dt}s)"

        err_msg = _extract_error_message(resp)
        return False, f"HTTP {resp.status_code}: {err_msg}"
    except httpx.TimeoutException:
        dt = round(time.time() - t0, 2)
        return False, f"Timeout after {dt}s"
    except httpx.ConnectError:
        return False, "Connection refused"
    except Exception as e:
        return False, str(e)


def test_upstream_model(
    chat_url: str,
    headers: dict,
    model_name: str,
    stream: bool,
    timeout: float = 25.0,
) -> Tuple[bool, str]:
    from academicai.transformation import build_request_body

    messages = [{"role": "user", "content": "Antworte mit nur einem Wort: OK"}]
    # transformation automatically strips problematic parameters (like max_tokens on gemini/gpt-5)
    body = build_request_body(model_name, messages, {})
    if stream:
        body["stream"] = True

    t0 = time.time()
    try:
        resp = httpx.post(chat_url, headers=headers, json=body, timeout=timeout)
        dt = round(time.time() - t0, 2)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            content = data.get("content", "").strip()
            short_content = (content[:35] + "...") if len(content) > 35 else content
            short_content = short_content.replace("\n", " ")
            return True, f"OK ({dt}s) -> '{short_content}'"

        err_msg = _extract_error_message(resp)
        return False, f"HTTP {resp.status_code}: {err_msg}"
    except httpx.TimeoutException:
        dt = round(time.time() - t0, 2)
        return False, f"Timeout after {dt}s"
    except Exception as e:
        return False, str(e)


def run_upstream(args: argparse.Namespace) -> None:
    from academicai.auth import get_base_url, get_headers

    base_url = get_base_url().rstrip("/")
    headers = get_headers()
    models_url = f"{base_url}/api/v1/llm/models"
    chat_url = f"{base_url}/api/v1/llm/chat"

    print(f"Connecting DIRECTLY to Upstream AcademicAI Backend ({base_url})...")
    try:
        resp = httpx.get(models_url, headers=headers, timeout=15.0)
        if resp.status_code != 200:
            err_msg = _extract_error_message(resp)
            print(f"Error fetching upstream models: HTTP {resp.status_code} ({err_msg})")
            sys.exit(1)
        raw_data = resp.json().get("data", [])
    except Exception as e:
        print(f"Failed to connect to upstream backend: {e}")
        sys.exit(1)

    # Filter models
    models = raw_data
    if args.model:
        query = args.model.lower()
        models = [m for m in models if query in m.get("modelName", "").lower()]

    if not models:
        print(f"No models found matching filter '{args.model}'. Total upstream models: {len(raw_data)}")
        return

    print(f"Found {len(models)} upstream model(s) (of {len(raw_data)} total).\n")

    if args.list:
        print(f"{'Model Name':<26} | {'Context':<10} | {'Output Limit':<12} | {'Pricing (€)':<35}")
        print("-" * 90)
        for m in models:
            name = m.get("modelName", "-")
            ctx = str(m.get("contextWindow", "-"))
            out = str(m.get("outputTokenLimit", "-"))
            costs = _format_costs(m.get("costs", []))
            print(f"{name:<26} | {ctx:<10} | {out:<12} | {costs:<35}")
        return

    test_stream = not args.no_stream
    test_non_stream = not args.stream_only

    print(f"{'Model Name':<26} | {'Direct Chat Test':<60}")
    print("-" * 90)

    for m in models:
        model_name = m.get("modelName")
        if not model_name:
            continue

        if test_non_stream:
            ok, detail = test_upstream_model(chat_url, headers, model_name, stream=False)
            status_tag = "[OK]  " if ok else "[FAIL]"
            print(f"{model_name:<26} | {status_tag} Non-Stream: {detail}")

        if test_stream:
            ok, detail = test_upstream_model(chat_url, headers, model_name, stream=True)
            status_tag = "[OK]  " if ok else "[FAIL]"
            print(f"{model_name:<26} | {status_tag} Stream:     {detail}")


def run_proxy(args: argparse.Namespace) -> None:
    port = args.port or int(os.environ.get("ACADEMICAI_PROXY_PORT", 11435))
    proxy_key = os.environ.get("ACADEMICAI_PROXY_API_KEY")

    if not proxy_key:
        print("Error: ACADEMICAI_PROXY_API_KEY is not configured in .env")
        sys.exit(1)

    base_url = f"http://127.0.0.1:{port}"
    models_url = f"{base_url}/v1/models"
    completions_url = f"{base_url}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {proxy_key}",
        "Content-Type": "application/json",
    }

    print(f"Connecting to Local AcademicAI Proxy on {base_url}...")
    try:
        resp = httpx.get(models_url, headers=headers, timeout=10.0)
        if resp.status_code != 200:
            err_msg = _extract_error_message(resp)
            print(f"Error fetching models list: HTTP {resp.status_code} ({err_msg})")
            sys.exit(1)
        models_data = resp.json().get("data", [])
    except httpx.ConnectError:
        print(f"\n[ERROR] Connection refused: Local proxy is not running on {base_url}.")
        print("-> Start the proxy with: 'python server.py'")
        print("-> Or test upstream directly with: 'python test_models_connectivity.py --upstream'\n")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to connect to proxy: {e}")
        sys.exit(1)

    if not models_data:
        print("No models returned by proxy.")
        return

    # Filter models
    models = models_data
    if args.model:
        query = args.model.lower()
        models = [m for m in models if query in m.get("id", "").lower()]

    if not models:
        print(f"No models found matching filter '{args.model}'. Total proxy models: {len(models_data)}")
        return

    print(f"Found {len(models)} model(s) on proxy (of {len(models_data)} total).\n")

    if args.list:
        print(f"{'Model ID':<30} | {'Context':<10} | {'Max Tokens':<12} | {'Pricing (€)':<35}")
        print("-" * 92)
        for m in models:
            m_id = m.get("id", "-")
            ctx = str(m.get("context_window", "-"))
            out = str(m.get("max_tokens", "-"))
            costs = _format_costs(m.get("costs", []))
            print(f"{m_id:<30} | {ctx:<10} | {out:<12} | {costs:<35}")
        return

    test_stream = not args.no_stream
    test_non_stream = not args.stream_only

    print(f"{'Model ID':<30} | {'Non-Stream':<28} | {'Stream':<28}")
    print("-" * 90)

    for m in models:
        model_name = m.get("id")
        if not model_name:
            continue

        ns_str = "-"
        if test_non_stream:
            ns_ok, ns_detail = test_proxy_model(completions_url, headers, model_name, stream=False)
            ns_str = f"{'[OK] ' if ns_ok else '[FAIL]'} {ns_detail}"

        s_str = "-"
        if test_stream:
            s_ok, s_detail = test_proxy_model(completions_url, headers, model_name, stream=True)
            s_str = f"{'[OK] ' if s_ok else '[FAIL]'} {s_detail}"

        print(f"{model_name:<30} | {ns_str:<28} | {s_str:<28}")


def main():
    parser = argparse.ArgumentParser(
        description="AcademicAI Proxy & Upstream Connectivity Diagnostic Tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-u", "--upstream",
        action="store_true",
        help="Test directly against upstream AcademicAI backend (bypassing local proxy)",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available models with context window and pricing without sending completions",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="",
        help="Filter models by name or substring (e.g. -m gpt-5, -m claude)",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=None,
        help="Custom proxy port (default: from ACADEMICAI_PROXY_PORT or 11435)",
    )
    parser.add_argument(
        "--stream-only",
        action="store_true",
        help="Run only streaming completion tests",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Run only non-streaming completion tests",
    )

    args = parser.parse_args()

    if args.upstream:
        run_upstream(args)
    else:
        run_proxy(args)


if __name__ == "__main__":
    main()
