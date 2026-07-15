import os
import sys
import httpx
from dotenv import load_dotenv

# Load local environment config
load_dotenv()

PROXY_PORT = int(os.environ.get("ACADEMICAI_PROXY_PORT", 11435))
PROXY_KEY = os.environ.get("ACADEMICAI_PROXY_API_KEY")

if not PROXY_KEY:
    print("Error: ACADEMICAI_PROXY_API_KEY is not configured in .env")
    sys.exit(1)

base_url = f"http://127.0.0.1:{PROXY_PORT}"
models_url = f"{base_url}/v1/models"
completions_url = f"{base_url}/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {PROXY_KEY}",
    "Content-Type": "application/json"
}

def test_model(model_name: str, stream: bool) -> tuple[bool, str]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 10,
        "response_format": {"type": "text"},
        "stream": stream
    }
    try:
        resp = httpx.post(completions_url, headers=headers, json=payload, timeout=15.0)
        if resp.status_code == 200:
            if stream:
                if "data: " in resp.text:
                    return True, "OK"
                else:
                    return False, f"Bad stream: {resp.text[:50]}"
            else:
                return True, "OK"
        else:
            try:
                err_detail = resp.json().get("detail", "")
                if "API request failed" in err_detail:
                    return False, "Backend 500"
                return False, f"HTTP {resp.status_code} ({err_detail[:15]})"
            except Exception:
                return False, f"HTTP {resp.status_code}"
    except httpx.TimeoutException:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def main():
    print(f"Connecting to local AcademicAI Proxy on {base_url}...")
    try:
        models_resp = httpx.get(models_url, headers=headers)
        if models_resp.status_code != 200:
            print(f"Error fetching models list: HTTP {models_resp.status_code}")
            sys.exit(1)
        models_data = models_resp.json().get("data", [])
    except Exception as e:
        print(f"Failed to connect to proxy: {e}")
        sys.exit(1)

    if not models_data:
        print("No models returned by proxy.")
        sys.exit(0)

    print(f"Found {len(models_data)} models. Starting verification...\n")
    print(f"{'Model ID':<30} | {'Non-Stream':<25} | {'Stream':<25}")
    print("-" * 88)

    for m in models_data:
        model_name = m.get("id")
        if not model_name:
            continue
        
        # Test non-streaming
        ns_ok, ns_detail = test_model(model_name, stream=False)
        
        # Test streaming
        s_ok, s_detail = test_model(model_name, stream=True)
        
        ns_str = "[OK] " + ns_detail if ns_ok else "[FAIL] " + ns_detail
        s_str = "[OK] " + s_detail if s_ok else "[FAIL] " + s_detail
        
        print(f"{model_name:<30} | {ns_str:<25} | {s_str:<25}")

if __name__ == "__main__":
    main()
