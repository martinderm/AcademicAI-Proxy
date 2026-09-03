# System Map — Objects (Nomen, Schemas & State)

> **Komponente**: Datenstrukturen, Schnittstellenverträge und Zustandsobjekte  
> **Kontext**: [`README.md`](README.md)

---

## 1. Inbound Request Schemas (Client $\to$ Proxy)

### `POST /v1/chat/completions`
Standard-OpenAI-Payload:
- **`model`** `(str, required)`: Zielmodell (z. B. `gpt-4o`, `gpt-4o-mini`, `academicai-default`).
- **`messages`** `(list[dict], required)`: Liste von Rollen-Objekten:
  - `role`: `system` | `user` | `assistant` | `tool`
  - `content`: `str` oder `list[dict]` (z. B. multimodaler Content / Text-Parts).
  - `tool_calls` `(optional, list)`: Vorherige Tool-Aufrufe bei Assistant-Nachrichten.
  - `tool_call_id` `(optional, str)`: Referenz-ID bei `role: tool`.
- **`tools`** `(list[dict], optional)`: OpenAI-Funktionsdefinitionen:
  - `type`: `"function"`
  - `function`: `{ "name": str, "description": str, "parameters": dict }`
- **`stream`** `(bool, default: False)`: SSE-Streaming aktivieren (`text/event-stream`).
- **`temperature`**, **`max_tokens`**, **`top_p`** `(optional)`: Generierungsparameter.

### `GET /v1/models`
Liefert verfügbare Modelle als OpenAI ModelList:
```json
{
  "object": "list",
  "data": [
    { "id": "gpt-4o", "object": "model", "owned_by": "academicai" },
    { "id": "gpt-4o-mini", "object": "model", "owned_by": "academicai" }
  ]
}
```

---

## 2. Outbound Backend Schemas (Proxy $\to$ BOKU AcademicAI)

Das BOKU-Backend verlangt spezifische Auth-Header und REST-Strukturen ([`academicai/auth.py`](../../academicai/auth.py)):
- **Headers**:
  - `X-Client-ID`: Client-ID aus Umgebungsvariable `ACADEMICAI_CLIENT_ID`.
  - `X-Client-Secret`: Secret aus `ACADEMICAI_CLIENT_SECRET`.
  - `Content-Type`: `application/json`.
- **Payload**:
  - Zusammengeführte Prompt-Struktur, optimiert für das Azure-Prefix-Caching (System-Instruktionen am Kopf der ersten User-Message).

---

## 3. Emulations- & Transformationsobjekte

### Message-Rollen-Normalisierung ([`academicai/transformation.py`](../../academicai/transformation.py))
Das BOKU-Backend unterstützt ausschließlich die Rollen **`user`** und **`assistant`**. Eingehende OpenAI-Rollen werden daher deterministisch transformiert:

| Eingehende OpenAI-Rolle | Zielrolle am BOKU-Backend | Transformationsregel |
| :--- | :--- | :--- |
| `system` | `user` | Inhalt wird an den Beginn der allerersten `user`-Message eingefügt (Sticky System Message / Azure Prefix Caching). |
| `tool` | `user` | Formatiert als `[Tool result (id: <tool_call_id>)]\n<content>`. |
| `assistant` (mit `tool_calls`) | `assistant` | Formatiert als `[Tool call: <name>(<arguments>)]`. |
| Aufeinanderfolgende gleiche Rollen | `user`/`assistant` | Werden zu einer einzigen Nachricht mit `\n\n` zusammengeführt. |

### `ToolCall` (OpenAI-konform synthetisiert)
Wird von [`academicai/tool_emulation.py`](../../academicai/tool_emulation.py) aus dem Modell-Freitext erzeugt:
```json
{
  "id": "call_abc123456",
  "type": "function",
  "function": {
    "name": "exec",
    "arguments": "{\"command\": \"dir\"}"
  }
}
```

### Interne Provider-Dataclasses ([`academicai/provider.py`](../../academicai/provider.py))
- `Message`: `role: str`, `content: str`
- `Choice`: `index: int`, `message: Message`, `finish_reason: str`
- `Usage`: `prompt_tokens: int`, `completion_tokens: int`, `total_tokens: int`
- `CompletionResponse`: `id: str`, `object: str`, `created: int`, `model: str`, `choices: list[Choice]`, `usage: Usage`

### `SkillSnippet` (Zustandsobjekt für Self-Learning)
In `skill_snippets.json` persistierte Heuristik-Tipps:
```json
{
  "id": "himalaya-envelope-list",
  "topics": ["mail", "envelope", "himalaya"],
  "snippet": "Use envelope list -s 50 to search mailboxes efficiently."
}
```

---

## 4. Konfiguration & Runtime State

| Variable / File | Typ | Zweck |
| :--- | :--- | :--- |
| `ACADEMICAI_PROXY_PORT` | `int` (Default: `11435`) | Live-Listen-Port für eingehende Client-Requests |
| `ACADEMICAI_PROXY_API_KEY` | `str (Secret)` | Bearer Token für Client-Authentifizierung am Proxy |
| `ACADEMICAI_CLIENT_ID` | `str (Secret)` | BOKU Backend API Client ID |
| `ACADEMICAI_CLIENT_SECRET` | `str (Secret)` | BOKU Backend API Client Secret |
| `ACADEMICAI_DEBUG_DUMPS` | `bool` | Schreibt Rohdaten nach `last_backend_request.json` |
| `ACADEMICAI_RETRY_MAX` | `int` (Default: `2`) | Maximale Retry-Wiederholungen bei Backend-Netzwerkfehlern |
| `ACADEMICAI_RETRY_BASE_MS` | `int` (Default: `300`) | Basis-Wartezeit für exponentielles Backoff bei Retries |
| `server.pid` | File (`int`) | Prozess-ID des laufenden Proxy-Daemons |
| `server.log` | File | Aktiver Log-Stream mit täglicher Rotation (30 Tage Retention) |
