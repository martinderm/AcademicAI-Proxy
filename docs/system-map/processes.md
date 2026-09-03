# System Map — Processes (Verben, Workflows & Pipelines)

> **Komponente**: Kontrollflüsse, Pipelines und dynamische Transformationen  
> **Kontext**: [`README.md`](README.md)

---

## 1. Request Lifecycle: `POST /v1/chat/completions`

Jeder Chat-Completion-Request durchläuft eine 7-Stufen-Pipeline in [`server.py`](../../server.py):

```
[Inbound Client Request]
       │
       ▼
 1. Authentifizierung & Insecure-Key-Check (Bearer Token)
       │
       ▼
 2. Message-Normalisierung & Heuristiken:
    ├─ _extract_text_content (Plain-Text-Extraktion)
    ├─ _is_human_readable_target (Human Channel vs. Cron)
    ├─ _inject_skill_snippet_context (Skill-Snippets injecten)
    └─ _apply_post_tool_guard (Fehler-Schutz nach Tool-Result)
       │
       ▼
 3. Tool-Injektion (inject_tools_into_messages in academicai/tool_emulation.py)
    └─ Wandelt JSON-Tools in System-Prompt-Instruktionen um
       │
       ▼
 4. BOKU-Backend HTTP-Aufruf (academicai.provider)
    └─ Übertragung mit X-Client-ID / X-Client-Secret & Azure Prefix Cache
       │
       ▼
 5. Parsing & Safety-Filter:
    ├─ parse_tool_calls (Extraktion von ```json ... ``` Calls)
    ├─ _enforce_write_before_mail_delete (Schutz vor unberechtigtem Mail-Löschen)
    └─ _extract_and_learn_tool_usage (Aktualisiert skill_snippets.json)
       │
       ▼
 6. Response-Formatierung:
    ├─ Streaming: SSE-Chunk-Generator (build_tool_calls_sse_chunks)
    └─ Non-Streaming: JSON Response (build_tool_calls_response)
       │
       ▼
[Outbound Client Response]
```

---

## 2. Tool-Emulation Pipeline ([`academicai/tool_emulation.py`](../../academicai/tool_emulation.py))

Da das BOKU-Backend native Tool-Calling-Felder ignoriert, nutzt der Proxy eine synthetische Emulation:

1. **Prompt Injektion (`inject_tools_into_messages`):**  
   Fügt den System-Prompts eine formale Anweisung hinzu: *"To use tools, respond strictly with a ```json { "tool_calls": [...] } ``` code block."*
2. **Extraktion (`parse_tool_calls`):**  
   Durchsucht die Rohantwort des LLM mit Regex-Patterns nach Markdown-JSON-Blöcken, repariert unvollständige JSON-Klammern und isoliert Tool-Aufrufe.
3. **Fallback-Handling:**  
   Falls das Modell JSON ausgibt, obwohl ein Mensch im Chat sitzt (`_is_human_readable_target`), formatiert [`format_arbitrary_json_for_humans`](../../academicai/tool_emulation.py) das JSON in lesbaren Fließtext um.

---

## 3. Post-Tool Result Guard (`_apply_post_tool_guard`)

Verhindert Endlos-Schleifen oder falsche Erfolgsmeldungen:
- **Fehler im Tool-Result (`error:`, `cannot parse`, `failed`):**  
  Injiziert eine System-Warning: *"TOOL_RESULT_ERROR: The latest tool result contains an error. Do NOT claim success. Issue a corrected tool call or report the failure."*
- **Erfolg im Tool-Result:**  
  Injiziert: *"NO_FURTHER_TOOL_CALLS: Produce the final user-facing answer."*

---

## 4. Test- & E2E-Workflow ([`run_local_tests.ps1`](../../run_local_tests.ps1))

1. Startet einen separaten Test-Server auf **Port 11436** (via `start_test_server.ps1`).
2. Führt `pytest` gegen den Test-Port aus, um den aktiven Live-Server auf Port 11435 nicht zu stören.
3. Beendet den Test-Server sauber über dessen PID.
