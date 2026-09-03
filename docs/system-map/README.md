# AcademicAI Proxy — System Map

> **Typ**: ICM Form 6 (`system-map`)  
> **Ziel**: Kompakte, agentenlesbare Architekturkarte des `AcademicAI-Proxy` zur Vermeidung von Context-Bloat und Attention Drift bei Refactorings und Feature-Erweiterungen.  
> **Gültig für**: `d:\Programs\AcademicAI-Proxy`

---

## 1. Systemübersicht

Der **AcademicAI Proxy** ist ein lokaler HTTP-Reverse-Proxy (FastAPI/Uvicorn), der das universitäre Backend der BOKU (AcademicAI) als OpenAI-kompatible API auf Port `11435` bereitstellt. Er emuliert Function Calling / Tool Calling für Agenten-Harnesses (wie OpenClaw, Antigravity, Codex), da das Backend selbst keine nativen Tools unterstützt.

```
┌──────────────────────────────────────────────┐
│ Clients (OpenClaw, Antigravity, Curated UI)  │
└──────────────────────┬───────────────────────┘
                       │ OpenAI-kompatibler HTTP / SSE
                       ▼
┌──────────────────────────────────────────────┐
│ AcademicAI Proxy (Port 11435 / Test: 11436)  │
│  ├─ server.py (FastAPI, Auth, Streaming)     │
│  └─ academicai/ (Emulation, Transform, Sec)  │
└──────────────────────┬───────────────────────┘
                       │ BOKU REST API + Azure Prefix Cache
                       ▼
┌──────────────────────────────────────────────┐
│ BOKU AcademicAI Backend (Azure OpenAI Stack) │
└──────────────────────────────────────────────┘
```

---

## 2. Navigationsmatrix der System Map

Die Architektur ist nach dem dreidimensionalen ICM-Kartenmodell (`objects`, `processes`, `effects`) strukturiert:

| Dimension | Dokument | Inhalt |
| :--- | :--- | :--- |
| **Nomen** (Struktur & Zustand) | [`objects.md`](objects.md) | Datenmodelle, Inbound-/Outbound-Schemas, Token-/Error-Typen, Config-Strukturen |
| **Verben** (Ablauf & Transformation) | [`processes.md`](processes.md) | Request-Lifecycle, Tool-Emulation-Pipeline, SSE-Streaming-Generator, Heuristiken |
| **Seiteneffekte** (Umwelt & Grenzen) | [`effects.md`](effects.md) | BOKU-Netzwerkcalls, Port-Bindings (11435/11436), Daily Log Rotation, File-Locks |

---

## 3. Architektur-Invarianten für Agenten

1. **Tool-Emulation ist heuristisch:** Tool-Aufrufe werden über Prompt-Injektion und JSON-Extraktion emuliert ([`academicai/tool_emulation.py`](../../academicai/tool_emulation.py)), nicht deterministisch im Modellkern.
2. **Port-Isolation beachten:** Der reguläre Service bindet `11435`. Automatische Tests laufen isoliert auf Port `11436` ([`run_local_tests.ps1`](../../run_local_tests.ps1)).
3. **Safety-Guard für destruktive Mails:** Batch-Tool-Calls, die Mails löschen oder verschieben (`exec` mit `message delete/move`), werden blockiert, wenn in derselben Batch nicht zuvor ein `write`/`edit` stattfand.
4. **Azure Prefix Caching:** System-Prompts werden an den Kopf der ersten User-Message gemergt, um den Cache-Hit am BOKU-Azure-Backend zu sichern.
