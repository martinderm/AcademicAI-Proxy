# OpenAI Codex Integration Guide

Dieser Leitfaden beschreibt die Integration und Konfiguration der [OpenAI Codex CLI](https://github.com/openai/codex) sowie der **Codex Desktop App** mit dem **AcademicAI Proxy**.

---

## 1. Überblick & Architektur

Aktuelle Versionen von OpenAI Codex (ab `v0.150+`, z. B. `v0.153.4`) unterstützen für Custom Model Provider ausschließlich die modernere **OpenAI Responses API** (`wire_api = "responses"`). Ältere Integrationen über `wire_api = "chat"` bzw. eine reine `/v1/chat/completions`-Schnittstelle werden von aktuellen Codex-Versionen für Custom Provider nicht mehr unterstützt.

### Koexistenz: Bestehende API bleibt unverändert
Die bisherigen Endpunkte bleiben zu 100 % erhalten und voll funktionsfähig:
- `GET /health`
- `GET /internal/cost-status`
- `GET /v1/models`
- `POST /v1/chat/completions`

Bestehende Clients (z. B. OpenClaw, Copilot, Cursor, Continue) können `/v1/chat/completions` weiterhin ohne jegliche Anpassung verwenden.

### Architektur

```text
                         ┌────────────────────────┐
                         │ Existing API clients   │
                         │ (OpenClaw, Cursor etc) │
                         └───────────┬────────────┘
                                     │
                         /v1/chat/completions
                                     │
                                     ▼
┌───────────────┐          ┌─────────────────────┐
│ Codex         │          │ AcademicAI-Proxy    │
│ CLI/Desktop   │─────────▶│                     │
└───────────────┘ responses│ shared core logic   │
                /v1/responses tool emulation     │
                           │ transformations     │
                           └──────────┬──────────┘
                                      │
                                      ▼
                              AcademicAI Backend
```

Beide Schnittstellen teilen sich intern dieselbe kanonische Pipeline (`_execute_completion_pipeline` in `academicai/app.py`), wodurch Tool-Emulation, Prompt-Prefix-Caching und Request-Guards zentral und konsistent angewendet werden. Das Grundprinzip entspricht dem offiziellen `codex-responses-api-proxy` von OpenAI.

---

## 2. Konfiguration (`config.toml`)

Codex liest seine Provider- und Modellkonfiguration aus:
- **Windows:** `%USERPROFILE%\.codex\config.toml` (oder `$env:CODEX_HOME\config.toml`)
- **macOS / Linux:** `~/.codex/config.toml` (oder `$CODEX_HOME/config.toml`)

### Konfigurationsbeispiel

```toml
model = "gpt-4o"
model_provider = "academicai"

[model_providers.academicai]
name = "AcademicAI"
base_url = "http://127.0.0.1:11435/v1"
wire_api = "responses"
env_key = "ACADEMICAI_PROXY_API_KEY"
supports_websockets = false
```

> [!TIP]
> **Projekt-Trust-Level:**  
> Wenn du Codex im vollautomatischen Modus ohne manuelle Bestätigung jedes Tool-Aufrufs betreiben möchtest, kannst du dein Projektverzeichnis als `trusted` einstufen:
> ```toml
> [projects.'d:\programs\academicai-proxy']
> trust_level = "trusted"
> ```

---

## 3. Umgebungsvariablen & Ausführung

Stelle sicher, dass der AcademicAI Proxy läuft (`pwsh -File .\start_server.ps1` oder `py server.py`).

Setze deinen Proxy-API-Key als Umgebungsvariable:

### PowerShell (Windows)
```powershell
$env:ACADEMICAI_PROXY_API_KEY = "dein-proxy-schluessel-aus-der-env"
codex exec "Sag genau: OK"
```

### Bash / Zsh (Linux / macOS)
```bash
export ACADEMICAI_PROXY_API_KEY="dein-proxy-schluessel-aus-der-env"
codex exec "Sag genau: OK"
```

---

## 4. Modelle auswählen & Custom Model Catalog

### Festes Modell in `config.toml`
Das Standardmodell für Codex wird direkt definiert:
```toml
model = "gpt-4o"
model_provider = "academicai"
```

### Modellwechsel per CLI
In der Codex CLI kann das Modell pro Befehl überschrieben werden:
```powershell
codex exec -m gpt-4o-mini "Erstelle eine kurze Zusammenfassung der aktuellen git commits."
```

Unterstützt werden alle im AcademicAI-Tenant aktiven Modelle (z. B. `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `gpt-5-mini`, `claude-opus-4-8`, `gemini-3.5-flash`, `sonar-pro` etc.).

### Custom Model Catalog (`model_catalog_json`)
Codex unterstützt optional einen lokalen Katalog zur Registrierung von Modellen im UI-Picker:
```toml
model_catalog_json = "/path/to/models.json"
```

> [!NOTE]
> `GET /v1/models` allein führt in aktuellen Codex-Versionen noch nicht dazu, dass alle dynamischen Modelle eines Custom Providers automatisch im Desktop-Picker auftauchen. Custom Model Catalogs weisen derzeit in der Codex Desktop App noch einige UI- und Metadaten-Eigenheiten auf.

---

## 5. Codex Desktop App: Besonderheiten & Best Practices

Die Codex Desktop App kann vollständig über den AcademicAI Proxy betrieben werden. Dabei stellt die App gewohnte Agentenfunktionen wie Repository-Zugriff, Dateibearbeitung und lokale Sandbox-Tools bereit, während der Proxy die Modellkommunikation übernimmt.

Folgende bekannte Eigenheiten der aktuellen Codex Desktop App sollten beachtet werden:

### 1. Umschalten zwischen OpenAI und AcademicAI
Der interne Desktop-Modell-Picker verbindet eine Modellauswahl derzeit **nicht** automatisch mit einem Wechsel des `model_provider`.
- **Funktioniert zuverlässig:** Wechsel zwischen verschiedenen AcademicAI-Modellen (wenn `model_provider = "academicai"` aktiv ist) bzw. zwischen verschiedenen OpenAI-Modellen (wenn `model_provider = "openai"` aktiv ist).
- **Einschränkung:** Ein direkter Wechsel zwischen OpenAI (offiziell) und AcademicAI-Proxy über denselben UI-Picker ohne Neustart/Config-Änderung wird von der Codex Desktop UI derzeit noch nicht sauber unterstützt.

### 2. Providerwechsel erfordert App- / Daemon-Neustart
Der Codex App Server behält für bereits geladene Sessions die Provider-Auswahl im Arbeitsspeicher bei. Wenn du `model_provider` in `config.toml` änderst:
1. `config.toml` bearbeiten (`model_provider` umstellen).
2. Codex Desktop App schließen und neu starten bzw. den Daemon neu starten:
   ```powershell
   codex app-server daemon restart
   ```
3. Einen neuen Chat mit dem gewählten Provider beginnen.

### 3. Thread-History nach Providerwechsel
Codex speichert den verwendeten Provider pro Chat-Thread in seiner internen Datenbank:
- Beim Wechsel zwischen OpenAI und einem Custom Provider kann es vorkommen, dass Threads des jeweils anderen Providers in der Seitenleiste scheinbar verschwinden.
- **Wichtig:** Die Chat-Daten werden **nicht gelöscht**! Sie werden lediglich durch die aktuelle Provider-Filterung der UI ausgeblendet und erscheinen wieder, sobald der entsprechende Provider wieder aktiv ist. Der Proxy manipuliert daher bewusst keine lokalen Codex-Datenbanken oder Threads.

---

## 6. Codex-Agent-Funktionen & Multi-Turn Tool Roundtrip

Codex ist ein vollwertiger Coding-Agent. Der Proxy unterstützt den kompletten interaktiven Werkzeug-Zyklus:

```text
Codex Request (Prompt + Tools)
         ↓
POST /v1/responses (AcademicAI-Proxy)
         ↓
Model generiert Tool-Call
         ↓
SSE-Event: response.output_item.added (type: function_call)
         ↓
Codex führt Tool lokal in Sandbox aus (z. B. exec_command)
         ↓
Follow-up Request mit function_call_output
         ↓
POST /v1/responses (normalisiert zu <tool_result>)
         ↓
Model generiert finale Antwort oder weiteren Tool Call
```

### Unterstützte Responses-API-Features
- **Strukturierter Input:** Normalisierung von `type: "message"`, `developer`-Rollen, `input_text`-Parts, `function_call` und `function_call_output`.
- **Streaming (SSE):** Vollständige Event-Sequenz (`response.created`, `response.in_progress`, Text- und Argument-Deltas, `response.output_item.done`, `response.completed`).
- **Stabile Call-IDs:** Eindeutige ID-Zuordnung (`call_id`) für fehlerfreie Multi-Turn-Tool-Zuweisung.
- **Token Accounting:** Das `response.completed`-Event liefert die vom Rust-Deserializer der Codex CLI strikt vorausgesetzten Felder `input_tokens` und `output_tokens` sowie die OpenAI-kompatiblen Felder `prompt_tokens`, `completion_tokens` und `total_tokens`.

---

## 7. Funktionsmatrix

| Funktion | Status | Hinweise |
| :--- | :---: | :--- |
| `POST /v1/chat/completions` | ✅ | Für alle bestehenden Clients unverändert aktiv |
| `GET /v1/models` | ✅ | Liefert dynamische Modellliste aus dem Backend |
| `POST /v1/responses` | ✅ | Vollständig implementiert für Codex CLI & Desktop |
| Bestehende Clients weiterverwenden | ✅ | Keine Codeänderung bei bestehenden Integrationen nötig |
| Codex CLI mit AcademicAI | ✅ | Vollständig unterstützt (`wire_api = "responses"`) |
| Codex Desktop mit AcademicAI | ✅ | Unterstützt via `config.toml` |
| Codex Tool- / Agent-Roundtrip | ✅ | Lokale Ausführung in Codex mit `function_call_output` |
| Festes AcademicAI-Modell | ✅ | Konfigurierbar via `model = "..."` |
| Custom Model Catalog | ⚠️ | Möglich via `model_catalog_json`, Desktop-UI noch im Reifeprozess |
| OpenAI ↔ AcademicAI im UI-Picker | ❌ | Desktop-Picker schaltet Provider nicht dynamisch um |
| Providerwechsel via `config.toml` | ✅ | Empfohlen mit anschließendem Daemon-/App-Neustart |
| Chat-History nach Providerwechsel | ⚠️ | Bekannte Codex-Filterung; Daten bleiben erhalten |

---

## 8. Upstream-Quellen & Referenzen

- **OpenAI Codex Repository:** [github.com/openai/codex](https://github.com/openai/codex)
- **Codex Model Provider Info (Responses-only):** [`codex-rs/model-provider-info/src/lib.rs`](https://github.com/openai/codex/blob/main/codex-rs/model-provider-info/src/lib.rs)
- **OpenAI Responses API Proxy:** [`codex-rs/responses-api-proxy/README.md`](https://github.com/openai/codex/blob/main/codex-rs/responses-api-proxy/README.md)
- **Codex App Server:** [`codex-rs/app-server/README.md`](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)
- **Codex App Server Daemon:** [`codex-rs/app-server-daemon/README.md`](https://github.com/openai/codex/blob/main/codex-rs/app-server-daemon/README.md)
- **Desktop Model Picker & Custom Provider:** [Issue #29156](https://github.com/openai/codex/issues/29156)
- **Custom Profiles im Desktop Picker:** [Issue #22160](https://github.com/openai/codex/issues/22160)
- **Providerwechsel und Desktop-History:** [Issue #31625](https://github.com/openai/codex/issues/31625)
- **Providerwechsel und lokale Sessions:** [Issue #15494](https://github.com/openai/codex/issues/15494)
