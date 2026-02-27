# Konzept: Tool-Call-Emulation im AcademicAI Proxy

> Stand: 2026-02-25 · Autor: Dagobert 🦆

## 1. Problemstellung

OpenClaw sendet **jeden** LLM-Request mit einem `tools`-Array (40+ Tool-Definitionen).
Das AcademicAI-Backend lehnt den `tools`-Parameter mit HTTP 400 ab (`"Extra inputs are not permitted"`).

Der Proxy strippt `tools` bereits vor dem Weiterleiten — das Modell antwortet also, aber **ohne Möglichkeit, Tools aufzurufen**. Damit sind academicai-Modelle auf reine Text-Generierung beschränkt: kein `web_search`, kein `exec`, kein `memory_search`, kein Modellwechsel via `session_status`.

### Bisheriger Ansatz (gescheitert)

Tool-Definitionen als Text in den System-Prompt injiziert, Modell aufgefordert `<tool_call>{"name":..., "arguments":...}</tool_call>` auszugeben. **Ergebnis:** Alle getesteten Modelle (gpt-4o, gpt-5, gpt-5-mini, Mistral-Large-3) verweigern die Ausgabe des Tags. Sie beschreiben stattdessen, was sie tun *würden*, oder beantworten die Frage direkt.

### Warum Prompt-Injection nicht funktioniert

1. **RLHF-Training:** GPT- und Mistral-Modelle sind darauf trainiert, Tool-Calls nur über den nativen `tools`-Parameter auszulösen. Ohne registrierte Tools wechseln sie nicht in den "function calling mode".
2. **Safety-Layer:** Die Modelle unterscheiden zwischen "ich habe Tools" und "jemand beschreibt mir Tools im Prompt". Im zweiten Fall verhalten sie sich beratend statt ausführend.
3. **Kein `tool_choice`-Signal:** Ohne nativen `tool_choice`-Parameter fehlt dem Modell das Signal, dass es einen strukturierten Output produzieren soll.

---

## 2. Strategien

### Strategie A: Structured Output via JSON-Mode

**Idee:** Statt XML-Tags den JSON-Mode (`response_format: {type: "json_object"}`) nutzen, um das Modell zu zwingen, strukturierten Output zu produzieren.

**Ablauf:**
1. Proxy erkennt `tools` im Request
2. Tool-Definitionen + Entscheidungsschema als System-Prompt injizieren
3. `response_format: {type: "json_object"}` an Backend senden
4. Modell antwortet mit JSON: `{"action": "tool_call", "name": "...", "arguments": {...}}` oder `{"action": "respond", "content": "..."}`
5. Proxy parst JSON → entweder OpenAI `tool_calls`-Response oder normaler Text-Response

**Vorteile:**
- JSON-Mode ist gut trainiert bei GPT-Modellen
- Erzwingt strukturierten Output (kein Freitext-Escape)
- Einfaches Parsing (kein Regex)

**Risiken:**
- Unklar ob AcademicAI `response_format` unterstützt (muss getestet werden)
- Modell könnte trotzdem `{"action": "respond", ...}` wählen statt Tool zu callen
- Jeder Response wird JSON → Overhead für normale Antworten
- Verschachtelte Tool-Calls (multiple) werden komplex

**Aufwand:** Mittel · **Erfolgswahrscheinlichkeit:** Mittel-Hoch (falls JSON-Mode unterstützt)

---

### Strategie B: Zwei-Pass-Architektur (Router + Executor)

**Idee:** Zwei aufeinanderfolgende LLM-Calls pro User-Request. Erster Call entscheidet *ob* ein Tool nötig ist und welches. Zweiter Call füllt die Parameter oder generiert die Text-Antwort.

**Ablauf:**
1. **Pass 1 — Router:** Komprimierte Frage an das Modell: "Gegeben diese Tools und diese User-Nachricht: Brauchst du ein Tool? Wenn ja, welches? Antworte nur mit dem Tool-Namen oder NONE."
   - Verwendetes Modell: das günstigste (`gpt-5-nano`, $0.06/M Input)
   - Minimaler Prompt → wenige Tokens
2. **Pass 2a — Tool-Call:** Falls Tool erkannt: gezielter Prompt mit nur *einem* Tool-Schema → Parameter extrahieren. Output: JSON mit den Argumenten.
3. **Pass 2b — Text:** Falls NONE: Original-Messages ohne Tool-Injection an das Backend → normale Antwort.

**Vorteile:**
- Router-Call ist billig (~100 Tokens, nano-Modell)
- Zweiter Call hat klaren, engen Scope (ein Tool, Parameter füllen)
- Funktioniert auch ohne `response_format: json_object`
- Gesamtlatenz nur +200-400ms für den Router-Call

**Risiken:**
- Doppelte Latenz bei jedem Request (auch wenn kein Tool gebraucht wird)
- Router kann falsch liegen → entweder verpasster Tool-Call oder unnötiger zweiter Call
- Conversation-State wird komplexer (Proxy muss internen State managen)
- Kosten: ~50% mehr Tokens pro Request (aber nano ist billig)

**Optimierung:** Router-Call nur wenn `tool_choice != "none"`. Bei `tool_choice: "none"` direkt Text-Antwort.

**Aufwand:** Hoch · **Erfolgswahrscheinlichkeit:** Hoch

---

### Strategie C: Regex/Heuristik-basierte Intent-Erkennung

**Idee:** Kein strukturierter Output vom Modell erzwingen. Stattdessen die natürliche Text-Antwort analysieren und Tool-Intents heuristisch erkennen.

**Ablauf:**
1. Tool-Definitionen als Kontext in den System-Prompt (wie bisher)
2. Zusätzliche Instruktion: "Wenn du ein Tool verwenden willst, beschreibe den Aufruf in diesem Format: `TOOL: tool_name(param1=value1, param2=value2)`"
3. Proxy parst die Antwort mit Regex auf `TOOL: ...`-Pattern
4. Match → OpenAI `tool_calls`-Response bauen
5. Kein Match → normaler Text-Response

**Vorteile:**
- Ein einziger LLM-Call
- Minimale Änderung am Prompt
- `TOOL: name(args)` ist einfacher als XML und näher an natürlicher Sprache

**Risiken:**
- Gleiche RLHF-Problematik wie bisheriger Ansatz — Modell könnte `TOOL:` verweigern
- Parsing von Freitext-Argumenten fehleranfällig (Strings mit Kommas, verschachtelte JSON-Werte)
- Kein Zwang zu strukturiertem Output → Modell kann Mischformen produzieren
- Argument-Typen (number, boolean, array) müssen erraten werden

**Aufwand:** Niedrig · **Erfolgswahrscheinlichkeit:** Niedrig (gleiche Root Cause wie bisheriger Ansatz)

---

### Strategie D: Proxy-seitige Tool-Ausführung (Agent-in-the-Middle)

**Idee:** Der Proxy wird selbst zum Mini-Agenten. Er führt bestimmte Tools lokal aus, anstatt sie an OpenClaw zurückzugeben.

**Ablauf:**
1. Proxy erkennt Tool-Intent im Modell-Output (via Strategie A, B oder C)
2. Statt `tool_calls`-Response an OpenClaw: Proxy führt das Tool selbst aus
3. Ergebnis wird als Kontext in einen Folge-Call an AcademicAI gegeben
4. Finale Text-Antwort geht an OpenClaw

**Vorteile:**
- OpenClaw sieht nur Text-Responses → keine Kompatibilitätsprobleme
- Bestimmte Tools sind einfach zu implementieren (`web_search` via Brave API)

**Nachteile:**
- **Dupliziert OpenClaw-Infrastruktur** — jedes Tool muss im Proxy reimplementiert werden
- Sicherheitsmodell wird umgangen (Proxy hat keine Permissions-Checks)
- `exec`, `browser`, `memory_search` etc. sind nicht sinnvoll im Proxy implementierbar
- Proxy wird komplex und fragil
- OpenClaw verliert Kontrolle über Tool-Aufrufe (Logging, Rate-Limiting, Audit)

**Aufwand:** Sehr hoch · **Erfolgswahrscheinlichkeit:** Mittel (nur für Subset von Tools)

**Fazit:** Nicht empfohlen. Widerspricht der Architektur (Proxy = Adapter, nicht Agent).

---

### Strategie E: Akzeptanz + Graceful Degradation

**Idee:** Tool-Calls sind mit AcademicAI nicht möglich. Das wird sauber dokumentiert und OpenClaw so konfiguriert, dass academicai-Modelle nur für passende Use-Cases verwendet werden.

**Ablauf:**
1. Proxy strippt `tools` weiterhin (Status Quo)
2. Keine Tool-Emulation
3. OpenClaw-Konfiguration: academicai-Modelle nicht als Default-Modell (kein Tool-Support)
4. Explizite Nutzung via `session_status(model=academicai/gpt-5)` für reine Text-Tasks
5. Dokumentation: "Kein Tool-Support" in README, TOOLS.md, Agent-Config

**Vorteile:**
- Kein zusätzlicher Code
- Keine Latenz- oder Kosten-Overhead
- Kein Risiko durch fehlgeschlagene Emulation
- Saubere Erwartungshaltung

**Nachteile:**
- academicai-Modelle sind Zweite-Klasse-Bürger in OpenClaw
- Kein `web_search`, `exec` etc. → viele Aufgaben nicht möglich
- Modellwechsel zur Laufzeit via Tool funktioniert nicht (Henne-Ei: man braucht Tools um das Modell zu wechseln)

**Aufwand:** Minimal · **Erfolgswahrscheinlichkeit:** 100% (kein Emulationsversuch)

---

## 3. Bewertung

| Strategie | Aufwand | Latenz | Kosten | Erfolgswahrsch. | Wartbarkeit |
|---|---|---|---|---|---|
| **A: JSON-Mode** | Mittel | +0ms | +0% | Mittel-Hoch¹ | Gut |
| **B: Zwei-Pass** | Hoch | +200-400ms | +30-50% | Hoch | Mittel |
| **C: Regex/Heuristik** | Niedrig | +0ms | +0% | Niedrig | Schlecht |
| **D: Agent-in-Middle** | Sehr hoch | +variabel | +variabel | Mittel | Schlecht |
| **E: Graceful Degrad.** | Minimal | +0ms | +0% | N/A | Sehr gut |

¹ Abhängig davon ob AcademicAI `response_format` unterstützt

---

## 4. Empfehlung

### Kurzfristig: Strategie A testen (JSON-Mode)

**Warum:** Geringster Aufwand mit höchster Chance auf Erfolg. Ein einziger API-Call, keine Architektur-Änderung. Muss nur getestet werden ob `response_format: {type: "json_object"}` vom Backend akzeptiert wird.

**Konkreter Test-Plan:**
1. Request an AcademicAI mit `response_format: {type: "json_object"}` senden
2. Wenn HTTP 400 → Strategie A fällt raus, weiter mit B
3. Wenn akzeptiert: minimalen Prompt testen der Tool-Entscheidung als JSON erzwingt

**Prompt-Entwurf für JSON-Mode:**
```
You have access to the following tools:
- web_search(query: string, count?: number) — Search the web
- memory_search(query: string) — Search memory files
[...]

Respond with a JSON object. Choose ONE:
1. If you need a tool: {"action": "tool_call", "name": "tool_name", "arguments": {"param": "value"}}
2. If you can answer directly: {"action": "respond", "content": "your answer here"}
```

### Fallback: Strategie B (Zwei-Pass)

Falls JSON-Mode nicht unterstützt oder unzuverlässig: Zwei-Pass mit nano als Router. Höhere Komplexität, aber robuster weil der Router-Call ein eng begrenztes Klassifikationsproblem löst (Tool-Name oder NONE) statt freier Text-Generierung.

### Langfristig: Strategie E als Baseline

Unabhängig von A oder B sollte der Proxy graceful degradieren wenn die Emulation fehlschlägt. Kein Hard-Fail, sondern: Tool-Calls werden ignoriert, Modell antwortet als Text. OpenClaw muss damit umgehen können.

---

## 5. Implementierungsplan (bei Strategie A)

### Phase 1: Feasibility (1h)
- [ ] Testen ob `response_format: {type: "json_object"}` von AcademicAI akzeptiert wird
- [ ] Testen ob GPT-5-mini im JSON-Mode zuverlässig `{"action": "tool_call", ...}` produziert
- [ ] Testen mit 3-5 verschiedenen Prompts (eindeutiger Tool-Fall, Grenzfall, kein Tool)

### Phase 2: Integration (2-3h)
- [ ] `tool_emulation.py` überarbeiten: JSON-Mode statt XML-Tags
- [ ] `server.py`: `response_format` nur setzen wenn Tools vorhanden
- [ ] Response-Parser: JSON → OpenAI `tool_calls`-Format oder Text-Response
- [ ] Streaming-Anpassung: JSON-Response als SSE-Chunks (tool_calls oder Text)

### Phase 3: Testing (1h)
- [ ] Bestehende Tests anpassen (Test 1-4 aus `test_tool_emulation.py`)
- [ ] Neue Tests: Edge Cases (leere Tools, tool_choice=none, mehrere Tool-Calls)
- [ ] Integration mit OpenClaw: academicai-Modell setzen, Tool-Aufrufe testen

### Phase 4: Hardening
- [ ] Fallback wenn JSON-Parsing fehlschlägt → Text-Response
- [ ] Logging: Tool-Call-Erkennung, Parse-Fehler, Fallback-Trigger
- [ ] Metriken: Erfolgsrate der Tool-Emulation

---

## 6. Offene Fragen

1. **Unterstützt AcademicAI `response_format`?** → Muss getestet werden. Wenn nein, fällt Strategie A weg.
2. **Wie verhält sich das Modell bei ambigen Fällen?** → "Wie wird das Wetter?" könnte web_search brauchen oder direkt beantwortet werden. Hängt vom Prompt-Design ab.
3. **Multiple Tool-Calls?** → OpenClaw sendet manchmal Requests die mehrere parallele Tool-Calls erwarten (z.B. `read` + `exec`). JSON-Mode kann das via Array abbilden, aber die Zuverlässigkeit sinkt.
4. **Tool-Call-Ketten?** → Modell ruft Tool A auf → bekommt Ergebnis → ruft Tool B auf. Funktioniert mit allen Strategien, weil OpenClaw die Schleife managed.
5. **Kontakt zum AcademicAI-Team?** → Langfristig wäre nativer `tools`-Support die sauberste Lösung. Lohnt sich die Anfrage?
