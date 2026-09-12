# OpenCode & OpenChamber Integration Guide

Dieser Leitfaden beschreibt die machine-neutrale Architektur, Einrichtung und Absicherung einer vollständigen lokalen Coding-Agent-Umgebung auf Basis von **OpenCode** (Coding-Harness) und **OpenChamber** (Web- & Mobile-UI), angebunden an den **AcademicAI Proxy** und remote erreichbar über ein privates **Tailscale**-Netzwerk.

---

## 1. Architektur & Komponenten

Das Setup trennt Kernaufgaben sauber in spezialisierte Schichten:

```text
┌─────────────────────────────────────────────────────────────┐
│                    AcademicAI Backend                       │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTPS (REST / Auth Token)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                AcademicAI-Proxy (Lokal)                     │
│  - Host: 127.0.0.1:11435                                    │
│  - OpenAI-kompatible API (/v1/chat/completions, /v1/models) │
│  - Tool-Emulation (TypeScript-Schema & JSON-Repair)         │
│  - KV-Prefix-Cache-Optimierung (Azure OpenAI)               │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON (127.0.0.1)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                OpenCode (Coding-Harness)                    │
│  - Headless Agent-Runtime & CLI                             │
│  - Provider: @ai-sdk/openai-compatible                      │
│  - Multi-Turn Tool-Execution (Bash, Edit, Read, Glob etc.)  │
│  - Lauscht strikt auf Loopback (127.0.0.1)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ IPC / Headless API (127.0.0.1)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             OpenChamber Server (Web / PWA)                  │
│  - Port 3333 (Port 3000 ist fuer MiroFish reserviert)       │
│  - Gesteuerter OpenCode-Lifecycle                           │
│  - Passwortgeschuetzte Web-Oberflaeche                      │
│  - PWA-Support fuer Mobile Browser (iOS Safari / Android)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Tailscale Mesh (E2E verschluesselt)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         Remote Client (z. B. iPhone via Tailscale)          │
│  - Zugriff ueber http://<tailscale-magicdns>:<port>         │
│  - Keine Router-Portfreigaben, keine Cloud-Relays           │
└─────────────────────────────────────────────────────────────┘
```

### Sicherheitsprinzipien
1. **Zero WAN Exposure:** Weder der AcademicAI-Proxy noch OpenCode oder OpenChamber sind ueber oeffentliche IP-Adressen (WAN) aus dem Internet erreichbar.
2. **Loopback-Isolation:** Proxy und OpenCode lauschen strikt auf `127.0.0.1`.
3. **Tailscale-Exklusivitaet:** Eingehende Verbindungen zu OpenChamber (Port 3333) sind per Host-Firewall auf die Tailscale-Schnittstelle beschraenkt.
4. **Secret-Hygiene:** Weder API-Schluessel noch Passwoerter stehen im Klartext in Konfigurationsdateien oder Git-Repositories; sie werden dynamisch ueber Umgebungsvariablen geladen.

---

## 2. Voraussetzungen

- **AcademicAI-Proxy:** Installiert und lokal lauffähig (Standard: Port `11435`).
- **Node.js:** Version 22+ (inklusive `npm` oder `pnpm`).
- **Tailscale:** Installiert und im Ziel-Tailnet eingeloggt.
- **Firewall:** Paketfilter auf Betriebssystemebene (Windows Defender Firewall, `nftables` oder `pf`).

---

## 3. OpenCode installieren & konfigurieren

### Installation
OpenCode wird über den offiziellen Node-Paketmanager installiert:

```bash
# Global via npm
npm install -g opencode-ai@latest

# Oder via pnpm
pnpm add -g opencode-ai@latest
```

Verifikation:
```bash
opencode --version
```

### Provider-Konfiguration (`opencode.json`)
OpenCode sucht seine globale Konfiguration machine-spezifisch unter:
- **Linux/macOS:** `~/.config/opencode/opencode.json`
- **Windows:** `%USERPROFILE%\.config\opencode\opencode.json`

Erstelle oder ergänze die Datei mit dem Provider `academicai`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "academicai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "AcademicAI",
      "options": {
        "baseURL": "http://127.0.0.1:11435/v1",
        "apiKey": "{env:ACADEMICAI_PROXY_API_KEY}"
      },
      "models": {
        "gpt-5-mini": {
          "name": "GPT-5 Mini",
          "tool_call": true
        },
        "gpt-4o": {
          "name": "GPT-4o",
          "tool_call": true
        },
        "gemini-3.5-flash": {
          "name": "Gemini 3.5 Flash",
          "tool_call": true
        },
        "claude-opus-4-6": {
          "name": "Claude Opus 4.6",
          "tool_call": true
        }
      }
    }
  }
}
```

> **Wichtig:** OpenCode unterstützt native Token-Substitution via `{env:VARIABLE}`. Der API-Schlüssel verbleibt in der Umgebung (`ACADEMICAI_PROXY_API_KEY`) und wird nicht in der JSON-Datei gespeichert.

Verifikation der Provider-Erkennung:
```bash
opencode models academicai
```

---

## 4. OpenChamber installieren & konfigurieren

OpenChamber fungiert als Web- und Mobil-Arbeitsplatz für OpenCode. Es verwaltet den OpenCode-Hintergrundprozess automatisch.

### Installation
```bash
npm install -g @openchamber/web@latest
```

Verifikation:
```bash
openchamber --version
```

### Authentifizierung & Start
OpenChamber bietet integrierten Passwortschutz für Browser-Sessions. Das Kennwort kann über die Umgebungsvariable `OPENCHAMBER_UI_PASSWORD` übergeben werden.

```bash
# Server mit Authentifizierung und Bindung an alle lokalen Schnittstellen starten (Port 3333)
openchamber serve --host 0.0.0.0 --port 3333
```

- **OpenCode-Anbindung:** OpenChamber erkennt das installierte `opencode`-Binary automatisch und startet es als Headless-Instanz ausschließlich auf `127.0.0.1`.
- **Health-Check:** `GET http://127.0.0.1:3333/health` gibt den Status von Web-UI und OpenCode-Prozess im JSON-Format aus.

---

## 5. Tailscale & Netzwerkabsicherung

### Firewall-Konfiguration
Um sicherzustellen, dass Port `3333` trotz Bindung an `0.0.0.0` nicht aus dem physischen LAN/WLAN erreichbar ist, wird eine Inbound-Firewall-Regel gesetzt:

- **Regel-Aktion:** Eingehenden TCP-Verkehr auf Port `3333` zulassen.
- **Schnittstellenbeschränkung:** Ausschließlich das virtuelle **Tailscale-Interface** (z. B. `InterfaceAlias: Tailscale` oder Subnetz `100.64.0.0/10`) und Loopback (`127.0.0.1`).
- Alle anderen Adapter (Ethernet, Wi-Fi, Mobilfunk) blockieren.

### Verbindungs-URLs
Sobald der Host im Tailnet aktiv ist, lauten die stabilen Adressen:

- **MagicDNS (bevorzugt):** `http://<hostname>.<tailnet-domain>.ts.net:3333`
- **Tailscale-IPv4:** `http://100.x.y.z:3333`

---

## 6. Mobiler Zugriff (iOS / Safari / PWA)

OpenChamber liefert ein vollständiges Web App Manifest (`/site.webmanifest`) und Touch-Icons aus.

### Einrichtung auf dem iPhone:
1. Sicherstellen, dass die Tailscale-App auf dem iPhone aktiv und mit demselben Tailnet verbunden ist.
2. Im mobilen Safari die MagicDNS- oder Tailscale-IP-Adresse aufrufen (`http://<hostname>.<tailnet>.ts.net:3333`).
3. Das hinterlegte UI-Passwort eingeben.
4. Auf das **Teilen-Symbol** (Viereck mit Pfeil nach oben) tippen.
5. **„Zum Home-Bildschirm“** auswählen.
6. OpenChamber startet nun als eigenständige Progressive Web App (PWA) im Vollbildmodus ohne Browserleisten.

---

## 7. Prozess-Lifecycle & Autostart

Für einen unterbrechungsfreien Betrieb sollten der AcademicAI-Proxy und OpenChamber nach einem Systemneustart automatisch gestartet werden.

### Startreihenfolge
1. **Schritt 1:** `AcademicAI-Proxy` starten (Port `11435` bereitstellen).
2. **Schritt 2:** `OpenChamber` starten (mit 10–15 Sekunden Verzögerung; OpenChamber initialisiert OpenCode bei Bedarf selbst).

### Betriebssystem-Integration:
- **Windows:** Windows Aufgabenplanung (Task Scheduler) mit zwei zeitversetzten Trigger-Aktionen beim Benutzer-Login.
- **Linux:** `systemd --user` Services mit `Wants=tailscaled.service` und aktivierter Linger-Funktion (`loginctl enable-linger $USER`).
- **macOS:** `launchd` User LaunchAgents unter `~/Library/LaunchAgents/`.

---

## 8. Speicherorte & Runtime-Datenverlagerung

OpenCode und OpenChamber speichern umfangreiche dynamische Laufzeitdaten (SQLite-Datenbank aller Sessions, Snapshots, heruntergeladene lokale Offline-Sprachmodelle für STT/TTS von bis zu 1 GB, Chat-Exporte, Modell-Caches und Logs).

Standardmäßig liegen diese im Benutzerverzeichnis (`~/.config`, `~/.local/share`, `~/.cache` bzw. `%USERPROFILE%`). Um System-Partitionen (z. B. `C:\`) zu entlasten und alle Daten auf einer dedizierten Datenpartition (z. B. `D:\`) zu bündeln, werden standardisierte Umgebungsvariablen verwendet:

### Konfigurations-Variablen

| Komponente | Variable | Standard | Beispiel für Datenpartition (`D:\`) |
| :--- | :--- | :--- | :--- |
| **OpenChamber** | `OPENCHAMBER_DATA_DIR` | `~/.config/openchamber` | `D:\users\dagobert\.openchamber` |
| **OpenChamber** | `OPENCHAMBER_MANAGED_PROCESS_REGISTRY` | `~/.config/openchamber/managed-opencode` | `D:\users\dagobert\.openchamber\managed-opencode` |
| **OpenCode** | `XDG_CONFIG_HOME` | `~/.config` | `D:\users\dagobert\.opencode\config` |
| **OpenCode** | `XDG_DATA_HOME` | `~/.local/share` | `D:\users\dagobert\.opencode\data` |
| **OpenCode** | `XDG_CACHE_HOME` | `~/.cache` | `D:\users\dagobert\.opencode\cache` |

OpenChamber vererbt diese Umgebungsvariablen automatisch an die verwaltete OpenCode-Kindinstanz. Dadurch werden Session-Daten (`opencode.db`), Logs, Snapshots und Modelle direkt auf dem Zielvolume gespeichert.

