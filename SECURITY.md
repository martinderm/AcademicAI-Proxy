# Security Policy

## Supported scope

This repository contains proxy code only. Secrets and production config are out of scope.

## Reporting a vulnerability

Please report vulnerabilities privately to the maintainer before public disclosure.

Include:

- affected version/commit
- reproduction steps
- expected vs actual behavior
- potential impact

## Secret handling

- Never commit `.env` or real credentials.
- Rotate any exposed secret immediately.
- Keep request dumps disabled in production (`ACADEMICAI_DEBUG_DUMPS=false`).
- If debug dumps are enabled, only redacted payloads are written.
- Proxy startup requires a non-placeholder `ACADEMICAI_PROXY_API_KEY` (minimum 16 chars).
- Do not use shared or guessable Bearer keys across environments.

## Request hardening

- Chat endpoint validates request schema and size limits before backend calls.
- Oversized payloads are rejected with `413`, malformed structures with `422`.
- Rate limiting is enabled by default to reduce abuse risk (`ACADEMICAI_RATE_LIMIT_PER_MINUTE`).
