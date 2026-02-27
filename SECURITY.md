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
