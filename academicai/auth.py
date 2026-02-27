"""
AcademicAI Authentifizierung.
Credentials werden aus Umgebungsvariablen gelesen (via .env / python-dotenv).
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_headers() -> dict:
    """
    Gibt Auth-Header für AcademicAI-Requests zurück.
    Liest ACADEMICAI_CLIENT_ID und ACADEMICAI_CLIENT_SECRET aus Umgebungsvariablen.
    """
    client_id = os.environ.get("ACADEMICAI_CLIENT_ID")
    client_secret = os.environ.get("ACADEMICAI_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "ACADEMICAI_CLIENT_ID und ACADEMICAI_CLIENT_SECRET müssen gesetzt sein "
            "(via .env oder Umgebungsvariablen)."
        )

    return {
        "X-Client-ID": client_id,
        "X-Client-Secret": client_secret,
        "Content-Type": "application/json",
    }


def get_base_url() -> str:
    """
    Gibt die AcademicAI Base-URL zurück.
    Muss explizit über ACADEMICAI_BASE_URL gesetzt sein.
    """
    url = os.environ.get("ACADEMICAI_BASE_URL")
    if not url:
        raise ValueError("ACADEMICAI_BASE_URL muss gesetzt sein (via .env oder Umgebungsvariable).")
    return url.rstrip("/")
