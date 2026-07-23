"""Output / analysis language for Grafix prompts."""

from __future__ import annotations

from typing import Any

SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "ru": {"flag": "🇷🇺", "name": "Русский", "native": "Русский"},
    "en": {"flag": "🇬🇧", "name": "English", "native": "English"},
    "es": {"flag": "🇪🇸", "name": "Español", "native": "Español"},
    "pt": {"flag": "🇵🇹", "name": "Português", "native": "Português"},
    "fr": {"flag": "🇫🇷", "name": "Français", "native": "Français"},
    "de": {"flag": "🇩🇪", "name": "Deutsch", "native": "Deutsch"},
    "sr": {"flag": "🇷🇸", "name": "Српски", "native": "Српски"},
    "kk": {"flag": "🇰🇿", "name": "Қазақша", "native": "Қазақша"},
}

DEFAULT_LANGUAGE = "ru"


def normalize_language(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("_", "-")
    if not raw:
        return DEFAULT_LANGUAGE
    # Accept en-US → en, sr-Latn → sr, etc.
    base = raw.split("-", 1)[0]
    aliases = {
        "rus": "ru",
        "russian": "ru",
        "eng": "en",
        "english": "en",
        "spa": "es",
        "esp": "es",
        "spanish": "es",
        "por": "pt",
        "portuguese": "pt",
        "fra": "fr",
        "fre": "fr",
        "french": "fr",
        "ger": "de",
        "deu": "de",
        "german": "de",
        "srb": "sr",
        "serbian": "sr",
        "српски": "sr",
        "kaz": "kk",
        "kazakh": "kk",
        "қазақша": "kk",
        "қазақ": "kk",
    }
    code = aliases.get(base, base)
    return code if code in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def language_meta(code: str | None) -> dict[str, str]:
    lang = normalize_language(code)
    return {"code": lang, **SUPPORTED_LANGUAGES[lang]}


def language_instruction(code: str | None) -> str:
    """Short block injected into extract / QA prompts."""
    meta = language_meta(code)
    lang = meta["code"]
    name = meta["native"]
    if lang == "ru":
        return (
            f"ЯЗЫК ВЫВОДА: {name}. "
            "Имена сущностей оставляй как в тексте (можно канонизировать орфографию). "
            "relation — короткий snake_case предпочтительно на русском (работает_в, руководит, связан_с). "
            "evidence, note и комментарии — на русском."
        )
    if lang == "en":
        return (
            f"OUTPUT LANGUAGE: {name}. "
            "Keep entity names as in the source text (canonicalize spelling if needed). "
            "relation — short English snake_case (works_at, leads, related_to). "
            "evidence, notes and comments — in English."
        )
    if lang == "fr":
        return (
            f"LANGUE DE SORTIE : {name}. "
            "Garde les noms d'entités tels que dans le texte. "
            "relation — snake_case court (travaille_chez, dirige, lié_à), de préférence en français. "
            "evidence, notes et commentaires — en français."
        )
    if lang == "de":
        return (
            f"AUSGABESPRACHE: {name}. "
            "Entitätsnamen wie im Quelltext belassen. "
            "relation — kurzes snake_case (arbeitet_bei, leitet, verbunden_mit), bevorzugt auf Deutsch. "
            "evidence, notes und Kommentare — auf Deutsch."
        )
    if lang == "es":
        return (
            f"IDIOMA DE SALIDA: {name}. "
            "Mantén los nombres de entidades como en el texto. "
            "relation — snake_case corto (trabaja_en, dirige, relacionado_con), preferiblemente en español. "
            "evidence, notas y comentarios — en español."
        )
    if lang == "pt":
        return (
            f"IDIOMA DE SAÍDA: {name}. "
            "Mantém os nomes das entidades como no texto. "
            "relation — snake_case curto (trabalha_em, dirige, relacionado_com), de preferência em português. "
            "evidence, notas e comentários — em português."
        )
    if lang == "kk":
        return (
            f"ШЫҒЫС ТІЛІ: {name}. "
            "Мән атауларын мәтіндегідей қалдыр. "
            "relation — қысқа snake_case (жұмыс_істейді, басқарады, байланысты), мүмкіндігінше қазақша. "
            "evidence, ескертпелер және пікірлер — қазақ тілінде (кирилл)."
        )
    # sr
    return (
        f"ЈЕЗИК ИЗЛАЗА: {name}. "
        "Имена ентитета остави као у тексту. "
        "relation — кратки snake_case (ради_у, руководи, повезан_са), пожељно на српском. "
        "evidence, напомене и коментари — на српском."
    )
