"""
llm_config.py
=============

Konfiguracja dostawcy i modelu LLM – w jednym miejscu, sterowana plikiem `.env`.

Platforma docelowo gra modelami Meta (Muse Spark przez Meta Model API).
Do czasu otrzymania klucza można jedną zmienną przełączyć się na darmowy Groq:

    LLM_PROVIDER=meta    -> meta/muse-spark-1.3          (klucz: META_API_KEY)   [docelowo]
    LLM_PROVIDER=groq    -> groq/llama-3.3-70b-versatile (klucz: GROQ_API_KEY)   [testy]
    LLM_PROVIDER=custom  -> dowolny model LiteLLM z LLM_MODEL (+ LLM_API_KEY_ENV)

Każdy profil zna: domyślny model, modele zapasowe, nazwę zmiennej z kluczem
oraz to, które parametry dany dostawca/model obsługuje (JSON mode,
reasoning_effort). Dzięki temu nie wysyłamy parametrów, które kończą się
błędem 400, a nazwy modeli są zawsze w poprawnej notacji LiteLLM
("<dostawca>/<model>") – to najczęstsza przyczyna błędów `model_not_found`.

Diagnostyka konfiguracji (bez uruchamiania serwera):
    python -m backend.llm_config            # pokaż aktywną konfigurację
    python -m backend.llm_config --ping     # + jedno mini-wywołanie każdego modelu
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("backend.llm")


class LLMConfigError(RuntimeError):
    """Błąd konfiguracji (zły dostawca, brak klucza, nieistniejący model) – ponawianie nie pomoże."""


# ---------------------------------------------------------------------------
# Profile dostawców
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderProfile:
    """Opis dostawcy: domyślne modele i obsługiwane parametry."""

    name: str
    litellm_prefix: str                 # prefiks w notacji LiteLLM, np. "groq"
    api_key_env: str | None             # zmienna środowiskowa z kluczem
    default_model: str
    default_fallbacks: tuple[str, ...] = ()
    supports_json_mode: bool = True
    #: Domyślny reasoning_effort (None = nie wysyłaj parametru). Modele bez
    #: rozumowania (np. Llama na Groq) odrzucają ten parametr błędem 400.
    default_reasoning_effort: str | None = None
    docs_url: str = ""


PROFILES: dict[str, ProviderProfile] = {
    # Docelowo: modele Meta przez Meta Model API.
    "meta": ProviderProfile(
        name="meta",
        litellm_prefix="meta",
        api_key_env="META_API_KEY",
        default_model="meta/muse-spark-1.3",
        default_fallbacks=(),
        supports_json_mode=True,
        default_reasoning_effort="low",
        docs_url="https://docs.litellm.ai/docs/providers/meta",
    ),
    # Tymczasowo / testy: darmowy Groq. Domyślnie stabilny ("production") model
    # Meta Llama; mniejszy Llama jako zapas (inne limity zapytań).
    "groq": ProviderProfile(
        name="groq",
        litellm_prefix="groq",
        api_key_env="GROQ_API_KEY",
        default_model="groq/llama-3.3-70b-versatile",
        default_fallbacks=("groq/llama-3.1-8b-instant",),
        supports_json_mode=True,
        default_reasoning_effort=None,
        docs_url="https://console.groq.com/docs/models",
    ),
    # Dowolny inny model obsługiwany przez LiteLLM (np. ollama/llama3.1 lokalnie).
    "custom": ProviderProfile(
        name="custom",
        litellm_prefix="",
        api_key_env=None,
        default_model="",
        supports_json_mode=True,
        default_reasoning_effort=None,
    ),
}

#: Prefiksy dostawców LiteLLM, które rozpoznajemy w nazwach modeli.
KNOWN_PREFIXES = {"meta", "groq", "openai", "anthropic", "ollama", "openrouter", "together_ai",
                  "gemini", "mistral", "deepseek", "cerebras", "fireworks_ai", "huggingface"}


# ---------------------------------------------------------------------------
# Ustawienia wynikowe
# ---------------------------------------------------------------------------

def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _bool_env(name: str, default: bool) -> bool:
    value = _env(name).lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


@dataclass
class LLMSettings:
    """Aktywna konfiguracja LLM (wyliczona raz przy starcie z .env)."""

    provider: ProviderProfile
    model: str
    fallbacks: list[str]
    temperature: float
    timeout: float
    max_retries: int                    # ponowienia przy niepoprawnym RUCHU (feedback do modelu)
    transient_retries: int              # ponowienia przy błędach sieci/timeout/5xx
    rate_limit_retries: int             # ponowienia przy 429 – zawsze TEN SAM model
    rate_limit_base_delay: float        # pierwszy odstęp backoffu (s), gdy dostawca nie poda Retry-After
    rate_limit_max_wait: float          # łączny limit czekania na jedno wywołanie (s)
    json_mode: bool
    reasoning_effort: str | None
    api_base: str | None
    api_key_env: str | None
    warnings: list[str] = field(default_factory=list)

    # --- klucz API -------------------------------------------------------------------

    @property
    def api_key(self) -> str | None:
        return _env(self.api_key_env) or None if self.api_key_env else None

    @property
    def api_key_present(self) -> bool:
        return self.api_key_env is None or bool(self.api_key)

    def masked_key(self) -> str:
        if not self.api_key_env:
            return "(nie wymagany)"
        key = self.api_key
        return f"{self.api_key_env}=…{key[-4:]}" if key else f"{self.api_key_env}=BRAK"

    def ensure_ready(self) -> None:
        """Rzuca LLMConfigError z instrukcją, jeśli konfiguracja nie pozwala na wywołanie."""
        if not self.model:
            raise LLMConfigError("Nie ustawiono modelu. Dla LLM_PROVIDER=custom podaj LLM_MODEL w .env.")
        if not self.api_key_present:
            hint = ("Brak klucza Meta? Na czas testów ustaw w .env: LLM_PROVIDER=groq i GROQ_API_KEY "
                    "(darmowy klucz: https://console.groq.com/keys)." if self.provider.name == "meta" else
                    f"Uzupełnij {self.api_key_env} w pliku .env.")
            raise LLMConfigError(f"Brak klucza API: zmienna {self.api_key_env} jest pusta. {hint}")

    # --- modele ------------------------------------------------------------------------

    def resolve_model(self, name: str | None) -> str:
        """Nazwa modelu w notacji LiteLLM (nadpisanie z żądania albo model domyślny)."""
        return normalize_model(name, self.provider) if name else self.model

    def candidate_models(self, override: str | None = None) -> list[str]:
        """Kolejność prób: model główny (lub nadpisany), potem zapasowe – bez duplikatów."""
        out: list[str] = []
        for m in [self.resolve_model(override), *self.fallbacks]:
            if m and m not in out:
                out.append(m)
        return out

    def call_kwargs(self, model: str) -> dict:
        """Parametry przekazywane do litellm.acompletion dla danego modelu."""
        kwargs: dict = {"temperature": self.temperature, "timeout": self.timeout}
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key and provider_of(model) == (self.provider.litellm_prefix or provider_of(model)):
            kwargs["api_key"] = self.api_key  # jawnie – nie polegamy na nazwach zmiennych LiteLLM
        return kwargs

    def describe(self) -> dict:
        """Opis do logów i /health (bez sekretów)."""
        return {
            "provider": self.provider.name,
            "model": self.model,
            "fallbacks": self.fallbacks,
            "api_key": self.masked_key(),
            "ready": self.api_key_present and bool(self.model),
            "json_mode": self.json_mode,
            "reasoning_effort": self.reasoning_effort,
            "api_base": self.api_base,
            "rate_limit": {"retries": self.rate_limit_retries, "max_wait_s": self.rate_limit_max_wait},
        }


def provider_of(model: str) -> str:
    """'groq/llama-3.3-70b-versatile' -> 'groq'; 'gpt-4o' -> ''."""
    head = model.split("/", 1)[0]
    return head if "/" in model and head in KNOWN_PREFIXES else ""


def normalize_model(name: str, profile: ProviderProfile) -> str:
    """
    Doprowadza nazwę do notacji LiteLLM. Brak prefiksu dostawcy to najczęstsza
    przyczyna błędów 'LLM Provider NOT provided' / 'model_not_found':
        'llama-3.3-70b-versatile'      + groq -> 'groq/llama-3.3-70b-versatile'
        'meta-models/muse-spark-open'  + groq -> 'groq/meta-models/muse-spark-open'
        'groq/llama-3.1-8b-instant'           -> bez zmian
    """
    name = name.strip()
    if not name or provider_of(name) or not profile.litellm_prefix:
        return name
    return f"{profile.litellm_prefix}/{name}"


def load_settings() -> LLMSettings:
    """Buduje ustawienia z .env. Nie rzuca wyjątków – problemy trafiają do `warnings`."""
    warnings: list[str] = []
    provider_name = _env("LLM_PROVIDER", "meta").lower()
    if provider_name not in PROFILES:
        warnings.append(f"Nieznany LLM_PROVIDER={provider_name!r} (dostępne: {', '.join(PROFILES)}). "
                        f"Używam 'meta'.")
        provider_name = "meta"
    profile = PROFILES[provider_name]

    model = normalize_model(_env("LLM_MODEL") or profile.default_model, profile)
    if model and profile.litellm_prefix and provider_of(model) not in ("", profile.litellm_prefix):
        warnings.append(f"LLM_MODEL={model} należy do innego dostawcy niż LLM_PROVIDER={provider_name}. "
                        f"Klucz {profile.api_key_env} może nie pasować – rozważ LLM_PROVIDER=custom.")

    raw_fallbacks = _env("LLM_FALLBACK_MODELS")
    if raw_fallbacks.lower() in ("none", "-", "off", "false", "0", "brak"):
        fallbacks: list[str] = []
    elif raw_fallbacks:
        fallbacks = [normalize_model(m, profile) for m in raw_fallbacks.split(",") if m.strip()]
    else:
        fallbacks = list(profile.default_fallbacks)
    fallbacks = [m for m in fallbacks if m != model]

    # reasoning_effort: puste = domyślne profilu, "none" = nie wysyłaj
    effort_env = _env("LLM_REASONING_EFFORT").lower()
    reasoning = None if effort_env == "none" else (effort_env or profile.default_reasoning_effort)

    api_key_env = profile.api_key_env
    if provider_name == "custom":
        api_key_env = _env("LLM_API_KEY_ENV") or None

    return LLMSettings(
        provider=profile,
        model=model,
        fallbacks=fallbacks,
        temperature=float(_env("LLM_TEMPERATURE", "0.3")),
        timeout=float(_env("LLM_TIMEOUT", "60")),
        max_retries=int(_env("LLM_MAX_RETRIES", "2")),
        transient_retries=int(_env("LLM_TRANSIENT_RETRIES", "2")),
        rate_limit_retries=int(_env("LLM_RATE_LIMIT_RETRIES", "5")),
        rate_limit_base_delay=float(_env("LLM_RATE_LIMIT_BASE_DELAY", "2")),
        rate_limit_max_wait=float(_env("LLM_RATE_LIMIT_MAX_WAIT", "90")),
        json_mode=_bool_env("LLM_JSON_MODE", profile.supports_json_mode),
        reasoning_effort=reasoning,
        api_base=_env("LLM_API_BASE") or None,
        api_key_env=api_key_env,
        warnings=warnings,
    )


def log_banner(settings: LLMSettings) -> None:
    """Czytelny baner w konsoli serwera: z jakiego dostawcy i modelu korzystamy."""
    d = settings.describe()
    target = "" if settings.provider.name == "meta" else "   (TRYB TESTOWY – docelowo LLM_PROVIDER=meta)"
    lines = [
        "=" * 72,
        f" LLM provider : {d['provider']}{target}",
        f" Model        : {d['model'] or '(nie ustawiono)'}",
        f" Zapasowe     : {', '.join(d['fallbacks']) or '-'}",
        f" Klucz        : {d['api_key']}",
        f" JSON mode    : {'tak' if d['json_mode'] else 'nie'}   reasoning_effort: {d['reasoning_effort'] or '-'}"
        + (f"   api_base: {d['api_base']}" if d["api_base"] else ""),
        f" Limit 429    : do {settings.rate_limit_retries} ponowień tego samego modelu, "
        f"maks. {settings.rate_limit_max_wait:.0f}s czekania (bez modeli zapasowych)",
        "=" * 72,
    ]
    for line in lines:
        logger.info(line)
    for w in settings.warnings:
        logger.warning("Konfiguracja LLM: %s", w)
    try:
        settings.ensure_ready()
    except LLMConfigError as exc:
        logger.error("LLM NIEGOTOWY: %s", exc)


# ---------------------------------------------------------------------------
# Diagnostyka z linii poleceń
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    import asyncio
    import json

    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Diagnostyka konfiguracji LLM.")
    parser.add_argument("--ping", action="store_true", help="Wyślij mini-zapytanie do każdego modelu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = load_settings()
    log_banner(settings)
    if not args.ping:
        return

    from backend import ai_service

    async def ping() -> None:
        for model in settings.candidate_models():
            messages = [{"role": "system", "content": 'Odpowiedz wyłącznie JSON-em: {"ok": true}'},
                        {"role": "user", "content": "ping"}]
            try:
                result = await ai_service.complete(messages, model=model, use_fallbacks=False)
                print(f"  ✓ {model}: {result.raw[:80]!r}")
            except Exception as exc:  # noqa: BLE001 – diagnostyka: pokaż każdy błąd
                print(f"  ✗ {model}: [{type(exc).__name__}] {exc}")

    print("\nPing modeli:")
    asyncio.run(ping())
    print(json.dumps(settings.describe(), ensure_ascii=False))


if __name__ == "__main__":
    _main()
