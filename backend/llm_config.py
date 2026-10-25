"""
llm_config.py
=============

Konfiguracja dostawców i modeli LLM – w jednym miejscu, sterowana plikiem `.env`.

Platforma docelowo gra modelami Meta (Muse Spark przez Meta Model API).
Do czasu otrzymania klucza można jedną zmienną przełączyć się na darmowego dostawcę:

    LLM_PROVIDER=meta    -> meta/muse-spark-1.3            (META_API_KEY)    [docelowo]
    LLM_PROVIDER=groq    -> groq/llama-3.3-70b-versatile   (GROQ_API_KEY)    [testy, darmowy]
    LLM_PROVIDER=gemini  -> gemini/gemini-flash-latest     (GEMINI_API_KEY)  [testy, darmowy]
    LLM_PROVIDER=custom  -> dowolny model LiteLLM z LLM_MODEL (+ LLM_API_KEY_ENV)

`LLM_PROVIDER` wybiera model DOMYŚLNY. Parametry (klucz, temperatura, JSON mode,
reasoning_effort) są dobierane OSOBNO DLA KAŻDEGO MODELU na podstawie jego prefiksu.
Dzięki temu w jednej partii Bot A może grać np. na Groq, a Bot B na Gemini –
o ile w .env są oba klucze (lista dostępnych modeli: `available_models()`).

Nazwy modeli są zawsze w notacji LiteLLM ("<dostawca>/<model>"). Brak prefiksu
to najczęstsza przyczyna błędów `model_not_found`; dla Gemini dodatkowo model
bez prefiksu `gemini/` trafiłby do Vertex AI (wymaga kont GCP).

Diagnostyka konfiguracji (bez uruchamiania serwera):
    python -m backend.llm_config            # pokaż aktywną konfigurację i dostępne modele
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
    """Opis dostawcy: domyślne modele i parametry, które jego modele obsługują."""

    name: str
    litellm_prefix: str                       # prefiks w notacji LiteLLM, np. "groq"
    api_key_env: str | None                   # zmienna środowiskowa z kluczem
    default_model: str
    default_fallbacks: tuple[str, ...] = ()
    #: Inne nazwy zmiennej z kluczem akceptowane jako zapas (np. GOOGLE_API_KEY).
    api_key_aliases: tuple[str, ...] = ()
    supports_json_mode: bool = True
    #: Domyślny reasoning_effort (None = nie wysyłaj parametru). Modele bez
    #: rozumowania (np. Llama na Groq) odrzucają ten parametr błędem 400.
    default_reasoning_effort: str | None = None
    #: Domyślna temperatura (None = nie wysyłaj – zostaje domyślna dostawcy).
    default_temperature: float | None = 0.3
    label: str = ""
    docs_url: str = ""
    free_tier: bool = False


PROFILES: dict[str, ProviderProfile] = {
    # Docelowo: modele Meta przez Meta Model API.
    "meta": ProviderProfile(
        name="meta", litellm_prefix="meta", api_key_env="META_API_KEY",
        default_model="meta/muse-spark-1.3",
        default_reasoning_effort="low",
        label="Meta Muse Spark (docelowy)",
        docs_url="https://docs.litellm.ai/docs/providers/meta",
    ),
    # Testy: darmowy Groq – stabilny ("production") model Meta Llama + mniejszy jako zapas.
    "groq": ProviderProfile(
        name="groq", litellm_prefix="groq", api_key_env="GROQ_API_KEY",
        default_model="groq/openai/gpt-oss-120b",
        default_fallbacks=("groq/llama-3.1-8b-instant",),
        label="Groq · Llama (darmowy)",
        docs_url="https://console.groq.com/docs/models",
        free_tier=True,
    ),
    # Testy: darmowy Gemini Flash (Google AI Studio). Alias "-latest" wskazuje
    # zawsze bieżący model Flash – nie kończy się model_not_found po wycofaniu wersji.
    # Gemini 3 nie lubi temperatur < 1.0 (pętle, słabsze rozumowanie) – nie wysyłamy jej.
    # reasoning_effort="low" ogranicza "myślenie" (domyślne "high" jest wolne i drogie w tokenach).
    "gemini": ProviderProfile(
        name="gemini", litellm_prefix="gemini", api_key_env="GEMINI_API_KEY",
        api_key_aliases=("GOOGLE_API_KEY",),
        default_model="gemini/gemini-flash-latest",
        default_fallbacks=("gemini/gemini-flash-lite-latest",),
        default_reasoning_effort="low",
        default_temperature=None,
        label="Google Gemini Flash (darmowy)",
        docs_url="https://ai.google.dev/gemini-api/docs/models",
        free_tier=True,
    ),
    # Dowolny inny model obsługiwany przez LiteLLM (np. ollama/llama3.1 lokalnie).
    "custom": ProviderProfile(
        name="custom", litellm_prefix="", api_key_env=None, default_model="",
        label="Własny model LiteLLM",
    ),
}

#: Prefiksy dostawców LiteLLM, które rozpoznajemy w nazwach modeli.
KNOWN_PREFIXES = {"meta", "groq", "openai", "anthropic", "ollama", "openrouter", "together_ai",
                  "gemini", "vertex_ai", "mistral", "deepseek", "cerebras", "fireworks_ai", "huggingface"}

#: Prefiks modelu -> profil (dla modeli spoza aktywnego dostawcy).
_PROFILE_BY_PREFIX = {p.litellm_prefix: p for p in PROFILES.values() if p.litellm_prefix}

_NONE_VALUES = ("none", "-", "off", "false", "0", "brak")


# ---------------------------------------------------------------------------
# Funkcje pomocnicze
# ---------------------------------------------------------------------------

def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _bool_env(name: str) -> bool | None:
    value = _env(name).lower()
    if value in ("1", "true", "yes", "on", "tak"):
        return True
    if value in ("0", "false", "no", "off", "nie"):
        return False
    return None


def provider_of(model: str) -> str:
    """'groq/llama-3.3-70b-versatile' -> 'groq'; 'gpt-4o' -> ''."""
    head = model.split("/", 1)[0]
    return head if "/" in model and head in KNOWN_PREFIXES else ""


def normalize_model(name: str, profile: ProviderProfile) -> str:
    """
    Doprowadza nazwę do notacji LiteLLM:
        'llama-3.3-70b-versatile'      + groq   -> 'groq/llama-3.3-70b-versatile'
        'gemini-flash-latest'          + gemini -> 'gemini/gemini-flash-latest'
        'meta-models/muse-spark-open'  + groq   -> 'groq/meta-models/muse-spark-open'
        'gemini/gemini-2.5-flash'               -> bez zmian (ma już prefiks)
    """
    name = name.strip()
    if not name or provider_of(name) or not profile.litellm_prefix:
        return name
    return f"{profile.litellm_prefix}/{name}"


# ---------------------------------------------------------------------------
# Ustawienia wynikowe
# ---------------------------------------------------------------------------

@dataclass
class LLMSettings:
    """Aktywna konfiguracja LLM (wyliczona przy starcie z .env)."""

    provider: ProviderProfile
    model: str
    fallbacks: list[str]
    temperature_env: float | None        # jawne LLM_TEMPERATURE (None = domyślne dostawcy modelu)
    timeout: float
    max_retries: int                     # ponowienia przy niepoprawnym RUCHU (feedback do modelu)
    transient_retries: int               # ponowienia przy błędach sieci/timeout/5xx
    rate_limit_retries: int              # ponowienia przy 429 – zawsze TEN SAM model
    rate_limit_base_delay: float         # pierwszy odstęp backoffu (s), gdy dostawca nie poda czasu
    rate_limit_max_wait: float           # łączny limit czekania na jedno wywołanie (s)
    json_mode_env: bool | None           # jawne LLM_JSON_MODE (None = według dostawcy)
    reasoning_env: str                   # LLM_REASONING_EFFORT: "" = według dostawcy, "none" = nigdy
    api_base: str | None
    custom_key_env: str | None           # LLM_API_KEY_ENV (tylko provider=custom)
    warnings: list[str] = field(default_factory=list)

    # --- profil i klucz dla konkretnego modelu ------------------------------------------

    def profile_for(self, model: str) -> ProviderProfile:
        """Profil dostawcy modelu (po prefiksie); nieznany prefiks -> profil 'custom'."""
        return _PROFILE_BY_PREFIX.get(provider_of(model), PROFILES["custom"])

    def key_envs_for(self, model: str) -> tuple[str, ...]:
        """Zmienne, z których czytamy klucz dla modelu (pierwsza niepusta wygrywa)."""
        profile = self.profile_for(model)
        if profile.name == "custom":
            return (self.custom_key_env,) if self.custom_key_env else ()
        return tuple(e for e in (profile.api_key_env, *profile.api_key_aliases) if e)

    def api_key_for(self, model: str) -> str | None:
        for env_name in self.key_envs_for(model):
            if _env(env_name):
                return _env(env_name)
        return None

    def key_ready(self, model: str) -> bool:
        return not self.key_envs_for(model) or self.api_key_for(model) is not None

    def masked_key(self, model: str | None = None) -> str:
        envs = self.key_envs_for(model or self.model)
        if not envs:
            return "(nie wymagany)"
        key = self.api_key_for(model or self.model)
        return f"{envs[0]}=…{key[-4:]}" if key else f"{envs[0]}=BRAK"

    def ensure_ready(self, model: str | None = None) -> None:
        """Rzuca LLMConfigError z instrukcją, jeśli nie da się wywołać modelu."""
        model = model or self.model
        if not model:
            raise LLMConfigError("Nie ustawiono modelu. Dla LLM_PROVIDER=custom podaj LLM_MODEL w .env.")
        if self.key_ready(model):
            return
        env_name = self.key_envs_for(model)[0]
        hint = {
            "meta": "Brak klucza Meta? Na czas testów ustaw w .env LLM_PROVIDER=groq (GROQ_API_KEY, "
                    "https://console.groq.com/keys) albo LLM_PROVIDER=gemini (GEMINI_API_KEY, "
                    "https://aistudio.google.com/apikey).",
            "gemini": "Darmowy klucz: https://aistudio.google.com/apikey",
            "groq": "Darmowy klucz: https://console.groq.com/keys",
        }.get(self.profile_for(model).name, f"Uzupełnij {env_name} w pliku .env.")
        raise LLMConfigError(f"Brak klucza API dla {model}: zmienna {env_name} jest pusta. {hint}")

    # --- zgodność wsteczna (pola używane wcześniej w innych modułach) --------------------

    @property
    def api_key_env(self) -> str | None:
        envs = self.key_envs_for(self.model)
        return envs[0] if envs else None

    @property
    def api_key_present(self) -> bool:
        return self.key_ready(self.model)

    @property
    def temperature(self) -> float | None:
        return self.temperature_for(self.model)

    @temperature.setter
    def temperature(self, value: float | None) -> None:  # np. debug_sandbox --temperature
        self.temperature_env = value

    @property
    def json_mode(self) -> bool:
        return self.json_mode_for(self.model)

    @property
    def reasoning_effort(self) -> str | None:
        return self.reasoning_for(self.model)

    # --- parametry dla konkretnego modelu --------------------------------------------------

    def temperature_for(self, model: str) -> float | None:
        if self.temperature_env is not None:
            return self.temperature_env
        return self.profile_for(model).default_temperature

    def json_mode_for(self, model: str) -> bool:
        if self.json_mode_env is not None:
            return self.json_mode_env
        return self.profile_for(model).supports_json_mode

    def reasoning_for(self, model: str) -> str | None:
        if self.reasoning_env in _NONE_VALUES:
            return None
        return self.reasoning_env or self.profile_for(model).default_reasoning_effort

    # --- modele ------------------------------------------------------------------------

    def resolve_model(self, name: str | None) -> str:
        """Nazwa modelu w notacji LiteLLM (nadpisanie z żądania albo model domyślny)."""
        return normalize_model(name, self.provider) if name else self.model

    def candidate_models(self, override: str | None = None) -> list[str]:
        """
        Kolejność prób: model główny (lub nadpisany), potem zapasowe TEGO SAMEGO dostawcy.
        Model innego dostawcy niż domyślny dostaje zapasowe swojego profilu (o ile ma klucz).
        """
        primary = self.resolve_model(override)
        if provider_of(primary) == provider_of(self.model):
            extra = self.fallbacks
        else:
            profile = self.profile_for(primary)
            extra = [m for m in profile.default_fallbacks if self.key_ready(m)]
        out: list[str] = []
        for m in [primary, *extra]:
            if m and m not in out:
                out.append(m)
        return out

    def call_kwargs(self, model: str) -> dict:
        """Parametry przekazywane do litellm.acompletion dla danego modelu."""
        kwargs: dict = {"timeout": self.timeout}
        temperature = self.temperature_for(model)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.json_mode_for(model):
            kwargs["response_format"] = {"type": "json_object"}
        reasoning = self.reasoning_for(model)
        if reasoning:
            kwargs["reasoning_effort"] = reasoning
        if self.api_base and self.profile_for(model).name == "custom":
            kwargs["api_base"] = self.api_base
        key = self.api_key_for(model)
        if key:
            kwargs["api_key"] = key  # jawnie – nie polegamy na nazwach zmiennych w LiteLLM
        return kwargs

    def available_models(self) -> list[dict]:
        """
        Modele do wyboru w interfejsie: model domyślny, jego zapasowe oraz domyślne
        modele innych dostawców – każdy z informacją, czy w .env jest do niego klucz.
        """
        seen: set[str] = set()
        out: list[dict] = []

        def add(model: str, note: str) -> None:
            if not model or model in seen:
                return
            seen.add(model)
            profile = self.profile_for(model)
            out.append({"model": model, "provider": profile.name, "label": profile.label or model,
                        "note": note, "ready": self.key_ready(model), "free_tier": profile.free_tier,
                        "default": model == self.model})

        add(self.model, "domyślny (.env)")
        for m in self.fallbacks:
            add(m, "zapasowy")
        for profile in PROFILES.values():
            add(profile.default_model, "")
            for m in profile.default_fallbacks:
                add(m, "lżejszy")
        return out

    def describe(self) -> dict:
        """Opis do logów i /health (bez sekretów)."""
        return {
            "provider": self.provider.name,
            "model": self.model,
            "fallbacks": self.fallbacks,
            "api_key": self.masked_key(),
            "ready": self.key_ready(self.model) and bool(self.model),
            "json_mode": self.json_mode,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "api_base": self.api_base,
            "rate_limit": {"retries": self.rate_limit_retries, "max_wait_s": self.rate_limit_max_wait},
        }


def load_settings() -> LLMSettings:
    """Buduje ustawienia z .env. Nie rzuca wyjątków – problemy trafiają do `warnings`."""
    warnings: list[str] = []
    provider_name = _env("LLM_PROVIDER", "meta").lower()
    if provider_name == "google":
        provider_name = "gemini"
    if provider_name not in PROFILES:
        warnings.append(f"Nieznany LLM_PROVIDER={provider_name!r} (dostępne: {', '.join(PROFILES)}). "
                        f"Używam 'meta'.")
        provider_name = "meta"
    profile = PROFILES[provider_name]

    model = normalize_model(_env("LLM_MODEL") or profile.default_model, profile)
    if model and profile.litellm_prefix and provider_of(model) not in ("", profile.litellm_prefix):
        warnings.append(f"LLM_MODEL={model} należy do innego dostawcy niż LLM_PROVIDER={provider_name} – "
                        f"użyję klucza właściwego dla {provider_of(model)}.")

    raw_fallbacks = _env("LLM_FALLBACK_MODELS")
    if raw_fallbacks.lower() in _NONE_VALUES:
        fallbacks: list[str] = []
    elif raw_fallbacks:
        fallbacks = [normalize_model(m, profile) for m in raw_fallbacks.split(",") if m.strip()]
    else:
        fallbacks = list(profile.default_fallbacks)
    fallbacks = [m for m in fallbacks if m != model]

    temperature_raw = _env("LLM_TEMPERATURE")
    temperature_env = None if temperature_raw.lower() in ("", "default") else float(temperature_raw)
    if temperature_env is not None and temperature_env < 1.0 and provider_of(model) == "gemini":
        warnings.append(f"LLM_TEMPERATURE={temperature_env} dla Gemini – Google zaleca dla Gemini 3 "
                        f"temperaturę 1.0 (niższa grozi pętlami). Zostaw LLM_TEMPERATURE puste.")

    return LLMSettings(
        provider=profile,
        model=model,
        fallbacks=fallbacks,
        temperature_env=temperature_env,
        timeout=float(_env("LLM_TIMEOUT", "60")),
        max_retries=int(_env("LLM_MAX_RETRIES", "2")),
        transient_retries=int(_env("LLM_TRANSIENT_RETRIES", "2")),
        rate_limit_retries=int(_env("LLM_RATE_LIMIT_RETRIES", "5")),
        rate_limit_base_delay=float(_env("LLM_RATE_LIMIT_BASE_DELAY", "2")),
        rate_limit_max_wait=float(_env("LLM_RATE_LIMIT_MAX_WAIT", "90")),
        json_mode_env=_bool_env("LLM_JSON_MODE"),
        reasoning_env=_env("LLM_REASONING_EFFORT").lower(),
        api_base=_env("LLM_API_BASE") or None,
        custom_key_env=_env("LLM_API_KEY_ENV") or None,
        warnings=warnings,
    )


def log_banner(settings: LLMSettings) -> None:
    """Czytelny baner w konsoli serwera: z jakiego dostawcy i modelu korzystamy."""
    d = settings.describe()
    target = "" if settings.provider.name == "meta" else "   (TRYB TESTOWY – docelowo LLM_PROVIDER=meta)"
    other = [m["model"] for m in settings.available_models()
             if m["ready"] and m["model"] != settings.model and m["model"] not in settings.fallbacks]
    lines = [
        "=" * 72,
        f" LLM provider : {d['provider']}{target}",
        f" Model        : {d['model'] or '(nie ustawiono)'}",
        f" Zapasowe     : {', '.join(d['fallbacks']) or '-'}",
        f" Klucz        : {d['api_key']}",
        f" Parametry    : JSON mode {'tak' if d['json_mode'] else 'nie'} | reasoning_effort "
        f"{d['reasoning_effort'] or '-'} | temperatura {d['temperature'] if d['temperature'] is not None else 'domyślna'}"
        + (f" | api_base {d['api_base']}" if d["api_base"] else ""),
        f" Limit 429    : do {settings.rate_limit_retries} ponowień tego samego modelu, "
        f"maks. {settings.rate_limit_max_wait:.0f}s czekania (bez modeli zapasowych)",
        f" Inne modele  : {', '.join(other) or '-'}  (dostępne do wyboru per bot – są klucze)",
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
    parser.add_argument("--ping", action="store_true", help="Wyślij mini-zapytanie do każdego modelu z kluczem")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = load_settings()
    log_banner(settings)
    print("\nModele:")
    for m in settings.available_models():
        print(f"  {'✓' if m['ready'] else '·'} {m['model']:<40} {m['label']}"
              f"{' – ' + m['note'] if m['note'] else ''}{'' if m['ready'] else '  (brak klucza)'}")
    if not args.ping:
        return

    from backend import ai_service

    async def ping() -> None:
        for model in [m["model"] for m in settings.available_models() if m["ready"]]:
            messages = [{"role": "system", "content": 'Odpowiedz wyłącznie JSON-em: {"ok": true}'},
                        {"role": "user", "content": "ping"}]
            try:
                result = await ai_service.complete(messages, model=model, use_fallbacks=False)
                print(f"  ✓ {model}: {result.raw[:80]!r}")
            except Exception as exc:  # noqa: BLE001 – diagnostyka: pokaż każdy błąd
                print(f"  ✗ {model}: [{type(exc).__name__}] {exc}")

    print("\nPing modeli z kluczem:")
    asyncio.run(ping())
    print(json.dumps(settings.describe(), ensure_ascii=False))


if __name__ == "__main__":
    _main()
