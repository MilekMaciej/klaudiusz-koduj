"""
ai_service.py
=============

Warstwa integracji z modelem językowym przez LiteLLM.

Odpowiedzialność modułu:
    1. Zbudowanie promptów:
         - SYSTEM PROMPT  = reguły kontraktu (format JSON) + STRATEGIA UŻYTKOWNIKA,
         - USER PROMPT    = kompletny stan gry (JSON) przygotowany przez silnik.
    2. Wywołanie modelu przez LiteLLM – dostawca i model z `llm_config.py` (.env):
       docelowo Meta (Muse Spark), na czas testów np. darmowy Groq.
    3. Odporność na awarie dostawcy: mapowanie wyjątków LiteLLM, ponowienia
       błędów przejściowych, modele zapasowe, automatyczne wyłączanie
       parametrów, których dany model nie obsługuje.
    4. Wymuszenie i walidacja odpowiedzi w czystym formacie JSON oraz ponowienie
       z informacją zwrotną, gdy model zwróci ruch spoza listy legalnych ruchów.

Moduł NIE zna zasad gry "Tysiąc" – o legalności ruchów decyduje silnik
(`engine/`), który przesyła listę `legal_moves` w stanie gry.

Hierarchia błędów (backend/main.py mapuje je na kody HTTP):
    LLMConfigError       -> 500  zły dostawca/klucz/model; ponawianie nie pomoże
    LLMUnavailableError  -> 503  dostawca chwilowo niedostępny (limit, sieć, awaria)
    AIServiceError       -> 502  model działa, ale nie podał poprawnego ruchu
"""

from __future__ import annotations

import asyncio
import json
import random
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

import litellm
from pydantic import BaseModel, Field, ValidationError

from backend.llm_config import LLMConfigError, LLMSettings, load_settings, log_banner

logger = logging.getLogger("backend.llm")

# ---------------------------------------------------------------------------
# Konfiguracja (wyliczana z .env – patrz llm_config.py)
# ---------------------------------------------------------------------------

SETTINGS: LLMSettings = load_settings()

#: Zgodność wsteczna (nazwy używane wcześniej w innych modułach).
LLM_MODEL: str = SETTINGS.model

# LiteLLM odrzuci parametry, o których WIE, że dostawca ich nie obsługuje.
# Dla przypadków, o których nie wie (konkretny model odrzuca parametr), mamy
# własny mechanizm poniżej (_disabled_params).
litellm.drop_params = True
litellm.suppress_debug_info = True


def reload_settings() -> LLMSettings:
    """Ponowne wczytanie konfiguracji z env (np. w testach lub po zmianie .env)."""
    global SETTINGS, LLM_MODEL
    SETTINGS = load_settings()
    LLM_MODEL = SETTINGS.model
    _disabled_params.clear()
    return SETTINGS


def log_active_config() -> None:
    """Baner w konsoli: aktywny dostawca, model, zapasowe, stan klucza."""
    log_banner(SETTINGS)


# ---------------------------------------------------------------------------
# Kontrakt odpowiedzi modelu
# ---------------------------------------------------------------------------

#: Dozwolone typy akcji. Silnik interpretuje je w zależności od fazy gry.
ActionType = Literal[
    "bid",           # licytacja: podbicie (pole `value`, np. 110)
    "pass",          # licytacja: pas
    "choose_musik",  # wybór musiku 0 lub 1 (pole `value`)
    "discard",       # odłożenie jednej karty po wzięciu musiku (pole `card`)
    "declare",       # deklaracja końcowej wartości kontraktu (pole `value`)
    "play_card",     # rozgrywka: zagranie karty (pole `card`)
    "meld",          # rozgrywka: wyjście K lub Q z pary + meldunek (pole `card`)
]


class LLMMove(BaseModel):
    """Ruch zwrócony przez model – dokładnie to, co parsuje silnik gry."""

    action: ActionType = Field(..., description="Typ akcji.")
    card: str | None = Field(
        default=None,
        description="Karta w formacie <ranga>_<kolor>, np. '10_hearts'. "
                    "Wymagana dla play_card, meld i discard.",
    )
    value: int | None = Field(
        default=None,
        description="Wartość licytacji / deklaracji (bid, declare) lub numer musiku (choose_musik).",
    )
    reasoning: str = Field(
        default="",
        description="Krótkie uzasadnienie ruchu (do logów i analizy strategii).",
    )


class AIServiceError(RuntimeError):
    """Model nie zwrócił poprawnego ruchu mimo ponowień."""


class LLMUnavailableError(RuntimeError):
    """Dostawca LLM chwilowo niedostępny (limit zapytań, timeout, awaria) – po wyczerpaniu prób."""


class ModelOutputError(RuntimeError):
    """Dostawca odrzucił wygenerowaną treść (np. Groq: json_validate_failed). Traktujemy
    to jak błędną odpowiedź modelu: ponowienie z informacją zwrotną."""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


# ---------------------------------------------------------------------------
# Budowanie promptów
# ---------------------------------------------------------------------------

# Stała część promptu systemowego – "kontrakt techniczny". Strategia
# użytkownika jest doklejana PONIŻEJ i może dowolnie kształtować styl gry,
# ale nie może zmienić formatu odpowiedzi (ten fragment jest zawsze pierwszy).
_SYSTEM_CONTRACT = """\
Jesteś autonomicznym graczem w polską grę karcianą "Tysiąc" w wariancie dla 2 osób.
W każdym kroku otrzymasz pełny, jawny stan gry w formacie JSON: swoją rękę,
karty na stole, historię wszystkich jawnych ruchów oraz listę legalnych ruchów
("legal_moves"). Twoim zadaniem jest wybrać JEDEN ruch.

ZASADY ODPOWIEDZI (obowiązkowe, nadrzędne wobec strategii):
1. Odpowiadasz WYŁĄCZNIE jednym obiektem JSON, bez markdownu, bez ``` i bez
   żadnego tekstu przed ani po obiekcie.
2. Schemat:
   {
     "action": "bid" | "pass" | "choose_musik" | "discard" | "declare" | "play_card" | "meld",
     "card": "<ranga>_<kolor>" lub null,
     "value": <liczba całkowita> lub null,
     "reasoning": "<krótkie uzasadnienie, max 2 zdania>"
   }
3. Wybrany ruch MUSI odpowiadać jednemu z elementów "legal_moves".
   Jeśli legalny ruch podaje zakres ("min", "max", "step"), wybierz "value"
   z tego zakresu. Akcja "meld" = zagranie K lub Q z pary i zgłoszenie meldunku.
4. Nie wymyślaj kart, których nie masz na ręce.

Poniżej znajduje się STRATEGIA zdefiniowana przez Twojego właściciela.
Kieruj się nią przy wyborze ruchu spośród legalnych możliwości.
"""


def build_system_prompt(user_strategy: str) -> str:
    """Łączy stały kontrakt techniczny ze strategią użytkownika."""
    strategy = user_strategy.strip() or "Graj rozsądnie i zachowawczo."
    return f"{_SYSTEM_CONTRACT}\n=== STRATEGIA ===\n{strategy}\n=== KONIEC STRATEGII ==="


def build_user_prompt(game_state: dict[str, Any]) -> str:
    """Serializuje stan gry do JSON-a, który trafia do modelu jako wiadomość użytkownika."""
    return (
        "Aktualny stan gry (JSON):\n"
        + json.dumps(game_state, ensure_ascii=False, indent=2)
        + "\n\nZwróć swój ruch jako obiekt JSON zgodny ze schematem."
    )


# ---------------------------------------------------------------------------
# Parsowanie i walidacja odpowiedzi
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw_text: str) -> dict[str, Any]:
    """
    Wyciąga obiekt JSON z odpowiedzi modelu.

    Nawet przy trybie JSON modele czasem opakowują wynik w ```json ... ```
    lub dodają zdanie wstępu – tu robimy "ostatnią linię obrony".
    """
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            raise
        return json.loads(match.group(0))


def _move_matches_legal(move: LLMMove, legal_moves: list[dict[str, Any]]) -> bool:
    """
    Sprawdza, czy ruch modelu odpowiada któremuś z legalnych ruchów.

    Porównujemy tylko pola obecne w opisie legalnego ruchu, np.:
        {"action": "play_card", "card": "10_hearts"}  -> musi zgadzać się action i card
        {"action": "choose_musik", "value": 1}        -> musi zgadzać się action i value
        {"action": "bid", "min": 110, "max": 300}     -> action + value w zakresie
    """
    for legal in legal_moves:
        if legal.get("action") != move.action:
            continue
        if "card" in legal and legal["card"] != move.card:
            continue
        if "value" in legal and legal["value"] != move.value:
            continue
        if "min" in legal or "max" in legal:
            if move.value is None:
                continue
            if move.value < legal.get("min", move.value) or move.value > legal.get("max", move.value):
                continue
            if "step" in legal and (move.value - legal.get("min", 0)) % legal["step"] != 0:
                continue
        return True
    return False


# ---------------------------------------------------------------------------
# Wywołanie modelu z obsługą błędów LiteLLM
# ---------------------------------------------------------------------------

def _exc(*names: str) -> tuple[type[BaseException], ...]:
    """Klasy wyjątków LiteLLM, które istnieją w zainstalowanej wersji biblioteki."""
    found = tuple(getattr(litellm, n) for n in names if isinstance(getattr(litellm, n, None), type))
    return found or (type("_Never", (Exception,), {}),)


_NOT_FOUND = _exc("NotFoundError")
_AUTH = _exc("AuthenticationError", "PermissionDeniedError")
_CONTEXT = _exc("ContextWindowExceededError")
_BAD_REQUEST = _exc("BadRequestError", "UnsupportedParamsError")
#: Limit zapytań (429) – ZAWSZE ponawiamy ten sam model z backoffem, bez modeli zapasowych.
_RATE_LIMIT = _exc("RateLimitError")
#: Inne błędy przejściowe (sieć, timeout, 5xx) – ponowienia, potem ewentualny model zapasowy.
_TRANSIENT = _exc("Timeout", "APIConnectionError", "ServiceUnavailableError",
                  "InternalServerError", "APIError")

#: Parametry wyłączone w trakcie działania dla konkretnych modeli, bo dostawca
#: je odrzucił (np. reasoning_effort dla modelu bez rozumowania).
_disabled_params: dict[str, set[str]] = {}


@dataclass
class CompletionResult:
    """Wynik pojedynczego udanego wywołania modelu."""

    raw: str
    model: str
    provider: str
    usage: dict[str, int]
    fallback_used: bool


def _short(exc: BaseException, limit: int = 300) -> str:
    text = " ".join(str(exc).split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _param_rejected(message: str) -> str | None:
    """Rozpoznaje, który parametr dostawca odrzucił (400), żeby go wyłączyć i ponowić."""
    m = message.lower()
    if "reasoning" in m:
        return "reasoning_effort"
    if "response_format" in m or "json mode" in m or "json_object" in m:
        return "response_format"
    if "temperature" in m:
        return "temperature"
    return None


def _looks_like_unknown_model(message: str) -> bool:
    m = message.lower()
    return any(k in m for k in ("model_not_found", "does not exist", "not found", "decommissioned",
                                "no longer supported", "unknown model", "invalid model"))


_RETRY_IN_RE = re.compile(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?\s*(?:([\d.]+)ms)?",
                          re.IGNORECASE)


def _retry_after_hint(exc: BaseException) -> float | None:
    """
    Ile sekund każe czekać dostawca. Źródła (w tej kolejności):
      1. nagłówek HTTP Retry-After (sekundy),
      2. treść komunikatu, np. Groq: "Please try again in 7.66s" / "in 1m30.5s" / "in 450ms".
    """
    for headers in (getattr(getattr(exc, "response", None), "headers", None),
                    getattr(exc, "litellm_response_headers", None)):
        try:
            value = headers.get("retry-after") if headers else None
            if value is not None:
                return max(0.0, float(value))
        except (TypeError, ValueError, AttributeError):
            pass
    match = _RETRY_IN_RE.search(str(exc))
    if match and any(match.groups()):
        h, m, sec, ms = match.groups()
        return (int(h or 0) * 3600 + int(m or 0) * 60 + float(sec or 0) + float(ms or 0) / 1000)
    return None


def _rate_limit_delay(exc: BaseException, attempt: int, settings: LLMSettings) -> tuple[float, bool]:
    """
    Czas oczekiwania przed kolejną próbą przy 429: podpowiedź dostawcy (+ mały zapas),
    a bez niej backoff wykładniczy z losowym rozrzutem (jitter): base·2^n, max 30s.
    Zwraca (sekundy, czy_z_podpowiedzi_dostawcy).
    """
    hint = _retry_after_hint(exc)
    if hint is not None:
        return hint + random.uniform(0.2, 1.0), True
    backoff = min(settings.rate_limit_base_delay * (2 ** attempt), 30.0)
    return backoff * random.uniform(0.8, 1.2), False


async def complete(messages: list[dict[str, str]], model: str | None = None,
                   use_fallbacks: bool = True) -> CompletionResult:
    """
    Jedno logiczne wywołanie modelu z pełną obsługą błędów:

      - model nie istnieje (model_not_found / 404 / wycofany)  -> następny model zapasowy
      - zły lub brak klucza (401/403)                          -> LLMConfigError (od razu)
      - parametr odrzucony przez model (400)                   -> wyłącz parametr, ponów
      - Groq json_validate_failed                              -> ModelOutputError (feedback)
      - limit zapytań (429 RateLimitError)                     -> czekanie (Retry-After /
                                                                  backoff) i ponowienie TEGO
                                                                  SAMEGO modelu; NIGDY zapasowy
      - timeout / awaria sieci / 5xx                           -> ponowienia z odstępem,
                                                                  potem model zapasowy
    """
    settings = SETTINGS
    settings.ensure_ready()  # brak klucza/modelu -> czytelny LLMConfigError zamiast wyjątku LiteLLM
    candidates = settings.candidate_models(model) if use_fallbacks else [settings.resolve_model(model)]
    problems: list[str] = []
    only_missing_models = True

    for index, model_name in enumerate(candidates):
        transient_attempt = 0
        rate_limit_attempt = 0
        rate_limit_waited = 0.0
        while True:
            kwargs = settings.call_kwargs(model_name)
            for param in _disabled_params.get(model_name, ()):
                kwargs.pop(param, None)
            logger.info("LLM → provider=%s model=%s%s", settings.provider.name, model_name,
                        "  [MODEL ZAPASOWY]" if index else "")
            try:
                response = await litellm.acompletion(model=model_name, messages=messages, **kwargs)

            except _AUTH as exc:
                raise LLMConfigError(
                    f"Dostawca odrzucił klucz API dla {model_name} ({type(exc).__name__}). "
                    f"Sprawdź {settings.api_key_env} w .env. Szczegóły: {_short(exc, 200)}") from exc

            except _NOT_FOUND as exc:
                logger.error("Model %s nie istnieje u dostawcy (%s).", model_name, _short(exc, 160))
                problems.append(f"{model_name}: model nie istnieje")
                break  # -> następny kandydat

            except _CONTEXT as exc:
                logger.error("Stan gry przekracza okno kontekstu modelu %s.", model_name)
                problems.append(f"{model_name}: za długi kontekst")
                only_missing_models = False
                break

            except _BAD_REQUEST as exc:
                message = str(exc)
                if "json_validate_failed" in message:
                    raise ModelOutputError(f"Dostawca odrzucił niepoprawny JSON modelu: {_short(exc, 200)}",
                                           raw=getattr(exc, "body", "") or "") from exc
                param = _param_rejected(message)
                if param and param in kwargs:
                    _disabled_params.setdefault(model_name, set()).add(param)
                    logger.warning("Model %s nie obsługuje parametru '%s' – wyłączam go i ponawiam.",
                                   model_name, param)
                    continue
                if "provider not provided" in message.lower() or "llm provider" in message.lower():
                    raise LLMConfigError(
                        f"LiteLLM nie rozpoznał dostawcy dla '{model_name}'. Podaj model z prefiksem, "
                        f"np. groq/llama-3.3-70b-versatile albo meta/muse-spark-1.3.") from exc
                if _looks_like_unknown_model(message):
                    logger.error("Model %s odrzucony jako nieznany (%s).", model_name, _short(exc, 160))
                    problems.append(f"{model_name}: model nie istnieje")
                    break
                raise LLMConfigError(f"Dostawca odrzucił żądanie (400) dla {model_name}: {_short(exc)}") from exc

            except _RATE_LIMIT as exc:
                # Limit zapytań: czekamy i odpytujemy PONOWNIE TEN SAM model. Przełączenie na
                # model zapasowy nie wchodzi w grę – zmieniałoby model w trakcie partii.
                delay, hinted = _rate_limit_delay(exc, rate_limit_attempt, settings)
                exhausted = rate_limit_attempt >= settings.rate_limit_retries
                too_long = rate_limit_waited + delay > settings.rate_limit_max_wait
                if exhausted or too_long:
                    if exhausted:
                        reason = f"po {rate_limit_attempt} ponowieniach ({rate_limit_waited:.0f}s czekania)"
                    elif hinted and delay > settings.rate_limit_max_wait:
                        reason = (f"dostawca wymaga ok. {delay:.0f}s czekania, więcej niż limit "
                                  f"{settings.rate_limit_max_wait:.0f}s – prawdopodobnie wyczerpany limit "
                                  f"dzienny/godzinowy")
                    else:
                        reason = (f"kolejne czekanie ({delay:.0f}s) przekroczyłoby łączny limit "
                                  f"{settings.rate_limit_max_wait:.0f}s; przeczekano już {rate_limit_waited:.0f}s")
                    logger.error("Limit zapytań dla %s nie ustąpił: %s.", model_name, reason)
                    raise LLMUnavailableError(
                        f"Przekroczony limit zapytań dostawcy '{settings.provider.name}' dla {model_name} "
                        f"({reason}). Odczekaj chwilę i spróbuj ponownie. Szczegóły: {_short(exc, 200)}"
                    ) from exc
                rate_limit_attempt += 1
                rate_limit_waited += delay
                logger.warning("Limit zapytań (429) dla %s – czekam %.1fs (%s) i ponawiam ten sam model "
                               "[%d/%d]", model_name, delay, "wg dostawcy" if hinted else "backoff",
                               rate_limit_attempt, settings.rate_limit_retries)
                await asyncio.sleep(delay)
                continue

            except _TRANSIENT as exc:
                only_missing_models = False
                if transient_attempt < settings.transient_retries:
                    transient_attempt += 1
                    delay = min(2 ** transient_attempt, 20)
                    logger.warning("Błąd przejściowy %s (%s) – ponawiam za %ss [%d/%d]",
                                   type(exc).__name__, model_name, delay, transient_attempt,
                                   settings.transient_retries)
                    await asyncio.sleep(delay)
                    continue
                problems.append(f"{model_name}: {type(exc).__name__}")
                break

            # --- sukces --------------------------------------------------------
            usage = getattr(response, "usage", None)
            return CompletionResult(
                raw=response.choices[0].message.content or "",
                model=model_name,
                provider=settings.provider.name,
                usage={"prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                       "completion_tokens": getattr(usage, "completion_tokens", 0) or 0},
                fallback_used=index > 0,
            )

    summary = "; ".join(problems) or "brak modeli do wywołania"
    if only_missing_models:
        raise LLMConfigError(
            f"Żaden ze skonfigurowanych modeli nie jest dostępny u dostawcy '{settings.provider.name}' "
            f"({summary}). Sprawdź LLM_MODEL / LLM_FALLBACK_MODELS"
            + (f" – aktualna lista: {settings.provider.docs_url}" if settings.provider.docs_url else "") + ".")
    raise LLMUnavailableError(f"Dostawca LLM niedostępny po wszystkich próbach: {summary}")


# ---------------------------------------------------------------------------
# Główna funkcja serwisu
# ---------------------------------------------------------------------------

async def get_move(
    game_state: dict[str, Any],
    user_strategy: str,
    model: str | None = None,
) -> tuple[LLMMove, dict[str, Any]]:
    """
    Pyta model o kolejny ruch.

    Args:
        game_state:    Kompletny stan gry przygotowany przez silnik (bezstanowo).
        user_strategy: Strategia/prompt systemowy zdefiniowany przez użytkownika.
        model:         Opcjonalne nadpisanie modelu (np. do porównań strategii);
                       nazwa bez prefiksu dostaje prefiks aktywnego dostawcy.

    Returns:
        (ruch, metadane) – metadane: provider, model (faktycznie użyty), attempts,
        usage, fallback_used.

    Raises:
        AIServiceError:      model nie podał poprawnego ruchu mimo ponowień,
        LLMConfigError:      błąd konfiguracji (klucz, model, dostawca),
        LLMUnavailableError: dostawca chwilowo niedostępny.
    """
    legal_moves: list[dict[str, Any]] = game_state.get("legal_moves") or []
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(user_strategy)},
        {"role": "user", "content": build_user_prompt(game_state)},
    ]

    last_error = ""
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    result: CompletionResult | None = None

    for attempt in range(1, SETTINGS.max_retries + 2):
        try:
            result = await complete(messages, model=model)
            raw = result.raw
            for k in total_usage:
                total_usage[k] += result.usage.get(k, 0)
        except ModelOutputError as exc:
            raw, last_error = exc.raw or "", f"Odpowiedź nie była poprawnym JSON-em ({exc})."
        else:
            # --- 1) Parsowanie JSON + walidacja schematu --------------------------
            try:
                move = LLMMove.model_validate(_extract_json(raw))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = f"Odpowiedź nie jest poprawnym JSON-em zgodnym ze schematem: {exc}"
            else:
                # --- 2) Walidacja legalności (jeśli silnik podał listę) -----------
                if not legal_moves or _move_matches_legal(move, legal_moves):
                    return move, {
                        "provider": result.provider,
                        "model": result.model,
                        "attempts": attempt,
                        "usage": total_usage,
                        "fallback_used": result.fallback_used,
                    }
                last_error = (f"Ruch {move.model_dump(exclude={'reasoning'})} nie znajduje się "
                              f"na liście legal_moves.")

        # --- 3) Informacja zwrotna dla modelu i kolejna próba -----------------
        logger.warning("Niepoprawny ruch (próba %d): %s", attempt, last_error)
        messages.append({"role": "assistant", "content": raw or "(brak treści)"})
        messages.append({
            "role": "user",
            "content": f"BŁĄD: {last_error}\nPopraw odpowiedź. Zwróć wyłącznie obiekt JSON.",
        })

    raise AIServiceError(f"Model nie zwrócił poprawnego ruchu. Ostatni błąd: {last_error}")
