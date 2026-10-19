"""
ai_service.py
=============

Warstwa integracji z modelem językowym przez LiteLLM.

Odpowiedzialność modułu:
    1. Zbudowanie promptów:
         - SYSTEM PROMPT  = reguły kontraktu (format JSON) + STRATEGIA UŻYTKOWNIKA,
         - USER PROMPT    = kompletny stan gry (JSON) przygotowany przez silnik.
    2. Wywołanie modelu Meta Muse Spark przez LiteLLM (asynchronicznie).
    3. Wymuszenie i walidacja odpowiedzi w czystym formacie JSON.
    4. Ponowienie zapytania, gdy model zwróci niepoprawny JSON lub ruch
       spoza listy legalnych ruchów (model dostaje informację o błędzie).

Moduł NIE zna zasad gry "Tysiąc" – o legalności ruchów decyduje silnik
(`engine/`), który przesyła listę `legal_moves` w stanie gry. Dzięki temu
backend pozostaje cienką, bezstanową warstwą pośrednią.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal

import litellm
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguracja (z pliku .env / zmiennych środowiskowych)
# ---------------------------------------------------------------------------

#: Identyfikator modelu w notacji LiteLLM: "<provider>/<model>".
#: Klucz API odczytywany jest przez LiteLLM ze zmiennej META_API_KEY.
LLM_MODEL: str = os.getenv("LLM_MODEL", "meta/muse-spark-1.3")

#: Temperatura – niska wartość = bardziej przewidywalne, "trzymające się
#: reguł" odpowiedzi. Strategię i tak definiuje prompt użytkownika.
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

#: Wysiłek rozumowania (Muse Spark to model "reasoning"):
#: minimal | low | medium | high | xhigh. Pusty string = domyślny modelu.
LLM_REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "low")

#: Ile razy maksymalnie ponawiamy zapytanie przy błędnej odpowiedzi.
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

#: Limit czasu pojedynczego wywołania (sekundy).
LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "60"))

# LiteLLM po cichu odrzuci parametry nieobsługiwane przez danego dostawcę,
# zamiast rzucać wyjątek (np. gdy podmienimy model na inny w .env).
litellm.drop_params = True


# ---------------------------------------------------------------------------
# Kontrakt odpowiedzi modelu
# ---------------------------------------------------------------------------

#: Dozwolone typy akcji. Silnik interpretuje je w zależności od fazy gry.
ActionType = Literal[
    "bid",          # licytacja: podbicie (pole `value`, np. 110)
    "pass",         # licytacja: pas
    "give_card",    # po wzięciu musiku: oddanie karty przeciwnikowi
    "declare",      # deklaracja końcowej wartości kontraktu (pole `value`)
    "play_card",    # rozgrywka: zagranie karty (pole `card`)
]


class LLMMove(BaseModel):
    """Ruch zwrócony przez model – dokładnie to, co parsuje silnik gry."""

    action: ActionType = Field(..., description="Typ akcji.")
    card: str | None = Field(
        default=None,
        description="Karta w formacie <ranga>_<kolor>, np. '10_hearts'. "
                    "Wymagana dla play_card i give_card.",
    )
    target_player: str | None = Field(
        default=None,
        description="Gracz, któremu oddajemy kartę (tylko give_card).",
    )
    value: int | None = Field(
        default=None,
        description="Wartość licytacji / deklaracji (bid, declare).",
    )
    reasoning: str = Field(
        default="",
        description="Krótkie uzasadnienie ruchu (do logów i analizy strategii).",
    )


class AIServiceError(RuntimeError):
    """Model nie zwrócił poprawnego ruchu mimo ponowień."""


# ---------------------------------------------------------------------------
# Budowanie promptów
# ---------------------------------------------------------------------------

# Stała część promptu systemowego – "kontrakt techniczny". Strategia
# użytkownika jest doklejana PONIŻEJ i może dowolnie kształtować styl gry,
# ale nie może zmienić formatu odpowiedzi (ten fragment jest zawsze pierwszy).
_SYSTEM_CONTRACT = """\
Jesteś autonomicznym graczem w polską grę karcianą "Tysiąc".
W każdym kroku otrzymasz pełny, jawny stan gry w formacie JSON: swoją rękę,
karty na stole, historię wszystkich jawnych ruchów oraz listę legalnych ruchów
("legal_moves"). Twoim zadaniem jest wybrać JEDEN ruch.

ZASADY ODPOWIEDZI (obowiązkowe, nadrzędne wobec strategii):
1. Odpowiadasz WYŁĄCZNIE jednym obiektem JSON, bez markdownu, bez ``` i bez
   żadnego tekstu przed ani po obiekcie.
2. Schemat:
   {
     "action": "bid" | "pass" | "give_card" | "declare" | "play_card",
     "card": "<ranga>_<kolor>" lub null,
     "target_player": "<id gracza>" lub null,
     "value": <liczba całkowita> lub null,
     "reasoning": "<krótkie uzasadnienie, max 2 zdania>"
   }
3. Wybrany ruch MUSI odpowiadać jednemu z elementów "legal_moves".
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
        {"action": "bid", "min": 110, "max": 300}     -> action + value w zakresie
    """
    for legal in legal_moves:
        if legal.get("action") != move.action:
            continue
        if "card" in legal and legal["card"] != move.card:
            continue
        if "target_player" in legal and legal["target_player"] != move.target_player:
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
        model:         Opcjonalne nadpisanie modelu (np. do porównań strategii).

    Returns:
        (ruch, metadane) – metadane zawierają m.in. liczbę prób i zużycie tokenów.

    Raises:
        AIServiceError: gdy po wszystkich próbach nie uzyskano poprawnego ruchu.
    """
    model_name = model or LLM_MODEL
    legal_moves: list[dict[str, Any]] = game_state.get("legal_moves") or []

    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(user_strategy)},
        {"role": "user", "content": build_user_prompt(game_state)},
    ]

    extra_kwargs: dict[str, Any] = {}
    if LLM_REASONING_EFFORT:
        extra_kwargs["reasoning_effort"] = LLM_REASONING_EFFORT

    last_error: str = ""
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    for attempt in range(1, LLM_MAX_RETRIES + 2):
        logger.info("LLM call (model=%s, attempt=%d)", model_name, attempt)

        response = await litellm.acompletion(
            model=model_name,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            timeout=LLM_TIMEOUT,
            # Tryb JSON – wymusza na modelu zwrócenie poprawnego obiektu JSON.
            # (Gdy dostawca go nie wspiera, drop_params=True go pominie,
            #  a format i tak egzekwuje prompt + parser poniżej.)
            response_format={"type": "json_object"},
            **extra_kwargs,
        )

        usage = getattr(response, "usage", None)
        if usage:
            total_usage["prompt_tokens"] += usage.prompt_tokens or 0
            total_usage["completion_tokens"] += usage.completion_tokens or 0

        raw = response.choices[0].message.content or ""

        # --- 1) Parsowanie JSON + walidacja schematu --------------------------
        try:
            move = LLMMove.model_validate(_extract_json(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = f"Odpowiedź nie jest poprawnym JSON-em zgodnym ze schematem: {exc}"
        else:
            # --- 2) Walidacja legalności (jeśli silnik podał listę) -----------
            if not legal_moves or _move_matches_legal(move, legal_moves):
                return move, {"model": model_name, "attempts": attempt, "usage": total_usage}
            last_error = (
                f"Ruch {move.model_dump(exclude={'reasoning'})} nie znajduje się "
                f"na liście legal_moves."
            )

        # --- 3) Informacja zwrotna dla modelu i kolejna próba -----------------
        logger.warning("Niepoprawny ruch (próba %d): %s", attempt, last_error)
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": f"BŁĄD: {last_error}\nPopraw odpowiedź. Zwróć wyłącznie obiekt JSON.",
        })

    raise AIServiceError(f"Model nie zwrócił poprawnego ruchu. Ostatni błąd: {last_error}")
