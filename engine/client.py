"""
client.py
=========

Klient HTTP łączący silnik gry z backendem FastAPI oraz prosta pętla meczu.

Przepływ jednego kroku:
    1. silnik buduje stan gry dla aktywnego gracza  (TysiacGame.get_state)
    2. klient wysyła  {game_state, strategy}  ->  POST http://localhost:8000/play-turn
    3. backend pyta model (LiteLLM) i zwraca     {"move": {...}, ...}
    4. klient parsuje ruch, silnik go waliduje i wykonuje (TysiacGame.apply_move)

Uruchomienie meczu dwóch strategii (backend musi działać):
    python -m engine.client --strategy-a strategie/agresywna.txt --strategy-b strategie/ostrozna.txt

Test bez backendu i bez kosztów API (obaj gracze to bot heurystyczny):
    python -m engine.client --offline --seed 7
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

from engine.game import MELD_POINTS, Card, IllegalMoveError, TysiacGame, melds_in, trick_winner
from engine.protocol import API_VERSION, HEALTH_PATH, PLAY_TURN_PATH

logger = logging.getLogger("engine.client")

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Wyjątki – każdy mówi CO poszło nie tak i CO zrobić
# ---------------------------------------------------------------------------

class BackendError(RuntimeError):
    """Bazowy błąd komunikacji z backendem. Gra NIE wykonuje wtedy ruchu za bota."""


class BackendConnectionError(BackendError):
    """Backend nieosiągalny (nie działa, zły host/port, timeout)."""


class BackendProtocolError(BackendError):
    """Klient i backend się nie rozumieją: 404/405/422, inna wersja API, zły format odpowiedzi.
    To błąd konfiguracji – ponawianie nic nie da."""


class BackendConfigError(BackendError):
    """Backend działa, ale ma błędną konfigurację LLM (HTTP 500: brak/zły klucz, nieistniejący
    model, zły dostawca). Ponawianie nie pomoże – trzeba poprawić .env backendu."""


class BackendUnavailableError(BackendError):
    """Backend działa, ale dostawca LLM jest chwilowo niedostępny (HTTP 503: limit zapytań, timeout)."""


class ModelMoveError(BackendError):
    """Model mimo ponowień w backendzie nie zwrócił poprawnego ruchu (HTTP 502)."""


# ---------------------------------------------------------------------------
# Klient HTTP
# ---------------------------------------------------------------------------

def normalize_base_url(base_url: str) -> str:
    """
    'http://localhost:8000/'           -> 'http://localhost:8000'
    'http://localhost:8000/play-turn'  -> 'http://localhost:8000'  (częsta pomyłka w .env)
    Ścieżka endpointu jest doklejana WYŁĄCZNIE tutaj, ze wspólnej stałej PLAY_TURN_PATH.
    """
    url = base_url.strip().rstrip("/")
    if url.endswith(PLAY_TURN_PATH):
        logger.warning("BACKEND_URL zawiera już ścieżkę %s – usuwam ją (podaj sam adres serwera).",
                       PLAY_TURN_PATH)
        url = url[: -len(PLAY_TURN_PATH)]
    return url


class BackendClient:
    """
    Klient backendu FastAPI.

    - Ścieżki pochodzą ze wspólnego modułu `engine.protocol` (te same co w backendzie).
    - `verify()` sprawdza przy starcie, czy pod adresem działa ZGODNY backend.
    - Błędy przejściowe (brak połączenia, HTTP 503) są ponawiane z odstępem;
      błędy konfiguracji (404, 405, 422) przerywają od razu czytelnym wyjątkiem.
    """

    def __init__(self, base_url: str = DEFAULT_BACKEND_URL, timeout: float = 180.0,
                 retries: int = 2, retry_delay: float = 2.0) -> None:
        self.base_url = normalize_base_url(base_url)
        self.url = self.base_url + PLAY_TURN_PATH
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self._session = requests.Session()  # ponowne użycie połączenia TCP
        self._verified = False

    # --- weryfikacja kontraktu ---------------------------------------------------

    def verify(self) -> dict[str, Any]:
        """
        Sprawdza, czy backend działa i obsługuje TEN SAM kontrakt co klient.
        Wykrywa m.in. starą wersję backend/main.py bez endpointu /play-turn.
        """
        health_url = self.base_url + HEALTH_PATH
        try:
            resp = self._session.get(health_url, timeout=10)
        except requests.RequestException as exc:
            raise BackendConnectionError(
                f"Backend nie odpowiada pod {self.base_url} ({type(exc).__name__}).\n"
                f"  -> Uruchom go:  uvicorn backend.main:app --reload\n"
                f"  -> albo popraw BACKEND_URL / --backend (teraz: {self.base_url})"
            ) from exc

        if resp.status_code == 404:
            raise BackendProtocolError(
                f"Pod {self.base_url} odpowiada serwer, ale bez {HEALTH_PATH} – to nie jest backend "
                f"tej aplikacji albo BACKEND_URL zawiera zbędną ścieżkę (np. /api)."
            )
        try:
            info = resp.json()
        except ValueError:
            raise BackendProtocolError(f"{health_url} nie zwrócił JSON-a: {resp.text[:200]}") from None

        server_version, server_path = info.get("api_version"), info.get("play_turn_path")
        if server_version != API_VERSION or server_path != PLAY_TURN_PATH:
            raise BackendProtocolError(
                "Backend i klient mają NIEZGODNY kontrakt API:\n"
                f"  klient:  api_version={API_VERSION}, endpoint={PLAY_TURN_PATH}\n"
                f"  backend: api_version={server_version}, endpoint={server_path}\n"
                "  -> Najpewniej działa stara wersja backend/main.py. Podmień pliki backendu,\n"
                "     zrestartuj uvicorn i uruchom ponownie."
            )
        if info.get("llm_ready") is False:
            raise BackendConfigError(
                f"Backend działa, ale nie ma skonfigurowanego LLM (dostawca: {info.get('provider')}, "
                f"model: {info.get('model')}).\n"
                "  -> Uzupełnij klucz w .env backendu (np. LLM_PROVIDER=groq + GROQ_API_KEY) i zrestartuj "
                "uvicorn.\n  -> Diagnostyka:  python -m backend.llm_config --ping"
            )
        self._verified = True
        return info

    # --- ruch ----------------------------------------------------------------------

    def request_move(self, game_state: dict[str, Any], strategy: str,
                     model: str | None = None) -> dict[str, Any]:
        """
        Wysyła stan gry i strategię, zwraca słownik ruchu, np.:
            {"action": "play_card", "card": "10_hearts", "reasoning": "..."}
        """
        return self.request_turn(game_state, strategy, model)[0]

    def request_turn(self, game_state: dict[str, Any], strategy: str,
                     model: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Jak `request_move`, ale zwraca też metadane backendu: (ruch, {model, attempts, usage})."""
        if not self._verified:
            self.verify()

        payload: dict[str, Any] = {"game_state": game_state, "strategy": strategy}
        if model:
            payload["model"] = model

        last_exc: BackendError | None = None
        for attempt in range(1, self.retries + 2):
            try:
                return self._post_once(payload)
            except (BackendConnectionError, BackendUnavailableError) as exc:
                last_exc = exc  # błąd przejściowy – spróbuj jeszcze raz
                if attempt <= self.retries:
                    logger.warning("Błąd przejściowy (próba %d/%d): %s – ponawiam za %.0fs",
                                   attempt, self.retries + 1, exc, self.retry_delay * attempt)
                    time.sleep(self.retry_delay * attempt)
        assert last_exc is not None
        raise last_exc

    def _post_once(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            resp = self._session.post(self.url, json=payload, timeout=self.timeout)
        except requests.Timeout as exc:
            raise BackendConnectionError(
                f"Przekroczono limit {self.timeout:.0f}s oczekiwania na {self.url}") from exc
        except requests.RequestException as exc:
            raise BackendConnectionError(f"Brak połączenia z {self.url}: {type(exc).__name__}") from exc

        if resp.status_code != 200:
            detail = _error_detail(resp)
            status = resp.status_code
            if status == 404:
                raise BackendProtocolError(
                    f"HTTP 404 – backend nie ma endpointu POST {PLAY_TURN_PATH} ({self.url}).\n"
                    f"  Odpowiedź serwera: {detail}\n"
                    "  -> Sprawdź, czy działa aktualny backend/main.py i czy BACKEND_URL to sam "
                    "adres serwera (np. http://localhost:8000)."
                )
            if status == 405:
                raise BackendProtocolError(f"HTTP 405 – {self.url} nie przyjmuje metody POST: {detail}")
            if status == 422:
                raise BackendProtocolError(
                    f"HTTP 422 – backend odrzucił format stanu gry (niezgodne wersje silnika i backendu?): "
                    f"{detail}")
            if status == 500:
                raise BackendConfigError(f"{detail}\n  -> Popraw .env backendu; diagnostyka: "
                                         "python -m backend.llm_config --ping")
            if status == 502:
                raise ModelMoveError(f"Model nie podał poprawnego ruchu: {detail}")
            if status == 503:
                raise BackendUnavailableError(f"Dostawca LLM niedostępny: {detail}")
            raise BackendError(f"Nieoczekiwany HTTP {status} z {self.url}: {detail}")

        try:
            data = resp.json()
            move = data["move"]
        except (ValueError, KeyError, TypeError) as exc:
            raise BackendProtocolError(f"Nieoczekiwany format odpowiedzi: {resp.text[:300]}") from exc

        meta = {k: data.get(k) for k in ("provider", "model", "attempts", "usage", "fallback_used")}
        return parse_move(move), meta


def _error_detail(resp: requests.Response) -> str:
    """Treść błędu z odpowiedzi FastAPI ({"detail": ...}) albo surowy tekst."""
    try:
        detail = resp.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail if detail is not None else resp.text[:300])


def parse_move(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalizuje ruch do postaci przyjmowanej przez silnik:
    zostawia tylko znane pola i usuwa wartości null.
    """
    if not isinstance(raw, dict) or "action" not in raw:
        raise BackendProtocolError(f"Ruch bez pola 'action': {raw!r}")
    move = {k: raw.get(k) for k in ("action", "card", "value", "reasoning")}
    if isinstance(move["value"], str) and move["value"].lstrip("-").isdigit():
        move["value"] = int(move["value"])  # modele czasem zwracają "120" zamiast 120
    return {k: v for k, v in move.items() if v is not None}


def concretize(legal: dict[str, Any]) -> dict[str, Any]:
    """Zamienia opis legalnego ruchu na konkretny ruch (zakres -> najniższa wartość)."""
    move = {k: v for k, v in legal.items() if k not in ("min", "max", "step")}
    if "min" in legal:
        move["value"] = legal["min"]
    return move


# ---------------------------------------------------------------------------
# Agenci (kto podejmuje decyzję za gracza)
# ---------------------------------------------------------------------------

class Agent(Protocol):
    name: str

    def choose(self, state: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class LLMAgent:
    """Gracz sterowany przez model językowy i strategię użytkownika."""

    name: str
    strategy: str
    client: BackendClient
    model: str | None = None

    def choose(self, state: dict[str, Any]) -> dict[str, Any]:
        return self.client.request_move(state, self.strategy, self.model)


@dataclass
class RandomAgent:
    """Losowy gracz – do testów silnika bez backendu i bez kosztów API."""

    name: str
    rng: random.Random = field(default_factory=random.Random)

    def choose(self, state: dict[str, Any]) -> dict[str, Any]:
        legal = self.rng.choice(state["legal_moves"])
        move = concretize(legal)
        if "min" in legal:  # losowa wartość z zakresu licytacji (z przewagą niskich)
            steps = (legal["max"] - legal["min"]) // legal.get("step", 10)
            move["value"] = legal["min"] + legal.get("step", 10) * min(steps, self.rng.randint(0, 2))
        move["reasoning"] = "losowy ruch"
        return move


@dataclass
class HeuristicAgent:
    """
    Prosty bot regułowy – punkt odniesienia dla strategii LLM i tryb offline.
    (Gracze czysto losowi prawie nigdy nie ugrywają kontraktu, więc partia
    z nimi praktycznie nie dobiega do 1000 pkt.)
    """

    name: str

    @staticmethod
    def _card(code: str) -> Card:
        return Card.from_code(code)

    def choose(self, state: dict[str, Any]) -> dict[str, Any]:
        legal, phase = state["legal_moves"], state["phase"]
        hand = [self._card(c) for c in state["hand"]]

        if phase == "bidding":
            bid = next((m for m in legal if m["action"] == "bid"), None)
            meld_value = sum(MELD_POINTS[s] for s in melds_in(hand))
            estimate = sum(c.points for c in hand) + meld_value + 30  # +~30 za musik i lewy
            if bid and (bid["min"] <= estimate or not any(m["action"] == "pass" for m in legal)):
                return {"action": "bid", "value": bid["min"], "reasoning": f"szacunek {estimate}"}
            return {"action": "pass", "reasoning": f"szacunek {estimate} za niski"}

        if phase == "choose_musik":
            return {"action": "choose_musik", "value": 0, "reasoning": "w ciemno"}

        if phase == "discard":
            keep = {(r, s) for s in melds_in(hand) for r in ("K", "Q")}
            cand = [c for c in hand if (c.rank, c.suit) not in keep] or hand
            worst = min(cand, key=lambda c: (c.points, c.strength))
            return {"action": "discard", "card": worst.code, "reasoning": "najsłabsza karta"}

        if phase == "declare":
            return {"action": "declare", "value": legal[0]["min"], "reasoning": "minimum"}

        # --- rozgrywka ---------------------------------------------------------
        melds = [m for m in legal if m["action"] == "meld"]
        if melds:  # najcenniejszy meldunek, zagrywając damę (król zostaje na ręce)
            best = max(melds, key=lambda m: (MELD_POINTS[m["card"].split("_")[1]], m["card"].startswith("Q")))
            return {**best, "reasoning": "meldunek"}
        playable = [self._card(m["card"]) for m in legal]
        if not state["table"]:
            top = max(playable, key=lambda c: (c.strength, c.points))
            return {"action": "play_card", "card": top.code, "reasoning": "wychodzę najsilniejszą"}
        lead = self._card(state["table"][0]["card"])
        winners = [c for c in playable if trick_winner(lead, c, state["trump"]) == 1]
        if winners:
            pick = min(winners, key=lambda c: (c.suit == state["trump"], c.strength))
            return {"action": "play_card", "card": pick.code, "reasoning": "biorę najtańszą wygrywającą"}
        pick = min(playable, key=lambda c: (c.points, c.strength))
        return {"action": "play_card", "card": pick.code, "reasoning": "oddaję najtańszą"}


# ---------------------------------------------------------------------------
# Pętla meczu
# ---------------------------------------------------------------------------

class MatchAborted(RuntimeError):
    """
    Partia przerwana z powodu błędu API lub ruchu niezgodnego z zasadami.
    Niesie kontekst (gdzie przerwano) i częściowy wynik – do logów i debugowania.
    """

    def __init__(self, cause: Exception, round_no: int, phase: str, player: str,
                 partial: dict[str, Any]) -> None:
        self.cause, self.round_no, self.phase, self.player, self.partial = cause, round_no, phase, player, partial
        super().__init__(
            f"Partia przerwana w rozdaniu {round_no}, faza '{phase}', ruch gracza {player}:\n"
            f"  [{type(cause).__name__}] {cause}"
        )


def play_match(game: TysiacGame, agents: dict[str, Agent], max_rounds: int | None = None,
               fallback_on_model_error: bool = False) -> dict[str, Any]:
    """
    Rozgrywa partię do 1000 pkt (lub do `max_rounds` rozdań – limit kosztów API;
    wtedy `winner` = None, a o wyniku decydują punkty w `scores`).

    Obsługa błędów:
      - błąd połączenia / konfiguracji / dostawcy (BackendError) -> MatchAborted;
        silnik NIE wykonuje żadnego ruchu za bota,
      - model nie podał poprawnego ruchu (ModelMoveError) lub ruch odrzucił silnik
        (IllegalMoveError) -> domyślnie także MatchAborted; tylko przy jawnym
        `fallback_on_model_error=True` wykonywany jest pierwszy legalny ruch,
        oznaczony w logu jako "fallback".
    """
    decisions: list[dict[str, Any]] = []
    fallbacks = {p: 0 for p in game.players}

    def summary() -> dict[str, Any]:
        return {
            "game_id": game.game_id,
            "winner": game.winner,
            "scores": dict(game.scores),
            "rounds": game.round.number if game.is_over else game.round.number - 1,
            "fallback_moves": fallbacks,
            "decisions": decisions,
        }

    while not game.is_over:
        if max_rounds is not None and game.round.number > max_rounds:
            logger.info("Osiągnięto limit %d rozdań – przerywam partię.", max_rounds)
            break
        player = game.current_player
        assert player is not None
        state = game.get_state(player)
        fallback = False

        try:
            move = agents[player].choose(state)
            reasoning = move.pop("reasoning", "")
            game.apply_move(player, move)
        except (ModelMoveError, IllegalMoveError) as exc:
            if not fallback_on_model_error:
                raise MatchAborted(exc, state["round"], state["phase"], player, summary()) from exc
            logger.warning("[%s] niepoprawny ruch modelu (%s) – ruch zastępczy (opcja włączona)", player, exc)
            move = concretize(state["legal_moves"][0])
            reasoning = f"RUCH ZASTĘPCZY: {exc}"
            game.apply_move(player, move)
            fallbacks[player] += 1
            fallback = True
        except BackendError as exc:  # połączenie, 404/422, dostawca LLM – zawsze przerywamy
            raise MatchAborted(exc, state["round"], state["phase"], player, summary()) from exc

        decisions.append({"round": state["round"], "phase": state["phase"], "player": player,
                          "move": move, "reasoning": reasoning, "fallback": fallback})
        logger.info("r%-2d %-12s %-10s %s  | %s", state["round"], state["phase"], player,
                    json.dumps(move, ensure_ascii=False), reasoning[:80])

    return summary()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_strategy(path: str | None, default: str) -> str:
    return Path(path).read_text(encoding="utf-8") if path else default


def main() -> None:
    parser = argparse.ArgumentParser(description="Mecz Tysiąca: strategia A vs strategia B.")
    parser.add_argument("--strategy-a", help="Plik z promptem-strategią gracza A")
    parser.add_argument("--strategy-b", help="Plik z promptem-strategią gracza B")
    parser.add_argument("--model-a", help="Opcjonalny model LiteLLM dla gracza A")
    parser.add_argument("--model-b", help="Opcjonalny model LiteLLM dla gracza B")
    parser.add_argument("--backend", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--seed", type=int, default=None, help="Ziarno tasowania (powtarzalność)")
    parser.add_argument("--offline", action="store_true",
                        help="Bez backendu: obaj gracze to bot heurystyczny")
    parser.add_argument("--max-rounds", type=int, default=60, help="Limit rozdań (kontrola kosztów, 0 = bez limitu)")
    parser.add_argument("--fallback-on-model-error", action="store_true",
                        help="Gdy model nie poda poprawnego ruchu: zagraj pierwszy legalny zamiast przerywać")
    parser.add_argument("--log", help="Zapisz pełny przebieg partii do pliku JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    game = TysiacGame(("A", "B"), seed=args.seed)

    if args.offline:
        agents: dict[str, Agent] = {"A": HeuristicAgent("A"), "B": HeuristicAgent("B")}
    else:
        client = BackendClient(args.backend)
        try:
            client.verify()  # przerwij od razu, jeśli backend nie działa lub jest niezgodny
        except BackendError as exc:
            raise SystemExit(f"\n{exc}") from None
        agents = {
            "A": LLMAgent("A", _read_strategy(args.strategy_a, "Graj agresywnie, licytuj wysoko."),
                          client, args.model_a),
            "B": LLMAgent("B", _read_strategy(args.strategy_b, "Graj ostrożnie, licytuj nisko."),
                          client, args.model_b),
        }

    try:
        result = play_match(game, agents, max_rounds=args.max_rounds or None,
                            fallback_on_model_error=args.fallback_on_model_error)
    except MatchAborted as exc:
        raise SystemExit(f"\n{exc}") from None
    print(f"\nZwycięzca: {result['winner']}  wyniki: {result['scores']}  "
          f"rozdań: {result['rounds']}  ruchy zastępcze: {result['fallback_moves']}")
    if args.log:
        Path(args.log).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
