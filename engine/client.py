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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

from engine.game import MELD_POINTS, Card, IllegalMoveError, TysiacGame, melds_in, trick_winner

logger = logging.getLogger("engine.client")

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PLAY_TURN_PATH = "/play-turn"


class BackendError(RuntimeError):
    """Backend nie zwrócił użytecznego ruchu (błąd sieci, HTTP lub formatu)."""


# ---------------------------------------------------------------------------
# Klient HTTP
# ---------------------------------------------------------------------------

class BackendClient:
    """Cienka warstwa nad `requests` do komunikacji z backendem FastAPI."""

    def __init__(self, base_url: str = DEFAULT_BACKEND_URL, timeout: float = 180.0) -> None:
        self.url = base_url.rstrip("/") + PLAY_TURN_PATH
        self.timeout = timeout
        self._session = requests.Session()  # ponowne użycie połączenia TCP

    def request_move(self, game_state: dict[str, Any], strategy: str,
                     model: str | None = None) -> dict[str, Any]:
        """
        Wysyła stan gry i strategię, zwraca słownik ruchu, np.:
            {"action": "play_card", "card": "10_hearts", "reasoning": "..."}
        """
        payload: dict[str, Any] = {"game_state": game_state, "strategy": strategy}
        if model:
            payload["model"] = model

        try:
            resp = self._session.post(self.url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise BackendError(f"Brak połączenia z backendem ({self.url}): {exc}") from exc

        if resp.status_code != 200:
            # FastAPI zwraca błędy jako {"detail": ...}
            try:
                detail = resp.json().get("detail")
            except ValueError:
                detail = resp.text[:500]
            raise BackendError(f"Backend zwrócił HTTP {resp.status_code}: {detail}")

        try:
            data = resp.json()
            move = data["move"]
        except (ValueError, KeyError, TypeError) as exc:
            raise BackendError(f"Nieoczekiwany format odpowiedzi: {resp.text[:500]}") from exc

        return parse_move(move)


def parse_move(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalizuje ruch do postaci przyjmowanej przez silnik:
    zostawia tylko znane pola i usuwa wartości null.
    """
    if not isinstance(raw, dict) or "action" not in raw:
        raise BackendError(f"Ruch bez pola 'action': {raw!r}")
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

def play_match(game: TysiacGame, agents: dict[str, Agent], max_rounds: int | None = None) -> dict[str, Any]:
    """
    Rozgrywa partię do 1000 pkt (lub do `max_rounds` rozdań – limit kosztów API;
    wtedy `winner` = None, a o wyniku decydują punkty w `scores`).
    Gdy agent zwróci błędny ruch albo backend zawiedzie, wykonywany jest ruch
    awaryjny (pierwszy legalny), a fakt ten trafia do logu – partia się nie zawiesza.

    Zwraca podsumowanie: zwycięzca, wyniki, liczba ruchów awaryjnych, log decyzji.
    """
    decisions: list[dict[str, Any]] = []
    fallbacks = {p: 0 for p in game.players}

    while not game.is_over:
        if max_rounds is not None and game.round.number > max_rounds:
            logger.info("Osiągnięto limit %d rozdań – przerywam partię.", max_rounds)
            break
        player = game.current_player
        assert player is not None
        state = game.get_state(player)

        try:
            move = agents[player].choose(state)
            reasoning = move.pop("reasoning", "")
            game.apply_move(player, move)
            fallback = False
        except (BackendError, IllegalMoveError) as exc:
            logger.warning("[%s] ruch odrzucony (%s) – wykonuję ruch awaryjny", player, exc)
            move = concretize(state["legal_moves"][0])
            reasoning = f"RUCH AWARYJNY: {exc}"
            game.apply_move(player, move)
            fallbacks[player] += 1
            fallback = True

        decisions.append({"round": state["round"], "phase": state["phase"], "player": player,
                          "move": move, "reasoning": reasoning, "fallback": fallback})
        logger.info("r%-2d %-12s %-10s %s  | %s", state["round"], state["phase"], player,
                    json.dumps(move, ensure_ascii=False), reasoning[:80])

    return {
        "game_id": game.game_id,
        "winner": game.winner,
        "scores": game.scores,
        "rounds": game.round.number if game.is_over else game.round.number - 1,
        "fallback_moves": fallbacks,
        "decisions": decisions,
    }


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
    parser.add_argument("--max-rounds", type=int, default=None, help="Limit rozdań (kontrola kosztów)")
    parser.add_argument("--log", help="Zapisz pełny przebieg partii do pliku JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    game = TysiacGame(("A", "B"), seed=args.seed)

    if args.offline:
        agents: dict[str, Agent] = {"A": HeuristicAgent("A"), "B": HeuristicAgent("B")}
    else:
        client = BackendClient(args.backend)
        agents = {
            "A": LLMAgent("A", _read_strategy(args.strategy_a, "Graj agresywnie, licytuj wysoko."),
                          client, args.model_a),
            "B": LLMAgent("B", _read_strategy(args.strategy_b, "Graj ostrożnie, licytuj nisko."),
                          client, args.model_b),
        }

    result = play_match(game, agents, max_rounds=args.max_rounds)
    print(f"\nZwycięzca: {result['winner']}  wyniki: {result['scores']}  "
          f"rozdań: {result['rounds']}  ruchy awaryjne: {result['fallback_moves']}")
    if args.log:
        Path(args.log).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
