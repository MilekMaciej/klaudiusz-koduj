"""
visibility.py
=============

Filtr stanów gry (Partial Observability) – "anti-cheat" dla botów.

Silnik (`TysiacGame.get_state`) buduje obserwację dla JEDNEGO gracza. Ten moduł
pilnuje, żeby do modelu nigdy nie trafiło nic, czego ten gracz nie mógłby
wiedzieć przy prawdziwym stole. Działa niezależnie od `get_state` – to druga,
niezależna linia obrony (audyt), a nie ta sama logika napisana drugi raz:

1. WHITELIST pól   – obserwacja może zawierać wyłącznie znane, jawne pola
                     (`OBSERVATION_KEYS`, `HISTORY_EVENT_KEYS`, `LEGAL_MOVE_KEYS`).
                     Nowe pole w `get_state` trzeba świadomie dopisać tutaj.
2. AUDYT KART      – wyliczamy z PEŁNEGO stanu silnika zbiór kart ukrytych
                     przed graczem (ręka przeciwnika, niewybrany musik, odłożone
                     karty rozgrywającego) i sprawdzamy, że ŻADNA z nich nie
                     występuje w części obserwacji dotyczącej bieżącego rozdania.
3. SANITYZACJA     – backend (`/play-turn`) przyjmuje stan od klienta; usuwa
                     nieznane pola, a pola wyglądające na ukryte informacje
                     (np. `opponent_hand`) odrzuca jako próbę oszustwa.

Arena (`backend/arena.py`) wywołuje `audit_observation` przed KAŻDYM zapytaniem
do modelu – przy wykryciu wycieku ruch nie jest wysyłany (fail closed).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import tylko dla podpowiedzi typów (unikamy cyklu importów)
    from engine.game import TysiacGame


class InformationLeakError(RuntimeError):
    """Obserwacja zawiera informację, której gracz nie powinien znać."""


#: Kod karty w formacie silnika, np. "10_hearts".
CARD_RE = re.compile(r"\b(?:9|J|Q|K|10|A)_(?:hearts|diamonds|clubs|spades)\b")

#: Karta w zapisie skróconym (Ah, 10h, Th, Q♠) – sprawdzana w polach tekstowych (notatkach).
SHORT_CARD_RE = re.compile(r"(?<![\w_])(10|[9JQKAT])([hdcs♥♦♣♠])(?![\w])", re.IGNORECASE)
_SHORT_SUIT = {"h": "hearts", "d": "diamonds", "c": "clubs", "s": "spades",
               "♥": "hearts", "♦": "diamonds", "♣": "clubs", "♠": "spades"}

#: Pola obserwacji dozwolone do wysłania modelowi (TysiacGame.get_state).
OBSERVATION_KEYS = frozenset({
    "game_id", "player_id", "opponent_id", "phase", "round", "dealer", "your_role",
    "hand", "opponent_hand_size", "table", "trick_leader", "trump",
    "current_bid", "bid_winner", "your_max_bid", "contract", "revealed_musik",
    "tricks_played", "cards_played_this_round", "melds_declared",
    "trick_points", "round_points", "scores", "barrel_rounds", "winner",
    "history", "legal_moves", "rules_summary",
    "your_discards", "your_discard_points",   # tylko u rozgrywającego – jego własne karty
})

#: Pola pojedynczego zdarzenia w historii.
HISTORY_EVENT_KEYS = frozenset({
    "seq", "round", "phase", "player", "action", "card", "cards", "value", "suit", "points",
    "new_trump", "dealer", "first_bidder", "trick_no", "bidder", "contract", "contract_made",
    "round_points", "score_change", "barrel_penalties", "scores", "revealed_discards",
    "unused_musik", "note",
})

#: Pola opisu legalnego ruchu.
LEGAL_MOVE_KEYS = frozenset({"action", "card", "value", "min", "max", "step"})

#: Nazwy pól, które wyglądają na ukryte informacje – ich obecność w stanie od
#: klienta to błąd lub próba oszustwa (żądanie jest odrzucane, nie "czyszczone").
FORBIDDEN_KEY_RE = re.compile(
    r"(opponent|rival|enemy|other)_?(hand|cards)$|^hands$|^all_hands$|musiki|^deck$|hidden|"
    r"^unused_musik$|^discards$|opponent_discards|opponent_strategy|strategy_of",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Karty ukryte przed graczem
# ---------------------------------------------------------------------------

def hidden_cards(game: "TysiacGame", player: str) -> set[str]:
    """
    Karty BIEŻĄCEGO rozdania, których `player` nie może znać:
      - ręka przeciwnika (z wyjątkiem kart z odkrytego musiku, które przeciwnik wziął),
      - musiki przed ich odkryciem / niewybrany musik,
      - odłożone karty rozgrywającego (jeśli `player` jest obrońcą).
    """
    r = game.round
    opponent = game.opponent(player)
    own = {c.code for c in r.hands[player]}
    revealed = ({c.code for c in r.musiki[r.chosen_musik]} if r.chosen_musik is not None else set())

    hidden = {c.code for c in r.hands[opponent]}
    for index, musik in enumerate(r.musiki):
        if index != r.chosen_musik:
            hidden |= {c.code for c in musik}
    if r.bidder is not None and r.bidder != player:
        hidden |= {c.code for c in r.discards}
    return hidden - revealed - own


def _current_round_part(observation: dict[str, Any]) -> dict[str, Any]:
    """Część obserwacji opisująca bieżące rozdanie (historia poprzednich rozdań dotyczy
    innego tasowania – te same kody kart mogą tam legalnie występować)."""
    current = observation.get("round")
    part = {k: v for k, v in observation.items() if k not in ("history", "rules_summary")}
    part["history"] = [e for e in observation.get("history", []) if e.get("round") == current]
    return part


# ---------------------------------------------------------------------------
# Audyt (arena – przed każdym wywołaniem modelu)
# ---------------------------------------------------------------------------

def audit_observation(game: "TysiacGame", player: str, observation: dict[str, Any]) -> None:
    """
    Niezależna kontrola obserwacji wysyłanej do modelu gracza `player`.
    Rzuca InformationLeakError przy jakimkolwiek naruszeniu – wtedy obserwacja
    NIE może trafić do modelu.
    """
    extra = set(observation) - OBSERVATION_KEYS
    if extra:
        raise InformationLeakError(f"Nieznane pola w obserwacji gracza {player}: {sorted(extra)}")
    if observation.get("player_id") != player:
        raise InformationLeakError(
            f"Obserwacja dla {player} ma player_id={observation.get('player_id')!r} – pomylone konteksty botów.")
    own = {c.code for c in game.round.hands[player]}
    if set(observation.get("hand", [])) != own:
        raise InformationLeakError(f"Pole 'hand' gracza {player} nie odpowiada jego ręce.")
    for event in observation.get("history", []):
        bad = set(event) - HISTORY_EVENT_KEYS
        if bad:
            raise InformationLeakError(f"Nieznane pola w zdarzeniu historii: {sorted(bad)}")
    for move in observation.get("legal_moves", []):
        if set(move) - LEGAL_MOVE_KEYS:
            raise InformationLeakError(f"Nieznane pola w legal_moves: {move}")

    current = _current_round_part(observation)
    text = json.dumps(current, ensure_ascii=False)
    mentioned = set(CARD_RE.findall(text))
    for event in current["history"]:           # notatki: także karty zapisane skrótem
        for rank, suit in SHORT_CARD_RE.findall(str(event.get("note", ""))):
            rank = rank.upper()
            mentioned.add(f"{'10' if rank in ('T', '10') else rank}_{_SHORT_SUIT[suit.lower()]}")
    leaked = mentioned & hidden_cards(game, player)
    if leaked:
        raise InformationLeakError(
            f"WYCIEK: obserwacja gracza {player} zawiera karty ukryte przed nim: {sorted(leaked)}")


# ---------------------------------------------------------------------------
# Sanityzacja stanu przysłanego przez klienta (backend /play-turn)
# ---------------------------------------------------------------------------

def sanitize_observation(state: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Zwraca (oczyszczony_stan, usunięte_pola). Usuwa pola spoza whitelisty (także
    w zdarzeniach historii i legal_moves). Rzuca InformationLeakError, gdy stan
    zawiera pole wyglądające na ukryte informacje (np. 'opponent_hand').
    """
    def forbidden(keys: Any, where: str) -> None:
        bad = sorted(k for k in keys if FORBIDDEN_KEY_RE.search(str(k)))
        if bad:
            raise InformationLeakError(
                f"Stan gry zawiera pola z ukrytymi informacjami ({where}): {bad}. "
                f"Model może dostać wyłącznie obserwację jednego gracza.")

    forbidden(state.keys(), "główny poziom")
    dropped = [k for k in state if k not in OBSERVATION_KEYS]
    clean = {k: v for k, v in state.items() if k in OBSERVATION_KEYS}

    history = []
    for event in clean.get("history", []) or []:
        if isinstance(event, dict):
            forbidden(event.keys(), "historia")
            dropped += [f"history.{k}" for k in event if k not in HISTORY_EVENT_KEYS]
            history.append({k: v for k, v in event.items() if k in HISTORY_EVENT_KEYS})
    if "history" in clean:
        clean["history"] = history

    if "legal_moves" in clean:
        moves = []
        for move in clean.get("legal_moves") or []:
            if isinstance(move, dict):
                dropped += [f"legal_moves.{k}" for k in move if k not in LEGAL_MOVE_KEYS]
                moves.append({k: v for k, v in move.items() if k in LEGAL_MOVE_KEYS})
        clean["legal_moves"] = moves
    return clean, sorted(set(dropped))
