"""
solver_bridge.py
================

Most między silnikiem Pythona (`engine/game.py`) a solverem C++ (`solver/tysiac_solver.cpp`).

Solver liczy dla gracza, który ma ruch w ROZGRYWCE:
    Q(a)   – średnia wartość double-dummy ruchu a po losowaniu "światów" (rozkładów
             ukrytych kart) zgodnych z tym, co gracz WIE w chwili decyzji,
    max Q  – wartość najlepszego ruchu,
    LOSS   – max Q − Q(zagrany ruch): ile punktów (różnicy wyników rozdania:
             rozgrywający − obrońca, z perspektywy gracza) kosztował wybór.

Obserwacja przekazywana do solvera zawiera wyłącznie wiedzę gracza (rękę, zagrane
karty, odkryty musik, puste kolory przeciwnika wynikające z historii lew) – to
uczciwa ocena decyzji przy niepełnej informacji, a nie "podglądanie" kart.

Notacja solvera: kolor S/C/D/H (pik/trefl/karo/kier) + ranga 9/W/D/K/T/A.
Uwaga: zasada beczki (800+) nie jest modelowana – solver ocenia wynik jednego rozdania.
"""

from __future__ import annotations

from typing import Any

from engine.game import CARD_POINTS, MELD_POINTS, SUITS, Card, Phase, TysiacGame

SUIT_TO_SOLVER = {"spades": "S", "clubs": "C", "diamonds": "D", "hearts": "H"}
RANK_TO_SOLVER = {"9": "9", "J": "W", "Q": "D", "K": "K", "10": "T", "A": "A"}
SOLVER_TO_SUIT = {v: k for k, v in SUIT_TO_SOLVER.items()}
SOLVER_TO_RANK = {v: k for k, v in RANK_TO_SOLVER.items()}


def to_solver(card: Card | str) -> str:
    """Card('10','hearts') / '10_hearts' -> 'HT'."""
    c = card if isinstance(card, Card) else Card.from_code(card)
    return SUIT_TO_SOLVER[c.suit] + RANK_TO_SOLVER[c.rank]


def from_solver(code: str) -> str:
    """'HT' -> '10_hearts'."""
    return f"{SOLVER_TO_RANK[code[1]]}_{SOLVER_TO_SUIT[code[0]]}"


def _cards(cards) -> str:
    items = [to_solver(c) for c in cards]
    return " ".join(items) if items else "-"


def solver_rules(game: TysiacGame) -> list[str]:
    """Flagi zasad solvera odpowiadające zasadom silnika (game.rules)."""
    rules = game.rules
    return [
        f"mustbeat {int(rules.must_beat_in_suit)}",
        f"musttrump {int(rules.must_trump)}",
        f"meldfirst {int(rules.meld_on_first_trick)}",
        "dead_to_last 0",   # odłożone karty -> rozgrywający, niewybrany musik -> nikt
    ]


def _check_play_phase(game: TysiacGame) -> None:
    r = game.round
    if game.phase != Phase.PLAY or r.bidder is None or r.contract is None:
        raise ValueError("Solver ocenia tylko ruchy w fazie rozgrywki.")


def _known_points(game: TysiacGame, player: str) -> int:
    """Punkty z wziętych lew + meldunki (jawne dla obu graczy)."""
    r = game.round
    return sum(c.points for c in r.taken[player]) + sum(MELD_POINTS[s] for s in r.melds[player])


def current_round_tricks(game: TysiacGame) -> list[dict[str, Any]]:
    """
    Odtwarza z historii bieżącego rozdania kolejne zagrania (z atutem obowiązującym
    w chwili zagrania). Zwraca listę lew: {lead, follow, lead_player, follow_player, trump}.
    Ostatnia, niepełna lewa (karta na stole) ma follow = None.
    """
    current = game.round.number
    trump: str | None = None
    tricks: list[dict[str, Any]] = []
    for entry in game._log:
        ev = entry["public"]
        if ev.get("round") != current or ev.get("action") not in ("play_card", "meld") or not ev.get("card"):
            if ev.get("round") == current and ev.get("action") == "meld" and ev.get("suit"):
                trump = ev["suit"]          # meldunek "bez lewy" (sandbox)
            continue
        card = Card.from_code(ev["card"])
        if ev["action"] == "meld":
            trump = card.suit
        if tricks and tricks[-1]["follow"] is None:
            tricks[-1].update(follow=card, follow_player=ev["player"], trump=trump)
        else:
            tricks.append({"lead": card, "lead_player": ev["player"], "follow": None,
                           "follow_player": None, "trump": trump})
    return tricks


def opponent_voids(game: TysiacGame, opponent: str) -> set[str]:
    """Kolory, których przeciwnik na pewno nie ma (nie dołożył do koloru / nie dał atutu)."""
    voids: set[str] = set()
    for t in current_round_tricks(game):
        if t["follow"] is None or t["follow_player"] != opponent:
            continue
        lead, follow, trump = t["lead"], t["follow"], t["trump"]
        if follow.suit != lead.suit:
            voids.add(lead.suit)
            if game.rules.must_trump and trump and lead.suit != trump and follow.suit != trump:
                voids.add(trump)
    return voids


def build_q_input(game: TysiacGame, player: str, move: dict[str, Any] | None = None,
                  samples: int = 60, seed: int = 1, use_voids: bool = True) -> str:
    """
    Wejście trybu `q` solvera dla gracza `player`, który ma teraz ruch w rozgrywce.
    Wywołuj PRZED wykonaniem ruchu (`move` = ruch do oceny).
    """
    _check_play_phase(game)
    r = game.round
    if game.current_player != player:
        raise ValueError(f"Teraz ruch ma {game.current_player}, nie {player}.")
    idx = {p: i for i, p in enumerate(game.players)}
    opponent = game.opponent(player)
    declarer = r.bidder
    hand = r.hands[player]
    played = set(r.played)                                    # zawiera kartę na stole
    own_discards = list(r.discards) if player == declarer else []
    revealed = r.musiki[r.chosen_musik] if r.chosen_musik is not None else []

    all_cards = {Card(rank, suit) for suit in SUITS for rank in CARD_POINTS}
    unseen = all_cards - set(hand) - played - set(own_discards)
    opp_size = len(r.hands[opponent])
    pts = {p: _known_points(game, p) for p in game.players}
    if player == declarer:
        pts[declarer] += sum(c.points for c in own_discards)  # rozgrywający zna swoje odłożone karty

    lines = [
        f"declarer {idx[declarer]}", f"contract {r.contract}", *solver_rules(game),
        f"actor {idx[player]}",
        f"hand {_cards(sorted(hand, key=lambda c: (c.suit, c.strength)))}",
        f"unseen {_cards(sorted(unseen, key=lambda c: (c.suit, c.strength)))}",
        f"opp_size {opp_size}",
        f"pts {pts[game.players[0]]} {pts[game.players[1]]}",
        f"hidden_dead {len(unseen) - opp_size}",
        f"hidden_disc {0 if player == declarer else len(r.discards)}",
        f"not_unused {_cards([c for c in revealed if c in unseen])}",
        "dead_pts 0",
        f"led {to_solver(r.trick[0][1]) if r.trick else '-'}",
        f"trump {SUIT_TO_SOLVER[r.trump] if r.trump else '-'}",
        f"tn {r.tricks_done}",
        f"samples {samples}", f"seed {seed}", "output json",
    ]
    voids = opponent_voids(game, opponent) if use_voids else set()
    if voids:
        lines.append("voids " + " ".join(SUIT_TO_SOLVER[s] for s in sorted(voids)))
    if move:
        lines.append(f"model {to_solver(move['card'])}" + (" meld" if move.get("action") == "meld" else ""))
    return "\n".join(lines) + "\n"


def build_solve_input(game: TysiacGame) -> str:
    """Wejście trybu `solve` – PEŁNA informacja (prawdziwe ręce), dla gracza z ruchem."""
    _check_play_phase(game)
    r = game.round
    idx = {p: i for i, p in enumerate(game.players)}
    pts = {p: _known_points(game, p) for p in game.players}
    pts[r.bidder] += sum(c.points for c in r.discards)
    return "\n".join([
        f"declarer {idx[r.bidder]}", f"contract {r.contract}", *solver_rules(game),
        f"h0 {_cards(r.hands[game.players[0]])}", f"h1 {_cards(r.hands[game.players[1]])}",
        f"turn {idx[game.current_player]}",
        f"led {to_solver(r.trick[0][1]) if r.trick else '-'}",
        f"trump {SUIT_TO_SOLVER[r.trump] if r.trump else '-'}",
        f"tn {r.tricks_done}",
        f"pts {pts[game.players[0]]} {pts[game.players[1]]}",
        "dead_pts 0",
    ]) + "\n"


def parse_q_output(data: dict[str, Any]) -> dict[str, Any]:
    """JSON z solvera -> karty w notacji silnika + akcje (play_card / meld)."""
    def move(m: dict[str, Any]) -> dict[str, Any]:
        return {"card": from_solver(m["card"]), "action": "meld" if m.get("meld") else "play_card",
                **{k: round(m[k], 2) for k in ("q", "se", "loss") if k in m}}
    out = {
        "moves": sorted((move(m) for m in data["moves"]), key=lambda m: -m["q"]),
        "best": move(data["best"]),
        "max_q": round(data["best"]["q"], 2),
        "model": move(data["model"]) if data.get("model") else None,
        "model_legal": data.get("model_legal", True),
        "worlds": data.get("worlds"),
        "nodes": data.get("nodes"),
        "solver_ms": data.get("ms"),
    }
    if out["model"]:
        out["q"] = out["model"]["q"]
        out["loss"] = out["model"]["loss"]
    return out


# ---------------------------------------------------------------------------
# Walidacja wejścia (zanim trafi do procesu C++)
# ---------------------------------------------------------------------------

class SolverInputError(ValueError):
    """Wejście dla solvera jest niepoprawne lub niespójne – nie wysyłamy go do C++."""


_VALID_CARD = {s + r for s in "SCDH" for r in "9WDKTA"}


def parse_kv(text: str) -> dict[str, list[str]]:
    """Ten sam format co readKV w C++: linia = klucz + wartości oddzielone spacjami, '#' = komentarz."""
    kv: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, *values = line.split()
        kv[key] = values
    return kv


def validate_q_input(text: str) -> dict[str, list[str]]:
    """
    Sprawdza wejście trybu q dokładnie tak, jak zinterpretuje je readKV/modeQ w C++.
    Rzuca SolverInputError z czytelnym opisem; zwraca sparsowane pola.
    """
    kv = parse_kv(text)
    required = ("declarer", "contract", "actor", "hand", "unseen", "opp_size", "hidden_dead",
                "led", "trump", "tn", "samples")
    missing = [k for k in required if k not in kv]
    if missing:
        raise SolverInputError(f"brak pól: {', '.join(missing)}")

    def ints(key: str) -> list[int]:
        try:
            return [int(v) for v in kv[key]]
        except ValueError:
            raise SolverInputError(f"pole '{key}' musi zawierać liczby całkowite: {kv[key]}") from None

    def cards(key: str) -> list[str]:
        values = [v for v in kv.get(key, []) if v != "-"]
        bad = [v for v in values if v not in _VALID_CARD]
        if bad:
            raise SolverInputError(f"pole '{key}': niepoprawne karty {bad} (format solvera: kolor SCDH + ranga 9WDKTA)")
        if len(set(values)) != len(values):
            raise SolverInputError(f"pole '{key}': powtórzone karty")
        return values

    for key in ("declarer", "contract", "actor", "opp_size", "hidden_dead", "tn", "samples"):
        if len(kv[key]) != 1:
            raise SolverInputError(f"pole '{key}' musi mieć dokładnie jedną wartość")
    declarer, contract, actor = ints("declarer")[0], ints("contract")[0], ints("actor")[0]
    opp_size, hidden_dead, tn, samples = ints("opp_size")[0], ints("hidden_dead")[0], ints("tn")[0], ints("samples")[0]
    hidden_disc = ints("hidden_disc")[0] if kv.get("hidden_disc") else 0
    hand, unseen = cards("hand"), cards("unseen")
    led = cards("led")
    if declarer not in (0, 1) or actor not in (0, 1):
        raise SolverInputError("declarer i actor muszą być 0 lub 1")
    if contract <= 0 or contract % 10:
        raise SolverInputError(f"kontrakt {contract} musi być dodatnią wielokrotnością 10")
    if not hand:
        raise SolverInputError("pusta ręka gracza – nie ma czego oceniać")
    if set(hand) & set(unseen):
        raise SolverInputError(f"te same karty w ręce i w unseen: {sorted(set(hand) & set(unseen))}")
    if len(led) > 1 or (led and (led[0] in hand or led[0] in unseen)):
        raise SolverInputError("karta wyjścia (led) musi być jedna i nie może być w ręce ani w unseen")
    if opp_size != len(hand) - (1 if led else 0):
        raise SolverInputError(f"opp_size={opp_size} nie pasuje do ręki ({len(hand)} kart"
                               f"{', przeciwnik już wyszedł' if led else ''})")
    if hidden_dead != len(unseen) - opp_size:
        raise SolverInputError(f"hidden_dead={hidden_dead}, a unseen − opp_size = {len(unseen) - opp_size}")
    if not 0 <= hidden_disc <= hidden_dead:
        raise SolverInputError("hidden_disc musi być w zakresie 0..hidden_dead")
    if not 0 <= tn <= 9 or samples < 1:
        raise SolverInputError("tn musi być w zakresie 0..9, samples ≥ 1")
    if kv["trump"] and kv["trump"][0] not in ("S", "C", "D", "H", "-"):
        raise SolverInputError(f"zły atut {kv['trump'][0]!r} (S/C/D/H lub -)")
    if len(hand) + len(unseen) + len(led) > 24:
        raise SolverInputError("więcej niż 24 karty w obserwacji")
    model = kv.get("model") or []
    if model and model[0] not in hand:
        raise SolverInputError(f"oceniany ruch {model[0]} nie jest w ręce gracza")
    return kv
