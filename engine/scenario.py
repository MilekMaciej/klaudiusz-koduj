"""
scenario.py
===========

Sandbox: budowanie ręcznie ustawionej sytuacji przy stole na PRAWDZIWYM silniku.

Zamiast rozgrywać partię od zera, opisujesz stan: ręce graczy, musik, licytację,
rozegrane lewy (z meldunkami), kartę na stole, wyniki. Moduł:
  - dolosowuje brakujące karty tak, żeby talia zawsze była spójna (24 karty),
  - odtwarza historię jawnych ruchów (licytacja, musik, lewy, meldunki),
  - zwraca zwykły obiekt TysiacGame – partię można grać dalej według zasad,
    a kolejne rozdania są już losowe.

Format kart: kody silnika ("A_hearts") albo skróty: Ah, 10h (Th), Kc, Qd, Js, 9s
(h=kier ♥, d=karo ♦, c=trefl ♣, s=pik ♠). Gwiazdka przy karcie wyjścia w lewie
oznacza meldunek, np. "Qh*".

Specyfikacja (słownik / JSON), wszystkie pola opcjonalne:
    players      ["A", "B"]
    phase        "bidding" | "choose_musik" | "discard" | "declare" | "play"   (domyślnie "play")
    to_move      gracz, który ma teraz ruch                                    (domyślnie players[0])
    hands        {"A": "Ah 10h Kh", "B": ""}   najdłuższa podana ręka wyznacza liczbę kart na
                 rękach (= ile lew rozegrano); krótsze ręce i puste pola są dolosowywane
    musik        licytacja/wybór musiku: ["Kc Qc", "9s"]  |  później: odkryty musik "Kc Qc"
    bid          licytacja: aktualna stawka przeciwnika; później: wylicytowana kwota
    bidder       (play) rozgrywający; w fazach musik/discard/declare zawsze to_move
    contract     (play) kontrakt                                               (domyślnie bid lub 100)
    trump        (play) atut: hearts/diamonds/clubs/spades (h/d/c/s); domyślnie z ostatniego meldunku
    melds        (play) meldunki bez wskazania lewy, np. {"A": ["hearts"]}
    table        (play) karta wyłożona w bieżącej lewie przez przeciwnika gracza to_move
    tricks       (play) rozegrane lewy – tekst (linia = lewa) albo lista:
                     "Ah 9h"          wychodzący wynika z zasad (1. lewa: rozgrywający,
                                      potem zwycięzca poprzedniej), zwycięzcę liczy silnik
                     "A: Qh* 9h"      A wychodzi damą kier z meldunkiem
                     "B: Kc Ac > A"   wymuszony zwycięzca (ostrzeżenie, jeśli niezgodny z zasadami)
    tricks_done  (play) liczba rozegranych lew (brakujące lewy są dolosowane)
    scores       {"A": 0, "B": 0}
    round        numer rozdania (domyślnie 1)
    history      dodatkowe zdarzenia (np. z poprzednich rozdań) – lista słowników
    seed         ziarno losowania brakujących kart (powtarzalność)
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from engine.game import (
    HAND_SIZE, MELD_POINTS, SUITS, TRICKS_PER_ROUND, Card, IllegalMoveError, Phase, RoundState,
    Rules, TysiacGame, new_deck, sort_hand, trick_winner,
)
from engine.visibility import HISTORY_EVENT_KEYS, InformationLeakError, audit_observation


class ScenarioError(ValueError):
    """Opis scenariusza jest niespójny lub niepoprawny (komunikat dla użytkownika)."""


SHORT_SUITS = {"h": "hearts", "d": "diamonds", "c": "clubs", "s": "spades",
               "♥": "hearts", "♦": "diamonds", "♣": "clubs", "♠": "spades",
               "kier": "hearts", "karo": "diamonds", "trefl": "clubs", "pik": "spades"}
PHASES = (Phase.BIDDING, Phase.CHOOSE_MUSIK, Phase.DISCARD, Phase.DECLARE, Phase.PLAY)
SANDBOX_NOTE = ("Sytuacja ustawiona ręcznie w sandboxie – wcześniejsza część historii "
                "rozdania może być odtworzona umownie.")


# ---------------------------------------------------------------------------
# Parsowanie
# ---------------------------------------------------------------------------

def parse_card(token: str) -> Card:
    """'A_hearts' | 'Ah' | '10h' | 'Th' | 'Q♠' -> Card."""
    token = token.strip().rstrip("*")
    if "_" in token:
        try:
            return Card.from_code(token)
        except IllegalMoveError as exc:
            raise ScenarioError(str(exc)) from None
    if len(token) < 2:
        raise ScenarioError(f"Niepoprawna karta {token!r} – użyj np. Ah, 10h, Kc, 9s.")
    rank, suit = token[:-1].upper(), SHORT_SUITS.get(token[-1].lower())
    rank = "10" if rank == "T" else rank
    if suit is None:
        raise ScenarioError(f"Nieznany kolor w karcie {token!r} (użyj h/d/c/s).")
    try:
        return Card.from_code(f"{rank}_{suit}")
    except IllegalMoveError:
        raise ScenarioError(f"Niepoprawna karta {token!r} – rangi: 9, J, Q, K, 10, A.") from None


def parse_cards(value: str | list[str] | None) -> list[Card]:
    if not value:
        return []
    tokens = value if isinstance(value, list) else value.replace(",", " ").split()
    return [parse_card(t) for t in tokens if str(t).strip()]


def parse_suit(value: str | None) -> str | None:
    if value is None or str(value).strip().lower() in ("", "none", "brak", "-"):
        return None
    suit = SHORT_SUITS.get(str(value).strip().lower(), str(value).strip().lower())
    if suit not in SUITS:
        raise ScenarioError(f"Nieznany kolor {value!r} (hearts/diamonds/clubs/spades lub h/d/c/s).")
    return suit


def parse_revealed_musik(value: Any) -> list[Card]:
    """Odkryty musik: "Kc Qc" | ["K_clubs", "Q_clubs"] | ["Kc Qc", "..."] (wtedy pierwszy)."""
    if isinstance(value, list):
        if any(isinstance(v, list) or " " in str(v).strip() for v in value):
            value = value[0] if value else None
    cards = parse_cards(value)
    if len(cards) > 2:
        raise ScenarioError("Odkryty musik ma 2 karty.")
    return cards


@dataclass
class TrickSpec:
    """Jedna rozegrana lewa z opisu scenariusza."""

    cards: list[Card]
    leader: str | None = None          # None = wynika z zasad
    winner: str | None = None          # None = liczy silnik
    meld: bool = False                 # karta wyjścia zagrana z meldunkiem
    source: str = ""                   # tekst źródłowy (do komunikatów)


_TRICK_LINE = re.compile(r"^\s*(?:(?P<leader>[^:>]+?)\s*:)?\s*(?P<cards>[^>]+?)\s*(?:>\s*(?P<winner>\S+))?\s*$")


def parse_tricks(value: Any, players: tuple[str, str]) -> list[TrickSpec]:
    """Lewy z tekstu (linia = lewa) lub listy słowników/napisów."""
    if not value:
        return []
    items = value.splitlines() if isinstance(value, str) else list(value)
    tricks: list[TrickSpec] = []
    for item in items:
        if isinstance(item, dict):
            raw_cards = item.get("cards") or []
            tokens = raw_cards if isinstance(raw_cards, list) else str(raw_cards).split()
            leader, winner, text = item.get("leader"), item.get("winner"), str(item)
        else:
            text = str(item).strip()
            if not text or text.startswith("#"):
                continue
            match = _TRICK_LINE.match(text)
            if not match:
                raise ScenarioError(f"Nie rozumiem lewy {text!r} – format: 'A: Ah 9h' lub 'Ah 9h > B'.")
            leader, winner = match.group("leader"), match.group("winner")
            tokens = match.group("cards").replace(",", " ").split()
        if len(tokens) != 2:
            raise ScenarioError(f"Lewa {text!r} musi mieć dokładnie 2 karty (gra dwuosobowa).")
        for who in (leader, winner):
            if who is not None and who.strip() not in players:
                raise ScenarioError(f"Nieznany gracz {who!r} w lewie {text!r} (gracze: {', '.join(players)}).")
        meld = str(tokens[0]).strip().endswith("*")
        if any(str(t).strip().endswith("*") for t in tokens[1:]):
            raise ScenarioError(f"Meldować można tylko kartą wyjścia: {text!r}.")
        tricks.append(TrickSpec(cards=[parse_card(str(t)) for t in tokens],
                                leader=leader.strip() if leader else None,
                                winner=winner.strip() if winner else None,
                                meld=meld, source=text))
    return tricks


# ---------------------------------------------------------------------------
# Budowanie scenariusza
# ---------------------------------------------------------------------------

@dataclass
class _Builder:
    spec: dict[str, Any]
    players: tuple[str, str]
    rng: random.Random
    warnings: list[str] = field(default_factory=list)
    pool: list[Card] = field(default_factory=list)

    def opp(self, p: str) -> str:
        return self.players[1] if p == self.players[0] else self.players[0]

    def player(self, value: Any, field_name: str, default: str) -> str:
        if value in (None, ""):
            return default
        if value not in self.players:
            raise ScenarioError(f"Pole {field_name}: nieznany gracz {value!r} (gracze: {', '.join(self.players)}).")
        return value

    def take(self, n: int, what: str) -> list[Card]:
        if n > len(self.pool):
            raise ScenarioError(f"Za mało kart w talii ({what}) – sprawdź liczby kart w scenariuszu.")
        return [self.pool.pop() for _ in range(n)]

    def remove_from_pool(self, cards: list[Card]) -> None:
        for c in cards:
            if c in self.pool:
                self.pool.remove(c)

    def fill(self, given: list[Card], size: int, what: str) -> list[Card]:
        if len(given) > size:
            raise ScenarioError(f"{what}: podano {len(given)} kart, a w tej sytuacji powinno być {size}.")
        return given + self.take(size - len(given), what)


def _check_duplicates(groups: dict[str, list[Card]]) -> None:
    counter = Counter(c for cards in groups.values() for c in cards)
    dupes = sorted(c.code for c, n in counter.items() if n > 1)
    if dupes:
        where = {d: [name for name, cards in groups.items() if any(c.code == d for c in cards)] for d in dupes}
        raise ScenarioError("Te karty występują więcej niż raz: "
                            + "; ".join(f"{d} ({', '.join(w)})" for d, w in where.items()))


def build_scenario(spec: dict[str, Any], rules: Rules | None = None,
                   game_id: str | None = None) -> tuple[TysiacGame, list[str]]:
    """
    Buduje grę z opisu scenariusza. Zwraca (gra, ostrzeżenia).
    Rzuca ScenarioError z czytelnym komunikatem, gdy opis jest niespójny.
    """
    players = tuple(spec.get("players") or ("A", "B"))
    if len(players) != 2 or players[0] == players[1]:
        raise ScenarioError("Scenariusz wymaga dokładnie dwóch różnych graczy.")
    phase = spec.get("phase") or Phase.PLAY
    if phase not in PHASES:
        raise ScenarioError(f"Nieobsługiwana faza {phase!r} (dostępne: {', '.join(PHASES)}).")

    seed = spec.get("seed")
    b = _Builder(spec=spec, players=players, rng=random.Random(seed))  # type: ignore[arg-type]
    to_move = b.player(spec.get("to_move"), "to_move", players[0])
    other = b.opp(to_move)

    game = TysiacGame(players, rules=rules, seed=seed, game_id=game_id)  # type: ignore[arg-type]
    hands_spec = spec.get("hands") or {}
    unknown = set(hands_spec) - set(players)
    if unknown:
        raise ScenarioError(f"Ręce nieznanych graczy: {sorted(unknown)}.")
    hands_given = {p: parse_cards(hands_spec.get(p)) for p in players}

    if phase == Phase.PLAY:
        _build_play(game, b, to_move, other, hands_given)
    else:
        _build_before_play(game, b, phase, to_move, other, hands_given)

    # --- wyniki i numer rozdania ----------------------------------------------
    scores = spec.get("scores") or {}
    if isinstance(scores, (list, tuple)):
        scores = dict(zip(players, scores))
    game.scores = {p: int(scores.get(p, 0) or 0) for p in players}
    game.round.number = int(spec.get("round") or 1)
    for event in game._log:
        event["public"]["round"] = game.round.number if event["public"].get("round") == "_" else event["public"]["round"]

    _check_conservation(game)
    if not game.legal_moves(to_move):
        raise ScenarioError(f"W tym scenariuszu gracz {to_move} nie ma żadnego legalnego ruchu.")
    for p in players:  # ręcznie dopisana historia nie może zdradzać ukrytych kart
        try:
            audit_observation(game, p, game.get_state(p))
        except InformationLeakError as exc:
            raise ScenarioError(f"Scenariusz ujawnia graczowi {p} ukryte karty – popraw historię. ({exc})") from None
    return game, b.warnings


# --- fazy przed rozgrywką ---------------------------------------------------------

def _build_before_play(game: TysiacGame, b: _Builder, phase: str, to_move: str, other: str,
                       hands_given: dict[str, list[Card]]) -> None:
    spec = b.spec
    bid = spec.get("bid")
    bid = int(bid) if bid not in (None, "") else None
    if bid is not None and (bid < 100 or bid % 10):
        raise ScenarioError("Stawka licytacji musi być wielokrotnością 10, co najmniej 100.")

    musik_spec = spec.get("musik")
    if phase in (Phase.BIDDING, Phase.CHOOSE_MUSIK):
        musiki_given = ([parse_cards(m) for m in musik_spec] if isinstance(musik_spec, list)
                        else [parse_cards(musik_spec), []])
        musiki_given = (musiki_given + [[], []])[:2]
        revealed_given: list[Card] = []
    else:
        musiki_given = [[], []]
        revealed_given = parse_revealed_musik(musik_spec)

    groups = {f"ręka {p}": hands_given[p] for p in b.players}
    groups.update({"musik 1": musiki_given[0], "musik 2": musiki_given[1]})
    if phase == Phase.DISCARD:  # odkryty musik jest już w ręce rozgrywającego
        hands_given[to_move] = hands_given[to_move] + [c for c in revealed_given if c not in hands_given[to_move]]
        groups[f"ręka {to_move}"] = hands_given[to_move]
    else:
        groups["odkryty musik"] = [c for c in revealed_given if c not in hands_given[to_move]]
    _check_duplicates(groups)

    b.pool = [c for c in new_deck() if not any(c in g for g in groups.values())]
    b.rng.shuffle(b.pool)

    if phase == Phase.DECLARE and len([c for c in revealed_given if c not in hands_given[to_move]]) > 2:
        raise ScenarioError("Odkryty musik ma 2 karty.")
    bidder_hand_size = HAND_SIZE + 2 if phase == Phase.DISCARD else HAND_SIZE
    hands = {to_move: b.fill(hands_given[to_move], bidder_hand_size, f"Ręka {to_move}"),
             other: b.fill(hands_given[other], HAND_SIZE, f"Ręka {other}")}

    r = game.round
    r.hands = {p: sort_hand(h) for p, h in hands.items()}
    r.taken = {p: [] for p in b.players}
    r.melds = {p: [] for p in b.players}
    r.trick, r.played, r.trump, r.tricks_done, r.discards = [], [], None, 0, []
    game._log.clear()
    game.phase = phase
    _event(game, None, "sandbox_note", Phase.BIDDING, note=SANDBOX_NOTE)
    _extra_history(game, b)

    if phase == Phase.BIDDING:
        r.musiki = [b.fill(musiki_given[0], 2, "Musik 1"), b.fill(musiki_given[1], 2, "Musik 2")]
        dealer = b.player(b.spec.get("dealer"), "dealer", other if bid is None else to_move)
        if bid is None and dealer == to_move:
            b.warnings.append(f"Licytację zaczyna gracz nie-rozdający, a ruch ma rozdający {to_move} – "
                              f"przyjęto, że rozdawał {other}.")
            dealer = other
        r.dealer = dealer
        r.bid, r.high_bidder, r.bidder, r.bid_turn = (bid, other, None, to_move) if bid else (None, None, None, to_move)
        _event(game, None, "deal", Phase.BIDDING, dealer=dealer, first_bidder=b.opp(dealer))
        if bid:
            if bid > game.max_bid(other):
                b.warnings.append(f"{other} zalicytował {bid}, choć bez meldunku na ręce mógł najwyżej "
                                  f"{game.max_bid(other)} – w tym scenariuszu przyjęto to umownie.")
            _event(game, other, "bid", Phase.BIDDING, value=bid)
        return

    # od wyboru musiku: to_move jest rozgrywającym
    bid = bid or 100
    r.dealer = b.player(b.spec.get("dealer"), "dealer", other)
    r.bid, r.high_bidder, r.bidder, r.bid_turn = bid, to_move, to_move, None
    _event(game, None, "deal", Phase.BIDDING, dealer=r.dealer, first_bidder=b.opp(r.dealer))
    _event(game, to_move, "bid", Phase.BIDDING, value=bid)
    _event(game, other, "pass", Phase.BIDDING)
    _event(game, to_move, "bidding_won", Phase.BIDDING, value=bid)

    if phase == Phase.CHOOSE_MUSIK:
        r.musiki = [b.fill(musiki_given[0], 2, "Musik 1"), b.fill(musiki_given[1], 2, "Musik 2")]
        return

    if phase == Phase.DISCARD:  # odkryty musik jest w 12-kartowej ręce rozgrywającego
        revealed = revealed_given + b.rng.sample([c for c in r.hands[to_move] if c not in revealed_given],
                                                 2 - len(revealed_given))
    else:  # DECLARE: rozgrywający odłożył już 2 karty; karty odkrytego musiku spoza ręki
        #      zostały więc odłożone
        not_in_hand = [c for c in revealed_given if c not in r.hands[to_move]]
        r.discards = not_in_hand + b.take(2 - len(not_in_hand), "odłożone karty")
        own = r.hands[to_move] + r.discards
        revealed = revealed_given + b.rng.sample([c for c in own if c not in revealed_given],
                                                 2 - len(revealed_given))
    r.musiki = [revealed, b.take(2, "niewybrany musik")]
    r.chosen_musik = 0
    _event(game, to_move, "choose_musik", Phase.CHOOSE_MUSIK, value=0, cards=[c.code for c in revealed])
    for c in r.discards:
        _event(game, to_move, "discard", Phase.DISCARD, private={"card": c.code})


# --- rozgrywka ------------------------------------------------------------------------

def _build_play(game: TysiacGame, b: _Builder, to_move: str, other: str,
                hands_given: dict[str, list[Card]]) -> None:
    spec, players = b.spec, b.players
    raw_table = spec.get("table")
    raw_table_text = " ".join(raw_table) if isinstance(raw_table, list) else str(raw_table or "")
    table = parse_cards(raw_table)
    table_meld = raw_table_text.strip().endswith("*")   # "Qh*" = wyjście z meldunkiem
    if len(table) > 1:
        raise ScenarioError("Na stole może leżeć najwyżej 1 karta (wyjście przeciwnika).")
    t = len(table)
    leader_now = other if t else to_move
    tricks = parse_tricks(spec.get("tricks"), players)

    # --- ile lew rozegrano -> ile kart na rękach --------------------------------
    # Liczba kart na rękach: jawne tricks_done > najdłuższa podana ręka > liczba podanych lew.
    # (Krótsza ręka drugiego gracza jest dolosowywana do wymaganej liczby.)
    if spec.get("tricks_done") not in (None, ""):
        done = int(spec["tricks_done"])
    elif hands_given[to_move] or hands_given[other]:
        done = HAND_SIZE - max(len(hands_given[to_move]), len(hands_given[other]) + t)
    elif tricks:
        done = len(tricks)
    else:
        done = 0
    for p in players:
        if len(hands_given[p]) > HAND_SIZE:
            raise ScenarioError(f"Ręka {p}: podano {len(hands_given[p])} kart, a w rozgrywce jest najwyżej {HAND_SIZE}.")
    if not 0 <= done <= TRICKS_PER_ROUND - 1:
        raise ScenarioError(f"Liczba rozegranych lew musi być w zakresie 0–9 (wyszło {done}).")
    if len(tricks) > done:
        raise ScenarioError(f"Podano {len(tricks)} lew, a rozegranych ma być {done}.")
    size = {to_move: HAND_SIZE - done, other: HAND_SIZE - done - t}
    for p in players:
        if len(hands_given[p]) > size[p]:
            raise ScenarioError(
                f"Ręka {p}: podano {len(hands_given[p])} kart, a po {done} lewach"
                + (" i z kartą na stole" if p == other and t else "") + f" powinno być {size[p]}.")

    bidder = b.player(spec.get("bidder"), "bidder", to_move)
    defender = b.opp(bidder)
    contract = int(spec.get("contract") or spec.get("bid") or 100)
    if contract < 100 or contract % 10:
        raise ScenarioError("Kontrakt musi być wielokrotnością 10, co najmniej 100.")
    won_bid = min(int(spec.get("bid") or contract), contract)  # deklaracja może podnieść kwotę
    revealed_given = parse_revealed_musik(spec.get("musik"))

    groups = {f"ręka {p}": hands_given[p] for p in players}
    groups["stół"] = table
    groups["lewy"] = [c for tr in tricks for c in tr.cards]
    # Karty odkrytego musiku, których nie ma w ręce ani w zagranych kartach, rozgrywający
    # musiał odłożyć – trafiają do jego odłożonych kart.
    seen = [c for cards in groups.values() for c in cards]
    musik_discarded = [c for c in revealed_given if c not in seen]
    for c in revealed_given:
        if c in hands_given[defender]:
            raise ScenarioError(f"Karta odkrytego musiku {c.code} nie może być w ręce obrońcy {defender}.")
    groups["odłożone (z musiku)"] = musik_discarded
    _check_duplicates(groups)
    b.pool = [c for c in new_deck() if not any(c in g for g in groups.values())]
    b.rng.shuffle(b.pool)

    # --- dolosowanie brakujących lew (na początku rozdania) ---------------------
    filler_count = done - len(tricks)
    filler: list[TrickSpec] = []
    for i in range(filler_count):
        high, low = _filler_pair(b)
        # Dolosowane lewy bierze wychodzący (rozgrywający). Jeśli to ostatnia lewa, a teraz
        # wychodzi obrońca – ostatnią dolosowaną lewę bierze obrońca (zgodnie z zasadami).
        last = i == filler_count - 1 and not tricks and leader_now != bidder
        filler.append(TrickSpec(cards=[low, high] if last else [high, low], source="(dolosowana)"))
    all_tricks = filler + tricks

    # --- ręce, odłożone karty, musik ------------------------------------------------
    hands = {p: b.fill(hands_given[p], size[p], f"Ręka {p}") for p in players}
    discards = musik_discarded + b.take(2 - len(musik_discarded), "odłożone karty rozgrywającego")
    bidder_played = []  # uzupełniane niżej (karty zagrane przez rozgrywającego)

    # --- przebieg lew: wychodzący, meldunki, atut, zwycięzcy -------------------------
    r = game.round
    r.hands = {p: sort_hand(h) for p, h in hands.items()}
    r.taken = {p: [] for p in players}
    r.melds = {p: [] for p in players}
    r.played, r.trick, r.discards = [], [], discards
    game._log.clear()
    game.phase = Phase.PLAY
    _event(game, None, "sandbox_note", Phase.BIDDING, note=SANDBOX_NOTE)
    _extra_history(game, b)

    trump: str | None = None
    leader = bidder
    trick_events: list[tuple] = []
    for index, tr in enumerate(all_tricks, start=1):
        if tr.leader and tr.leader != leader:
            b.warnings.append(f"Lewa {tr.source!r}: wg zasad wychodził {leader}, przyjęto {tr.leader}.")
            leader = tr.leader
        follower = b.opp(leader)
        lead, follow = tr.cards
        if tr.meld:
            if lead.rank not in ("K", "Q"):
                raise ScenarioError(f"Lewa {tr.source!r}: meldować można tylko królem lub damą.")
            partner = Card("Q" if lead.rank == "K" else "K", lead.suit)
            later = hands[leader] + table + [c for x in all_tricks[index:] for c in x.cards]
            if partner not in later:
                b.warnings.append(f"Lewa {tr.source!r}: meldunek bez drugiej karty pary ({partner.code}) w ręce {leader}.")
            if index == 1 and not game.rules.meld_on_first_trick:
                b.warnings.append(f"Lewa {tr.source!r}: meldunek w pierwszej lewie (wg zasad niedozwolony).")
            trump = lead.suit
            r.melds[leader].append(lead.suit)
        computed = (leader, follower)[trick_winner(lead, follow, trump)]
        winner = tr.winner or computed
        if tr.winner and tr.winner != computed:
            b.warnings.append(f"Lewa {tr.source!r}: wg zasad bierze ją {computed}, przyjęto {tr.winner}.")
        # ostatnia lewa musi oddać wyjście graczowi, który wychodzi teraz
        if index == len(all_tricks) and winner != leader_now:
            b.warnings.append(f"Ostatnią lewę ({tr.source}) wg zasad bierze {winner}, ale teraz wychodzi "
                              f"{leader_now} – przyjęto, że lewę wziął {leader_now}.")
            winner = leader_now
        trick_events.append((index, leader, follower, lead, follow, tr.meld, winner))
        r.taken[winner] += [lead, follow]
        r.played += [lead, follow]
        if leader == bidder:
            bidder_played.append(lead)
        else:
            bidder_played.append(follow)
        leader = winner
    if not all_tricks and leader_now != bidder:
        b.warnings.append(f"Pierwszą lewę wychodzi rozgrywający ({bidder}) – przyjęto, że wychodzi {leader_now}.")

    # meldunki podane bez wskazania lewy
    loose_melds: list[tuple[str, str]] = []
    for p, suits in (spec.get("melds") or {}).items():
        if p not in players:
            raise ScenarioError(f"Meldunki nieznanego gracza {p!r}.")
        for suit in (suits if isinstance(suits, list) else str(suits).replace(",", " ").split()):
            suit = parse_suit(suit)
            if suit and suit not in r.melds[p]:
                r.melds[p].append(suit)
                loose_melds.append((p, suit))
                trump = suit
    if table_meld:
        card = table[0]
        if card.rank not in ("K", "Q"):
            raise ScenarioError("Meldować można tylko królem lub damą (karta na stole).")
        if card.suit not in r.melds[leader_now]:
            r.melds[leader_now].append(card.suit)
        trump = card.suit
    trump = parse_suit(spec.get("trump")) if spec.get("trump") not in (None, "") else trump
    r.trump = trump

    # --- odkryty i niewybrany musik ------------------------------------------------
    bidder_cards = hands[bidder] + discards + bidder_played + ([table[0]] if t and other == bidder else [])
    for c in revealed_given:
        if c not in bidder_cards:
            raise ScenarioError(f"Odkryty musik ({c.code}) musi należeć do rozgrywającego {bidder} "
                                f"(ręka, odłożone lub zagrane przez niego karty).")
    revealed = revealed_given + b.rng.sample([c for c in bidder_cards if c not in revealed_given],
                                             2 - len(revealed_given))
    r.musiki = [revealed, b.take(2, "niewybrany musik")]
    r.chosen_musik = 0

    # --- stan rozdania -----------------------------------------------------------------
    r.dealer = b.player(spec.get("dealer"), "dealer", defender)
    r.bid, r.contract = won_bid, contract
    r.high_bidder = r.bidder = bidder
    r.bid_turn = None
    r.tricks_done = done
    r.leader = leader_now
    if t:
        r.trick = [(leader_now, table[0])]
        r.played.append(table[0])

    # --- historia jawnych ruchów ---------------------------------------------------------
    _event(game, None, "deal", Phase.BIDDING, dealer=r.dealer, first_bidder=b.opp(r.dealer))
    _event(game, bidder, "bid", Phase.BIDDING, value=won_bid)
    _event(game, defender, "pass", Phase.BIDDING)
    _event(game, bidder, "bidding_won", Phase.BIDDING, value=r.bid)
    _event(game, bidder, "choose_musik", Phase.CHOOSE_MUSIK, value=0, cards=[c.code for c in revealed])
    for c in discards:
        _event(game, bidder, "discard", Phase.DISCARD, private={"card": c.code})
    _event(game, bidder, "declare", Phase.DECLARE, value=contract)
    for index, leader, follower, lead, follow, meld, winner in trick_events:
        if meld:
            _event(game, leader, "meld", Phase.PLAY, card=lead.code, suit=lead.suit,
                   points=MELD_POINTS[lead.suit], new_trump=lead.suit)
        else:
            _event(game, leader, "play_card", Phase.PLAY, card=lead.code)
        _event(game, follower, "play_card", Phase.PLAY, card=follow.code)
        _event(game, winner, "trick_won", Phase.PLAY, cards=[lead.code, follow.code],
               points=lead.points + follow.points, trick_no=index)
    for p, suit in loose_melds:
        _event(game, p, "meld", Phase.PLAY, suit=suit, points=MELD_POINTS[suit], new_trump=suit,
               note="meldunek ustawiony w sandboxie (lewa nieznana)")
    if t and table_meld:
        _event(game, leader_now, "meld", Phase.PLAY, card=table[0].code, suit=table[0].suit,
               points=MELD_POINTS[table[0].suit], new_trump=table[0].suit)
    elif t:
        _event(game, leader_now, "play_card", Phase.PLAY, card=table[0].code)


def _filler_pair(b: _Builder) -> tuple[Card, Card]:
    """Dwie karty na dolosowaną lewę: ten sam kolor, wychodząca silniejsza (lewa zgodna z zasadami)."""
    by_suit: dict[str, list[Card]] = {}
    for c in b.pool:
        by_suit.setdefault(c.suit, []).append(c)
    for cards in sorted(by_suit.values(), key=len, reverse=True):
        if len(cards) >= 2:
            low, high = sorted(b.rng.sample(cards, 2), key=lambda c: c.strength)
            b.remove_from_pool([high, low])
            return high, low
    a, c = b.take(2, "dolosowane lewy")
    return a, c


# --- historia ---------------------------------------------------------------------------

def _event(game: TysiacGame, player: str | None, action: str, phase: str,
           private: dict[str, Any] | None = None, **public: Any) -> None:
    """Dopisuje zdarzenie do dziennika gry (z fazą, w której się odbyło)."""
    event = {"seq": len(game._log) + 1, "round": "_", "phase": phase, "player": player, "action": action,
             **{k: v for k, v in public.items() if v is not None}}
    game._log.append({"public": event, "private": private or {}, "owner": player})


def _extra_history(game: TysiacGame, b: _Builder) -> None:
    """Zdarzenia podane ręcznie (np. z poprzednich rozdań) – tylko pola z whitelisty."""
    for raw in b.spec.get("history") or []:
        if not isinstance(raw, dict) or "action" not in raw:
            raise ScenarioError(f"Zdarzenie historii musi być obiektem z polem 'action': {raw!r}")
        dropped = set(raw) - HISTORY_EVENT_KEYS
        if dropped:
            b.warnings.append(f"Pominięto nieznane pola zdarzenia historii: {sorted(dropped)}")
        event = {k: v for k, v in raw.items() if k in HISTORY_EVENT_KEYS and k != "seq"}
        # karty zawsze w notacji silnika – inaczej "Ah" ominęłoby audyt widoczności
        if event.get("card"):
            event["card"] = parse_card(str(event["card"])).code
        for key in ("cards", "revealed_discards", "unused_musik"):
            if event.get(key):
                event[key] = [c.code for c in parse_cards(event[key])]
        event.setdefault("round", "_")
        event.setdefault("player", None)
        game._log.append({"public": {"seq": len(game._log) + 1, **event}, "private": {}, "owner": None})


def _check_conservation(game: TysiacGame) -> None:
    """Każda z 24 kart musi być dokładnie w jednym miejscu (ręce, stół, lewy, odłożone, musiki)."""
    r = game.round
    places: list[Card] = [c for h in r.hands.values() for c in h]
    places += [c for _, c in r.trick] + [c for cards in r.taken.values() for c in cards] + list(r.discards)
    if r.chosen_musik is None:
        places += [c for m in r.musiki for c in m]
    else:
        places += r.musiki[1 - r.chosen_musik]
    counter = Counter(places)
    if len(places) != 24 or any(n > 1 for n in counter.values()) or set(counter) != set(new_deck()):
        raise ScenarioError("Wewnętrzny błąd scenariusza: talia niespójna "
                            f"({len(places)} kart, duplikaty: {[c.code for c, n in counter.items() if n > 1]}).")
