"""
game.py
=======

Silnik gry "Tysiąc" w wariancie WYŁĄCZNIE DLA 2 GRACZY.

Moduł nie ma żadnego interfejsu (ani terminalowego, ani graficznego).
Udostępnia trzy rzeczy, których potrzebuje reszta platformy:

    game.current_player            -> kto ma teraz ruch
    game.get_state(player_id)      -> kompletny, JAWNY stan gry dla tego gracza (dict -> JSON)
    game.apply_move(player_id, mv) -> wykonanie ruchu (z pełną walidacją)

Przyjęte zasady (klasyczny Tysiąc dwuosobowy)
--------------------------------------------
Talia:      24 karty: 9, J, Q, K, 10, A w czterech kolorach.
Siła kart:  A > 10 > K > Q > J > 9.
Punkty:     A=11, 10=10, K=4, Q=3, J=2, 9=0  (razem 120 w talii).
Meldunki:   para K+Q w jednym kolorze: kier 100, karo 80, trefl 60, pik 40.

Przebieg rozdania:
 1. ROZDANIE    – każdy gracz dostaje 10 kart, pozostałe 4 karty tworzą
                  dwa zakryte musiki po 2 karty. Rozdający zmienia się co rundę.
 2. LICYTACJA   – zaczyna gracz NIE-rozdający i musi zalicytować co najmniej 100
                  (nie może spasować). Potem gracze na zmianę podbijają o
                  wielokrotność 10 albo pasują. Pas kończy licytację.
                  Nie wolno licytować powyżej 120 bez meldunku na ręce:
                  maksymalna stawka = 120 + suma meldunków posiadanych na ręce.
 3. MUSIK       – zwycięzca licytacji wybiera jeden z dwóch musików, odkrywa go
                  (obaj gracze go widzą) i bierze do ręki (12 kart).
                  Drugi musik pozostaje zakryty i nie bierze udziału w grze.
 4. ODKŁADANIE  – rozgrywający odkłada zakryte 2 karty (wraca do 10 kart).
                  Odłożone karty liczą się do jego punktów (jak wzięta lewa).
 5. DEKLARACJA  – rozgrywający może podwyższyć kontrakt (wielokrotność 10),
                  maksymalnie do 120 + meldunki na ręce.
 6. ROZGRYWKA   – 10 lew. Pierwszą lewę wychodzi rozgrywający, kolejną ten,
                  kto wziął poprzednią. Obowiązek dokładania do koloru; przy
                  braku koloru obowiązek dania atutu (jeśli jest atut i gracz
                  go ma); w przeciwnym razie dowolna karta.
                  MELDUNEK: gracz WYCHODZĄCY do lewy, mając K i Q w jednym
                  kolorze, może zagrać jedną z nich i zameldować – dostaje punkty
                  meldunku, a ten kolor staje się atutem (do kolejnego meldunku).
                  Domyślnie nie można meldować w pierwszej lewie rozdania.
 7. PUNKTACJA   – rozgrywający: jeśli (punkty z kart + meldunki) >= kontrakt,
                  dostaje +kontrakt, w przeciwnym razie -kontrakt.
                  Obrońca: punkty z kart + meldunki, zaokrąglone do pełnych 10.
                  BECZKA: gracz z wynikiem >= 800 dopisuje punkty tylko jako
                  rozgrywający, który ugrał kontrakt; obrońca nie może
                  przekroczyć 800. Kto przez 3 rozdania nie zejdzie z beczki
                  (nie wygra), traci 120 pkt.
 8. KONIEC GRY  – wygrywa gracz, który osiągnie 1000 punktów.

Elementy, które w różnych regionach bywają różne, są zebrane w klasie `Rules`.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Stałe gry
# ---------------------------------------------------------------------------

SUITS: tuple[str, ...] = ("hearts", "diamonds", "clubs", "spades")
SUITS_PL: dict[str, str] = {"hearts": "kier", "diamonds": "karo", "clubs": "trefl", "spades": "pik"}

#: Rangi w kolejności rosnącej siły (indeks = siła karty w lewie).
RANKS: tuple[str, ...] = ("9", "J", "Q", "K", "10", "A")

CARD_POINTS: dict[str, int] = {"9": 0, "J": 2, "Q": 3, "K": 4, "10": 10, "A": 11}
MELD_POINTS: dict[str, int] = {"hearts": 100, "diamonds": 80, "clubs": 60, "spades": 40}

HAND_SIZE = 10
MUSIK_SIZE = 2
TRICKS_PER_ROUND = 10


class Phase:
    """Fazy rozdania (stałe tekstowe – trafiają wprost do JSON-a)."""

    BIDDING = "bidding"            # licytacja
    CHOOSE_MUSIK = "choose_musik"  # wybór jednego z dwóch musików
    DISCARD = "discard"            # odłożenie 2 kart
    DECLARE = "declare"            # ostateczna deklaracja kontraktu
    PLAY = "play"                  # rozgrywka lew
    GAME_OVER = "game_over"        # gra zakończona


@dataclass(frozen=True)
class Rules:
    """Parametry zasad. Wartości domyślne = klasyczny Tysiąc dla 2 osób."""

    target_score: int = 1000
    min_bid: int = 100
    bid_step: int = 10
    max_bid_without_meld: int = 120
    #: Czy wolno meldować przy wyjściu do PIERWSZEJ lewy rozdania.
    meld_on_first_trick: bool = False
    #: Obowiązek dania atutu, gdy brak koloru wyjścia.
    must_trump: bool = True
    #: Obowiązek przebicia w kolorze (wariant – domyślnie wyłączony).
    must_beat_in_suit: bool = False
    #: Zasada beczki (800).
    barrel_enabled: bool = True
    barrel_threshold: int = 800
    barrel_max_rounds: int = 3
    barrel_penalty: int = 120


class IllegalMoveError(ValueError):
    """Ruch niezgodny z zasadami lub wykonany przez gracza bez prawa ruchu."""


# ---------------------------------------------------------------------------
# Karta
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Card:
    """Pojedyncza karta. Kod tekstowy: '<ranga>_<kolor>', np. '10_hearts'."""

    rank: str
    suit: str

    @property
    def code(self) -> str:
        return f"{self.rank}_{self.suit}"

    @property
    def points(self) -> int:
        return CARD_POINTS[self.rank]

    @property
    def strength(self) -> int:
        return RANKS.index(self.rank)

    @classmethod
    def from_code(cls, code: str) -> "Card":
        try:
            rank, suit = code.split("_", 1)
        except (ValueError, AttributeError):
            raise IllegalMoveError(f"Niepoprawny kod karty: {code!r}") from None
        if rank not in RANKS or suit not in SUITS:
            raise IllegalMoveError(f"Niepoprawny kod karty: {code!r}")
        return cls(rank, suit)

    def __str__(self) -> str:  # czytelność w logach
        return self.code


def new_deck() -> list[Card]:
    """Pełna talia 24 kart."""
    return [Card(r, s) for s in SUITS for r in RANKS]


def sort_hand(cards: list[Card]) -> list[Card]:
    """Sortowanie ręki: kolorami, w kolorze od najsilniejszej."""
    return sorted(cards, key=lambda c: (SUITS.index(c.suit), -c.strength))


def melds_in(cards: list[Card]) -> list[str]:
    """Kolory, w których na ręce jest para K+Q (potencjalne meldunki)."""
    present = {(c.rank, c.suit) for c in cards}
    return [s for s in SUITS if ("K", s) in present and ("Q", s) in present]


def round_to_ten(points: int) -> int:
    """Zaokrąglenie do pełnych dziesiątek (5 w górę): 64->60, 65->70."""
    return (points + 5) // 10 * 10


def trick_winner(lead: Card, follow: Card, trump: str | None) -> int:
    """
    Zwraca indeks zwycięskiej karty w lewie dwuosobowej (0 = wyjście, 1 = dołożenie).
    Wygrywa najwyższy atut, a gdy nie ma atutów – najwyższa karta koloru wyjścia.
    """
    if follow.suit == lead.suit:
        return 1 if follow.strength > lead.strength else 0
    if trump is not None and follow.suit == trump:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Stan pojedynczego rozdania
# ---------------------------------------------------------------------------

@dataclass
class RoundState:
    """Wszystko, co dotyczy jednego rozdania (czyszczone przy nowym rozdaniu)."""

    number: int
    dealer: str
    hands: dict[str, list[Card]]
    musiki: list[list[Card]]

    # licytacja
    bid: int | None = None
    high_bidder: str | None = None
    bid_turn: str | None = None
    bidder: str | None = None          # zwycięzca licytacji (rozgrywający)

    # musik / kontrakt
    chosen_musik: int | None = None
    discards: list[Card] = field(default_factory=list)
    contract: int | None = None

    # rozgrywka
    trump: str | None = None
    leader: str | None = None
    trick: list[tuple[str, Card]] = field(default_factory=list)
    tricks_done: int = 0
    taken: dict[str, list[Card]] = field(default_factory=dict)
    melds: dict[str, list[str]] = field(default_factory=dict)
    played: list[Card] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Silnik gry
# ---------------------------------------------------------------------------

class TysiacGame:
    """
    Pełna partia Tysiąca dla dwóch graczy (wiele rozdań, aż do 1000 pkt).

    Przykład:
        game = TysiacGame(("alice", "bob"), seed=42)
        while not game.is_over:
            p = game.current_player
            state = game.get_state(p)                 # -> wysyłamy do LLM
            game.apply_move(p, {"action": "pass"})    # <- ruch od LLM
    """

    def __init__(
        self,
        players: tuple[str, str] = ("player_1", "player_2"),
        rules: Rules | None = None,
        seed: int | None = None,
        game_id: str | None = None,
    ) -> None:
        if len(players) != 2 or players[0] == players[1]:
            raise ValueError("Tysiąc w tej wersji wymaga dokładnie 2 różnych graczy.")
        self.players: tuple[str, str] = tuple(players)  # type: ignore[assignment]
        self.rules = rules or Rules()
        self.game_id = game_id or uuid.uuid4().hex[:12]
        self._rng = random.Random(seed)

        self.scores: dict[str, int] = {p: 0 for p in self.players}
        self.barrel_rounds: dict[str, int] = {p: 0 for p in self.players}
        self.winner: str | None = None
        self.phase: str = Phase.BIDDING

        # Pełny dziennik zdarzeń całej partii. Każde zdarzenie ma część jawną
        # i (opcjonalnie) część prywatną widoczną tylko dla jednego gracza.
        self._log: list[dict[str, Any]] = []

        # Pierwszym rozdającym jest losowy gracz; potem zmiana co rozdanie.
        self.round: RoundState = self._deal(number=1, dealer=self._rng.choice(self.players))

    # ------------------------------------------------------------------ util

    def opponent(self, player: str) -> str:
        return self.players[1] if player == self.players[0] else self.players[0]

    @property
    def is_over(self) -> bool:
        return self.phase == Phase.GAME_OVER

    @property
    def current_player(self) -> str | None:
        """Gracz, który ma teraz wykonać ruch (None po zakończeniu gry)."""
        r = self.round
        if self.phase == Phase.BIDDING:
            return r.bid_turn
        if self.phase in (Phase.CHOOSE_MUSIK, Phase.DISCARD, Phase.DECLARE):
            return r.bidder
        if self.phase == Phase.PLAY:
            return r.leader if not r.trick else self.opponent(r.leader)  # type: ignore[arg-type]
        return None

    def _record(self, player: str | None, action: str, private: dict[str, Any] | None = None,
                **public: Any) -> None:
        """Dopisuje zdarzenie do dziennika. `private` widzi tylko `player`."""
        event = {
            "seq": len(self._log) + 1,
            "round": self.round.number if hasattr(self, "round") else 1,
            "phase": self.phase,
            "player": player,
            "action": action,
            **{k: v for k, v in public.items() if v is not None},
        }
        self._log.append({"public": event, "private": private or {}, "owner": player})

    def max_bid(self, player: str) -> int:
        """Najwyższa stawka dopuszczalna dla gracza: 120 + meldunki na ręce."""
        hand = self.round.hands[player]
        return self.rules.max_bid_without_meld + sum(MELD_POINTS[s] for s in melds_in(hand))

    # --------------------------------------------------------------- deal

    def _deal(self, number: int, dealer: str) -> RoundState:
        """Tasowanie i rozdanie: 10 + 10 kart oraz dwa musiki po 2 karty."""
        deck = new_deck()
        self._rng.shuffle(deck)
        first, second = self.opponent(dealer), dealer
        hands = {first: sort_hand(deck[0:10]), second: sort_hand(deck[10:20])}
        musiki = [deck[20:22], deck[22:24]]

        r = RoundState(number=number, dealer=dealer, hands=hands, musiki=musiki)
        r.taken = {p: [] for p in self.players}
        r.melds = {p: [] for p in self.players}
        r.bid_turn = first  # licytację zaczyna gracz nie-rozdający
        self.round = r
        self.phase = Phase.BIDDING
        self._record(None, "deal", dealer=dealer, first_bidder=first)
        return r

    # --------------------------------------------------------- legal moves

    def legal_moves(self, player: str) -> list[dict[str, Any]]:
        """
        Lista legalnych ruchów w formacie zrozumiałym dla backendu/LLM:
            {"action": "play_card", "card": "A_spades"}
            {"action": "bid", "min": 110, "max": 180, "step": 10}
            {"action": "choose_musik", "value": 0}
        """
        if self.is_over or player != self.current_player:
            return []
        r, rules = self.round, self.rules

        if self.phase == Phase.BIDDING:
            cap = self.max_bid(player)
            if r.bid is None:  # pierwsza, obowiązkowa licytacja (min. 100)
                return [{"action": "bid", "min": rules.min_bid, "max": cap, "step": rules.bid_step}]
            moves: list[dict[str, Any]] = [{"action": "pass"}]
            if cap >= r.bid + rules.bid_step:
                moves.insert(0, {"action": "bid", "min": r.bid + rules.bid_step, "max": cap,
                                 "step": rules.bid_step})
            return moves

        if self.phase == Phase.CHOOSE_MUSIK:
            return [{"action": "choose_musik", "value": i} for i in range(len(r.musiki))]

        if self.phase == Phase.DISCARD:
            return [{"action": "discard", "card": c.code} for c in r.hands[player]]

        if self.phase == Phase.DECLARE:
            cap = max(r.bid or 0, self.max_bid(player))
            return [{"action": "declare", "min": r.bid, "max": cap, "step": rules.bid_step}]

        if self.phase == Phase.PLAY:
            return self._legal_plays(player)
        return []

    def _legal_plays(self, player: str) -> list[dict[str, Any]]:
        r, rules = self.round, self.rules
        hand = r.hands[player]

        # --- wyjście do lewy: dowolna karta + ewentualne meldunki -------------
        if not r.trick:
            moves = [{"action": "play_card", "card": c.code} for c in hand]
            if r.tricks_done > 0 or rules.meld_on_first_trick:
                for suit in melds_in(hand):
                    for rank in ("K", "Q"):
                        moves.append({"action": "meld", "card": Card(rank, suit).code})
            return moves

        # --- dokładanie do lewy ----------------------------------------------
        lead = r.trick[0][1]
        same_suit = [c for c in hand if c.suit == lead.suit]
        if same_suit:
            candidates = same_suit
            if rules.must_beat_in_suit:
                higher = [c for c in same_suit if c.strength > lead.strength]
                candidates = higher or same_suit
        else:
            trumps = [c for c in hand if r.trump is not None and c.suit == r.trump]
            candidates = trumps if (rules.must_trump and trumps) else hand
        return [{"action": "play_card", "card": c.code} for c in candidates]

    @staticmethod
    def move_matches(move: dict[str, Any], legal: dict[str, Any]) -> bool:
        """Czy konkretny ruch pasuje do opisu legalnego ruchu (ta sama logika co w backendzie)."""
        if move.get("action") != legal.get("action"):
            return False
        for key in ("card", "value"):
            if key in legal and move.get(key) != legal[key]:
                return False
        if "min" in legal:
            v = move.get("value")
            if not isinstance(v, int) or isinstance(v, bool):
                return False
            if not legal["min"] <= v <= legal["max"]:
                return False
            if (v - legal["min"]) % legal.get("step", 1) != 0:
                return False
        return True

    # ---------------------------------------------------------- apply move

    def apply_move(self, player: str, move: dict[str, Any]) -> None:
        """
        Wykonuje ruch gracza. Rzuca IllegalMoveError, gdy ruch jest niedozwolony
        (stan gry pozostaje wtedy nienaruszony).
        """
        if self.is_over:
            raise IllegalMoveError("Gra jest zakończona.")
        if player != self.current_player:
            raise IllegalMoveError(f"Teraz ruch ma {self.current_player}, nie {player}.")
        if not any(self.move_matches(move, lm) for lm in self.legal_moves(player)):
            raise IllegalMoveError(f"Niedozwolony ruch w fazie '{self.phase}': {move}")

        action = move["action"]
        handler = {
            "bid": self._do_bid,
            "pass": self._do_pass,
            "choose_musik": self._do_choose_musik,
            "discard": self._do_discard,
            "declare": self._do_declare,
            "play_card": self._do_play,
            "meld": self._do_play,
        }[action]
        handler(player, move)

    # --- licytacja -----------------------------------------------------------

    def _do_bid(self, player: str, move: dict[str, Any]) -> None:
        r = self.round
        r.bid, r.high_bidder = move["value"], player
        self._record(player, "bid", value=r.bid)
        r.bid_turn = self.opponent(player)

    def _do_pass(self, player: str, move: dict[str, Any]) -> None:
        r = self.round
        self._record(player, "pass")
        # W grze dwuosobowej pas jednego gracza kończy licytację.
        r.bidder = r.high_bidder
        r.bid_turn = None
        self._record(r.bidder, "bidding_won", value=r.bid)
        self.phase = Phase.CHOOSE_MUSIK

    # --- musik, odkładanie, deklaracja ---------------------------------------

    def _do_choose_musik(self, player: str, move: dict[str, Any]) -> None:
        r = self.round
        r.chosen_musik = move["value"]
        cards = r.musiki[r.chosen_musik]
        # Wybrany musik jest odkrywany – widzą go obaj gracze.
        self._record(player, "choose_musik", value=r.chosen_musik, cards=[c.code for c in cards])
        r.hands[player] = sort_hand(r.hands[player] + cards)
        self.phase = Phase.DISCARD

    def _do_discard(self, player: str, move: dict[str, Any]) -> None:
        r = self.round
        card = Card.from_code(move["card"])
        r.hands[player].remove(card)
        r.discards.append(card)
        # Przeciwnik wie TYLKO, że karta została odłożona – nie wie jaka.
        self._record(player, "discard", private={"card": card.code})
        if len(r.hands[player]) == HAND_SIZE:
            self.phase = Phase.DECLARE

    def _do_declare(self, player: str, move: dict[str, Any]) -> None:
        r = self.round
        r.contract = move["value"]
        self._record(player, "declare", value=r.contract)
        r.leader = player  # rozgrywający wychodzi do pierwszej lewy
        self.phase = Phase.PLAY

    # --- rozgrywka -------------------------------------------------------------

    def _do_play(self, player: str, move: dict[str, Any]) -> None:
        r = self.round
        card = Card.from_code(move["card"])
        r.hands[player].remove(card)
        r.trick.append((player, card))
        r.played.append(card)

        if move["action"] == "meld":
            r.melds[player].append(card.suit)
            r.trump = card.suit
            self._record(player, "meld", card=card.code, suit=card.suit,
                         points=MELD_POINTS[card.suit], new_trump=card.suit)
        else:
            self._record(player, "play_card", card=card.code)

        if len(r.trick) == 2:
            self._finish_trick()

    def _finish_trick(self) -> None:
        r = self.round
        (p_lead, c_lead), (p_follow, c_follow) = r.trick
        winner = (p_lead, p_follow)[trick_winner(c_lead, c_follow, r.trump)]
        r.taken[winner].extend([c_lead, c_follow])
        r.tricks_done += 1
        self._record(winner, "trick_won", cards=[c_lead.code, c_follow.code],
                     points=c_lead.points + c_follow.points, trick_no=r.tricks_done)
        r.trick = []
        r.leader = winner
        if r.tricks_done == TRICKS_PER_ROUND:
            self._score_round()

    # --- punktacja -------------------------------------------------------------

    def _card_points(self, player: str) -> int:
        r = self.round
        pts = sum(c.points for c in r.taken[player])
        if player == r.bidder:
            pts += sum(c.points for c in r.discards)  # odłożone karty liczą się rozgrywającemu
        return pts

    def _score_round(self) -> None:
        r, rules = self.round, self.rules
        bidder, defender = r.bidder, self.opponent(r.bidder)  # type: ignore[arg-type]
        assert bidder is not None and r.contract is not None

        raw = {p: self._card_points(p) + sum(MELD_POINTS[s] for s in r.melds[p]) for p in self.players}
        made = raw[bidder] >= r.contract
        delta = {bidder: r.contract if made else -r.contract, defender: round_to_ten(raw[defender])}

        before = dict(self.scores)
        on_barrel_before = {p: self._on_barrel(p) for p in self.players}

        # --- zasada beczki ---------------------------------------------------
        if rules.barrel_enabled:
            # Obrońca na beczce nic nie dopisuje; obrońca poniżej beczki nie
            # może jej przekroczyć (zatrzymuje się na 800).
            if on_barrel_before[defender]:
                delta[defender] = 0
            else:
                delta[defender] = min(delta[defender], max(0, rules.barrel_threshold - before[defender]))

        for p in self.players:
            self.scores[p] += delta[p]

        penalties: dict[str, int] = {}
        if rules.barrel_enabled:
            for p in self.players:
                if self.scores[p] >= rules.target_score or not self._on_barrel(p):
                    self.barrel_rounds[p] = 0
                    continue
                if on_barrel_before[p]:
                    self.barrel_rounds[p] += 1
                    if self.barrel_rounds[p] >= rules.barrel_max_rounds:
                        # "Spadnięcie z beczki"
                        self.scores[p] -= rules.barrel_penalty
                        penalties[p] = rules.barrel_penalty
                        self.barrel_rounds[p] = 0
                else:
                    self.barrel_rounds[p] = 0  # dopiero wszedł na beczkę

        self._record(
            None, "round_end",
            bidder=bidder, contract=r.contract, contract_made=made,
            round_points=raw, score_change=delta, barrel_penalties=penalties or None,
            scores=dict(self.scores),
            # Po rozdaniu odkrywa się odłożone karty i niewybrany musik.
            revealed_discards=[c.code for c in r.discards],
            unused_musik=[c.code for c in r.musiki[1 - (r.chosen_musik or 0)]],
        )

        # --- koniec gry? ------------------------------------------------------
        reached = [p for p in self.players if self.scores[p] >= rules.target_score]
        if reached:
            # Przy (teoretycznym) remisie wygrywa rozgrywający.
            self.winner = max(reached, key=lambda p: (self.scores[p], p == bidder))
            self.phase = Phase.GAME_OVER
            self._record(self.winner, "game_won", scores=dict(self.scores))
            return

        self._deal(number=r.number + 1, dealer=self.opponent(r.dealer))

    def _on_barrel(self, player: str) -> bool:
        s = self.scores[player]
        return self.rules.barrel_threshold <= s < self.rules.target_score

    # ------------------------------------------------------------ state view

    def history_for(self, player: str) -> list[dict[str, Any]]:
        """Pełna historia jawnych zdarzeń (wszystkie rozdania) z perspektywy gracza."""
        out = []
        for e in self._log:
            ev = dict(e["public"])
            if e["owner"] == player and e["private"]:
                ev.update(e["private"])
            out.append(ev)
        return out

    def get_state(self, player: str) -> dict[str, Any]:
        """
        Kompletny, BEZSTANOWY opis sytuacji dla jednego gracza – gotowy do
        `json.dumps`. Zawiera wyłącznie informacje, które ten gracz zna przy
        prawdziwym stole: własną rękę, jawne karty, historię wszystkich ruchów
        (także z poprzednich rozdań) i listę legalnych ruchów.
        """
        if player not in self.players:
            raise ValueError(f"Nieznany gracz: {player}")
        r = self.round
        opp = self.opponent(player)
        role = None if r.bidder is None else ("bidder" if r.bidder == player else "defender")

        # Punkty z wziętych lew są jawne (gracz może je policzyć). Odłożone
        # karty rozgrywającego zna tylko on sam.
        trick_pts = {p: sum(c.points for c in r.taken[p]) for p in self.players}

        state: dict[str, Any] = {
            "game_id": self.game_id,
            "player_id": player,
            "opponent_id": opp,
            "phase": self.phase,
            "round": r.number,
            "dealer": r.dealer,
            "your_role": role,
            "hand": [c.code for c in r.hands[player]],
            "opponent_hand_size": len(r.hands[opp]),
            "table": [{"player": p, "card": c.code} for p, c in r.trick],
            "trick_leader": r.leader if self.phase == Phase.PLAY else None,
            "trump": r.trump,
            "current_bid": r.bid,
            "bid_winner": r.bidder or r.high_bidder,
            "your_max_bid": self.max_bid(player),
            "contract": r.contract,
            "revealed_musik": ([c.code for c in r.musiki[r.chosen_musik]]
                               if r.chosen_musik is not None else None),
            "tricks_played": r.tricks_done,
            "cards_played_this_round": [c.code for c in r.played],
            "melds_declared": {p: list(r.melds[p]) for p in self.players},
            "trick_points": trick_pts,
            "round_points": {
                p: trick_pts[p] + sum(MELD_POINTS[s] for s in r.melds[p]) for p in self.players
            },
            "scores": dict(self.scores),
            "barrel_rounds": dict(self.barrel_rounds) if self.rules.barrel_enabled else None,
            "winner": self.winner,
            "history": self.history_for(player),
            "legal_moves": self.legal_moves(player),
            "rules_summary": self.rules_summary(),
        }
        if role == "bidder" and r.discards:
            state["your_discards"] = [c.code for c in r.discards]
            state["your_discard_points"] = sum(c.points for c in r.discards)
        return state

    def rules_summary(self) -> dict[str, Any]:
        """Zwięzła ściągawka zasad dołączana do stanu (model nie pamięta niczego między krokami)."""
        rules = self.rules
        return {
            "players": 2,
            "card_ranking_high_to_low": list(reversed(RANKS)),
            "card_points": CARD_POINTS,
            "meld_points": MELD_POINTS,
            "suits_pl": SUITS_PL,
            "target_score": rules.target_score,
            "bidding": f"start {rules.min_bid} (obowiązkowo), krok {rules.bid_step}, "
                       f"powyżej {rules.max_bid_without_meld} tylko z meldunkiem na ręce",
            "play": "trzeba dołożyć do koloru; bez koloru trzeba dać atut"
                    + (" (jeśli jest)" if rules.must_trump else " (nieobowiązkowo)")
                    + ("; w kolorze trzeba przebić" if rules.must_beat_in_suit else ""),
            "meld": "tylko przy wyjściu do lewy, zagrywając K lub Q z pary; kolor meldunku staje się atutem"
                    + ("" if rules.meld_on_first_trick else "; nie w pierwszej lewie"),
            "scoring": "rozgrywający: +kontrakt gdy ugrał, -kontrakt gdy nie; "
                       "obrońca: swoje punkty zaokrąglone do 10",
            "barrel": (f"od {rules.barrel_threshold} pkt punkty dopisuje tylko wygrany kontrakt; "
                       f"{rules.barrel_max_rounds} rozdania na beczce bez wygranej = -{rules.barrel_penalty}")
                      if rules.barrel_enabled else None,
        }
