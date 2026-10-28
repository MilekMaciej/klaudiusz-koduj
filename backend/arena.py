"""
arena.py
========

Mecze rozgrywane w przeglądarce: sesje gier, izolowane kanały botów, gracz-człowiek.

Moduł nie zależy od FastAPI – `backend/main.py` tylko mapuje trasy HTTP na
metody `ArenaService` (dzięki temu logikę da się testować bez serwera).

Izolacja botów (anti-cheat)
---------------------------
Każdy gracz-bot ma własny `BotChannel` – osobny, niepołączony kontekst:
  * zna WYŁĄCZNIE swoją strategię (system prompt) i swój model,
  * w każdym kroku dostaje WŁASNĄ kopię obserwacji zbudowanej dla niego przez
    silnik (`TysiacGame.get_state(player)`), sprawdzonej przez niezależny audyt
    (`engine.visibility.audit_observation`) – przy wykryciu wycieku zapytanie
    nie jest wysyłane (fail closed),
  * każde wywołanie modelu to nowa, jednorazowa rozmowa (system + user) –
    nie ma wspólnej historii czatu, cache ani pamięci między Botem A i Botem B,
  * uzasadnienia (reasoning) bota nigdy nie trafiają do historii gry, więc
    przeciwnik ich nie widzi.

Analiza ruchów (solver C++)
---------------------------
Każdy ruch w rozgrywce (bota i człowieka) jest w tle oceniany solverem double-dummy
(`backend/analysis.py`): Q zagranego ruchu, najlepszy ruch, max Q i strata (LOSS).
Solver dostaje wyłącznie wiedzę gracza z chwili decyzji. Wyniki nigdy nie trafiają
do botów. Człowiek nie widzi analizy ruchów bota z trwającego rozdania (lista
ocenionych alternatyw zdradzałaby karty bota) – chyba że gra z jawnymi kartami.

Gracz-człowiek
--------------
Mecz może mieć najwyżej jednego człowieka. Przy tworzeniu meczu dostaje on
tajny token; tylko z nim może wykonać ruch i zobaczyć SWOJĄ rękę. Widok
sędziego (obie ręce, obserwacje botów) jest niedostępny, dopóki w meczu
gra człowiek i partia trwa – inaczej można by podejrzeć karty bota.

Wyjątek: w sandboxie człowiek może świadomie wybrać tryb "jawny" (otwarte karty) –
wtedy jego widok zawiera wszystko, co widzi sędzia (do analizy i testów strategii).
Boty nadal dostają wyłącznie własne obserwacje.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import re
import secrets
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from backend import ai_service
from backend.ai_service import AIServiceError, LLMUnavailableError
from backend.analysis import SOLVER
from backend.llm_config import LLMConfigError
from engine.client import HeuristicAgent
from engine.game import MELD_POINTS, IllegalMoveError, Phase, TysiacGame
from engine.scenario import ScenarioError, build_scenario
from engine.solver_bridge import build_q_input
from engine.visibility import InformationLeakError, audit_observation

logger = logging.getLogger("backend.arena")

PLAYERS = ("A", "B")
STRATEGY_DIR = Path(__file__).resolve().parent.parent / "strategies"
_PRESET_NAME = re.compile(r"^[a-z0-9_-]{1,40}$")
MAX_MATCHES = 50
MATCH_TTL_S = 6 * 3600


class ArenaError(Exception):
    """Błąd zgłaszany do klienta HTTP (status + czytelny komunikat)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------------------
# Presety strategii
# ---------------------------------------------------------------------------

def list_presets() -> list[dict[str, str]]:
    """Strategie z katalogu strategies/*.txt (nazwa, tytuł = pierwsza linia, treść)."""
    out = []
    for path in sorted(STRATEGY_DIR.glob("*.txt")):
        if _PRESET_NAME.match(path.stem):
            text = path.read_text(encoding="utf-8").strip()
            out.append({"name": path.stem, "title": text.splitlines()[0] if text else path.stem, "text": text})
    return out


def load_preset(name: str) -> str:
    """Treść presetu – tylko nazwy z katalogu strategies/ (bez ścieżek z zewnątrz)."""
    if not _PRESET_NAME.match(name or ""):
        raise ArenaError(400, f"Niepoprawna nazwa presetu: {name!r}")
    path = STRATEGY_DIR / f"{name}.txt"
    if not path.is_file():
        raise ArenaError(404, f"Nie ma presetu '{name}'. Dostępne: {', '.join(p['name'] for p in list_presets())}")
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Żądania (walidacja Pydantic)
# ---------------------------------------------------------------------------

class SeatConfig(BaseModel):
    """Kto gra na danym miejscu i jaką strategią."""

    kind: Literal["llm", "human", "heuristic"] = "llm"
    strategy_source: Literal["preset", "custom"] = "preset"
    preset: str | None = "zbalansowana"
    strategy: str = Field(default="", max_length=8000)
    model: str | None = None
    label: str | None = Field(default=None, max_length=40)

    @field_validator("model", "preset", "label")
    @classmethod
    def _empty_to_none(cls, v: str | None) -> str | None:
        return v.strip() or None if isinstance(v, str) else v


class CreateMatchRequest(BaseModel):
    mode: Literal["normal", "sandbox"] = "normal"
    seats: dict[str, SeatConfig]
    seed: int | None = None
    max_rounds: int | None = Field(default=10, ge=1, le=200)
    sandbox: dict[str, Any] | None = None
    #: Sandbox z człowiekiem: True = karty jawne (człowiek widzi karty przeciwnika).
    open_cards: bool = False
    #: Analiza ruchów solverem C++ (Q / strata) w tle.
    analysis: bool = True

    @field_validator("seats")
    @classmethod
    def _two_seats(cls, v: dict[str, SeatConfig]) -> dict[str, SeatConfig]:
        if set(v) != set(PLAYERS):
            raise ValueError(f"Wymagane miejsca: {', '.join(PLAYERS)}")
        return v


# ---------------------------------------------------------------------------
# Kanały decyzyjne – jeden na gracza, bez współdzielonego stanu
# ---------------------------------------------------------------------------

@dataclass
class BotChannel:
    """
    Izolowany kontekst JEDNEGO bota LLM. Nie ma referencji do drugiego bota,
    jego strategii ani do obiektu gry – dostaje tylko gotową obserwację.
    """

    player: str
    strategy: str
    model: str | None
    calls: list[dict[str, Any]] = field(default_factory=list)
    last_observation: dict[str, Any] | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def context_id(self) -> str:
        """Odcisk kontekstu (strategia + model) – do weryfikacji izolacji w logach i testach."""
        return hashlib.sha256(f"{self.player}\0{self.model}\0{self.strategy}".encode()).hexdigest()[:12]

    async def decide(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        async with self._lock:
            own_copy = copy.deepcopy(observation)          # bot nie współdzieli obiektów z nikim
            self.last_observation = copy.deepcopy(observation)
            t0 = time.perf_counter()
            move, meta = await ai_service.get_move(own_copy, self.strategy, model=self.model)
            meta = {**meta, "ms": round((time.perf_counter() - t0) * 1000), "context_id": self.context_id}
            self.calls.append({"round": observation["round"], "phase": observation["phase"],
                               "move": move.model_dump(exclude_none=True), "meta": meta})
            return move.model_dump(exclude_none=True), meta


@dataclass
class HeuristicChannel:
    """Bot regułowy – darmowy przeciwnik do testów."""

    player: str
    strategy: str = "(bot regułowy – bez promptu)"
    model: str | None = "heuristic"
    last_observation: dict[str, Any] | None = None
    context_id: str = "heuristic"

    async def decide(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.last_observation = copy.deepcopy(observation)
        move = HeuristicAgent(self.player).choose(copy.deepcopy(observation))
        return move, {"provider": "heuristic", "model": "heuristic", "attempts": 1, "usage": {}, "ms": 0,
                      "context_id": self.context_id}


# ---------------------------------------------------------------------------
# Mecz
# ---------------------------------------------------------------------------

SUIT_PL = {"hearts": "♥", "diamonds": "♦", "clubs": "♣", "spades": "♠"}


def _card_pl(code: str | None) -> str:
    if not code:
        return "-"
    rank, suit = code.split("_", 1)
    return f"{rank}{SUIT_PL.get(suit, suit)}"


def describe_move(move: dict[str, Any]) -> str:
    action = move.get("action")
    return {
        "bid": lambda: f"licytuje {move.get('value')}",
        "pass": lambda: "pasuje",
        "choose_musik": lambda: f"wybiera musik nr {int(move.get('value', 0)) + 1}",
        "discard": lambda: "odkłada kartę",
        "declare": lambda: f"deklaruje kontrakt {move.get('value')}",
        "play_card": lambda: f"zagrywa {_card_pl(move.get('card'))}",
        "meld": lambda: f"MELDUJE ({MELD_POINTS[move['card'].split('_')[1]]}) i zagrywa {_card_pl(move.get('card'))}",
    }.get(action, lambda: str(move))()


@dataclass
class Match:
    id: str
    mode: str
    game: TysiacGame
    seats: dict[str, SeatConfig]
    channels: dict[str, BotChannel | HeuristicChannel | None]
    max_rounds: int | None
    human: str | None = None
    human_token: str | None = None
    open_cards: bool = False
    analysis: bool = False
    #: wejścia solvera dla wpisów dziennika (seq -> (wejście, wejście zapasowe)) – do ponowienia analizy
    solver_inputs: dict[int, tuple[str, str]] = field(default_factory=dict, repr=False)
    warnings: list[str] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    last_error: dict[str, Any] | None = None
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # --- stan -------------------------------------------------------------------

    def _refresh_status(self) -> None:
        g = self.game
        if g.is_over:
            self.status = "finished"
        elif self.max_rounds and g.round.number > self.max_rounds:
            self.status = "round_limit"
        elif self.status == "error":
            pass
        elif self.human and g.current_player == self.human:
            self.status = "waiting_human"
        else:
            self.status = "running"

    @property
    def in_progress(self) -> bool:
        return self.status not in ("finished", "round_limit")

    # --- ruchy ---------------------------------------------------------------------

    async def step(self) -> None:
        """Wykonuje JEDEN ruch bota, który ma teraz ruch (człowiek: nic nie robi)."""
        async with self.lock:
            self.touched = time.time()
            self.status = "running" if self.status == "error" else self.status
            self._refresh_status()
            if not self.in_progress or self.status == "waiting_human":
                return
            g = self.game
            player = g.current_player
            channel = self.channels[player]
            assert channel is not None

            # 1) Obserwacja WYŁĄCZNIE dla tego gracza + niezależny audyt (fail closed)
            observation = g.get_state(player)
            try:
                audit_observation(g, player, observation)
            except InformationLeakError as exc:
                logger.critical("ANTI-CHEAT: zablokowano zapytanie bota %s: %s", player, exc)
                self._set_error("anti_cheat", str(exc), player, observation)
                return

            # 2) Decyzja w izolowanym kanale tego bota
            try:
                move, meta = await channel.decide(observation)
            except AIServiceError as exc:
                self._set_error("model_move", str(exc), player, observation)
                return
            except LLMConfigError as exc:
                self._set_error("llm_config", str(exc), player, observation)
                return
            except LLMUnavailableError as exc:
                self._set_error("llm_unavailable", str(exc), player, observation)
                return

            reasoning = str(move.pop("reasoning", "") or "")
            solver_inputs = self._solver_inputs(player, move)   # PRZED ruchem: stan z chwili decyzji
            try:
                g.apply_move(player, move)
            except IllegalMoveError as exc:
                self._set_error("illegal_move", f"Silnik odrzucił ruch {move}: {exc}", player, observation)
                return
            self._record(player, observation, move, reasoning, meta, solver_inputs)
            self.last_error = None
            self._refresh_status()

    def human_move(self, token: str | None, move: dict[str, Any]) -> None:
        if not self.human:
            raise ArenaError(400, "W tym meczu nie ma gracza-człowieka.")
        if not token or not secrets.compare_digest(token, self.human_token or ""):
            raise ArenaError(403, "Niepoprawny token gracza.")
        g = self.game
        if g.current_player != self.human:
            raise ArenaError(409, f"Teraz ruch ma {g.current_player}, nie Ty.")
        move = {k: v for k, v in move.items() if k in ("action", "card", "value")}
        observation = g.get_state(self.human)
        legal = any(g.move_matches(move, lm) for lm in g.legal_moves(self.human))
        solver_inputs = self._solver_inputs(self.human, move) if legal else None
        try:
            g.apply_move(self.human, move)
        except IllegalMoveError as exc:
            raise ArenaError(400, f"Niedozwolony ruch: {exc}") from None
        self.touched = time.time()
        self._record(self.human, observation, move, "", {"provider": "human", "model": "człowiek"}, solver_inputs)
        self._refresh_status()

    def set_open_cards(self, token: str | None, value: bool) -> None:
        """Przełącznik jawnych kart w trakcie gry (tylko sandbox z człowiekiem)."""
        if self.mode != "sandbox" or not self.human:
            raise ArenaError(400, "Jawne karty są dostępne tylko w sandboxie z graczem-człowiekiem.")
        if not token or not secrets.compare_digest(token, self.human_token or ""):
            raise ArenaError(403, "Niepoprawny token gracza.")
        self.open_cards = bool(value)

    # --- analiza solverem ------------------------------------------------------------

    def _solver_inputs(self, player: str, move: dict[str, Any]) -> tuple[str, str] | dict[str, Any] | None:
        """
        Wejście solvera dla ruchu w rozgrywce (z wiedzy gracza w chwili decyzji).
        None = ruch nie podlega analizie; dict = opis błędu (ruch i tak się wykona).
        """
        if not self.analysis or self.game.phase != Phase.PLAY or move.get("action") not in ("play_card", "meld"):
            return None
        seed = (zlib.crc32(f"{self.id}:{len(self.log)}".encode()) & 0x7FFFFFFF) or 1   # powtarzalne ziarno
        try:
            return (build_q_input(self.game, player, move, samples=SOLVER.samples, seed=seed),
                    build_q_input(self.game, player, move, samples=SOLVER.samples, seed=seed, use_voids=False))
        except Exception as exc:  # noqa: BLE001 – przygotowanie analizy nigdy nie blokuje ruchu
            logger.exception("Nie da się przygotować wejścia solvera dla ruchu %s gracza %s", move, player)
            return {"status": "error", "kind": "input", "title": "Niepoprawne dane dla solvera",
                    "message": f"nie udało się zbudować obserwacji dla solvera: {type(exc).__name__}: {exc}",
                    "retryable": False}

    def retry_analysis(self, seq: int) -> None:
        """Ponowna analiza wpisu dziennika (np. po chwilowej awarii solvera)."""
        entry = next((e for e in self.log if e["seq"] == seq), None)
        if entry is None:
            raise ArenaError(404, f"Brak ruchu nr {seq}.")
        inputs = self.solver_inputs.get(seq)
        if not inputs:
            raise ArenaError(400, "Tego ruchu nie da się przeanalizować (to nie ruch w rozgrywce albo brak danych).")
        if (entry.get("analysis") or {}).get("status") == "pending":
            return
        if not SOLVER.available:
            raise ArenaError(503, f"Solver niedostępny: {SOLVER.reason}")
        entry["analysis"] = {"status": "pending"}
        SOLVER.submit(inputs[0], inputs[1], lambda result, entry=entry: entry.__setitem__("analysis", result))

    def _record(self, player: str, observation: dict[str, Any], move: dict[str, Any], reasoning: str,
                meta: dict[str, Any], solver_inputs: tuple[str, str] | dict[str, Any] | None = None) -> None:
        usage = meta.get("usage") or {}
        self.log.append({
            "seq": len(self.log) + 1,
            "round": observation["round"],
            "phase": observation["phase"],
            "player": player,
            "kind": self.seats[player].kind,
            "move": move,
            "text": describe_move(move),
            "reasoning": reasoning,
            "model": meta.get("model"),
            "provider": meta.get("provider"),
            "attempts": meta.get("attempts"),
            "fallback_used": meta.get("fallback_used", False),
            "tokens": (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0),
            "ms": meta.get("ms"),
            "context_id": meta.get("context_id"),
            "analysis": None,
        })
        entry = self.log[-1]
        if isinstance(solver_inputs, dict):           # nie udało się przygotować wejścia – pokaż powód
            entry["analysis"] = solver_inputs
        elif solver_inputs and SOLVER.available:
            self.solver_inputs[entry["seq"]] = solver_inputs
            entry["analysis"] = {"status": "pending"}

            def done(result: dict[str, Any], entry: dict[str, Any] = entry) -> None:
                entry["analysis"] = result          # przypisanie słownika – atomowe w CPython

            SOLVER.submit(solver_inputs[0], solver_inputs[1], done)
        elif solver_inputs:
            entry["analysis"] = {"status": "unavailable", "kind": "unavailable",
                                 "message": SOLVER.reason, "retryable": False}

    def _set_error(self, kind: str, message: str, player: str, observation: dict[str, Any]) -> None:
        self.status = "error"
        self.last_error = {"kind": kind, "message": message, "player": player,
                           "round": observation["round"], "phase": observation["phase"]}
        logger.warning("Mecz %s: błąd (%s) przy ruchu %s: %s", self.id, kind, player, message)

    # --- widoki ----------------------------------------------------------------------

    def view(self, viewer: str, token: str | None = None) -> dict[str, Any]:
        """
        viewer = "spectator" – widok sędziego (wszystko), tylko gdy nie gra człowiek
                               albo partia jest zakończona,
        viewer = gracz-człowiek – wyłącznie jego obserwacja (wymaga tokenu).
        Obserwacji bota nie da się pobrać jako "viewer=A/B" – to byłoby podglądanie kart.
        """
        self._refresh_status()
        g = self.game
        if viewer == "spectator":
            if self.human and self.in_progress:
                raise ArenaError(403, "Widok sędziego (karty obu graczy) jest zablokowany, dopóki w partii "
                                      "gra człowiek. Użyj widoku swojego gracza.")
            return self._base_view(judge=True, viewer="spectator") | {"viewer": "spectator", "judge": self._judge_view()}
        if viewer not in PLAYERS:
            raise ArenaError(400, f"Nieznany widok {viewer!r}.")
        if viewer != self.human:
            raise ArenaError(403, "Obserwacja bota jest dostępna tylko w widoku sędziego (mecz bez człowieka).")
        if not token or not secrets.compare_digest(token, self.human_token or ""):
            raise ArenaError(403, "Niepoprawny token gracza.")
        observation = g.get_state(viewer)
        view = self._base_view(judge=self.open_cards, viewer=viewer) | {
            "viewer": viewer, "observation": observation,
            "your_turn": g.current_player == viewer and self.in_progress}
        if self.open_cards:   # świadomie wybrany tryb jawny (sandbox): człowiek widzi to co sędzia
            view["judge"] = self._judge_view()
        return view

    def _base_view(self, judge: bool, viewer: str) -> dict[str, Any]:
        g, r = self.game, self.game.round
        current_round = r.number
        log = []
        for entry in self.log:
            e = dict(entry)
            # Człowiek nie widzi uzasadnień ani analizy ruchów bota z trwającego rozdania
            # (mogłyby zdradzać karty bota) – chyba że gra z jawnymi kartami.
            if not judge and e["player"] != self.human and e["round"] == current_round and self.in_progress:
                e["reasoning"] = "(ukryte do końca rozdania)" if e["reasoning"] else ""
                if e.get("analysis"):
                    e["analysis"] = {"status": "hidden"}
            log.append(e)
        seats = {}
        for p, s in self.seats.items():
            ch = self.channels.get(p)
            seats[p] = {"kind": s.kind, "label": s.label or {"llm": "Bot LLM", "human": "Człowiek",
                                                               "heuristic": "Bot regułowy"}[s.kind],
                        "model": ch.model if isinstance(ch, BotChannel) else (s.kind if s.kind != "llm" else None),
                        "preset": s.preset if s.kind == "llm" and s.strategy_source == "preset" else None,
                        "strategy": (ch.strategy if judge and ch else None),
                        "context_id": ch.context_id if judge and ch else None}
        return {
            "match_id": self.id,
            "mode": self.mode,
            "status": self.status,
            "last_error": self.last_error,
            "warnings": self.warnings,
            "seats": seats,
            "human": self.human,
            "round": r.number,
            "max_rounds": self.max_rounds,
            "phase": g.phase,
            "current_player": g.current_player,
            "dealer": r.dealer,
            "scores": dict(g.scores),
            "winner": g.winner,
            "public": {
                "table": [{"player": p, "card": c.code} for p, c in r.trick],
                "trump": r.trump,
                "bid": r.bid,
                "bid_winner": r.bidder or r.high_bidder,
                "contract": r.contract,
                "revealed_musik": [c.code for c in r.musiki[r.chosen_musik]] if r.chosen_musik is not None else None,
                "tricks_played": r.tricks_done,
                "trick_points": {p: sum(c.points for c in r.taken[p]) for p in g.players},
                "melds": {p: list(r.melds[p]) for p in g.players},
                "hand_sizes": {p: len(r.hands[p]) for p in g.players},
            },
            "log": log,
            "round_results": [e["public"] for e in g._log if e["public"]["action"] == "round_end"],
            "open_cards": self.open_cards,
            "analysis_enabled": self.analysis,
            "solver": SOLVER.status() if self.analysis else None,
            "analysis_summary": _summary(log),
        }

    def _judge_view(self) -> dict[str, Any]:
        """Wszystko, co wie silnik – TYLKO dla ludzi-obserwatorów, nigdy dla modeli."""
        g, r = self.game, self.game.round
        history = []
        for e in g._log:
            ev = dict(e["public"])
            if e["private"]:
                ev["private"] = e["private"]
            history.append(ev)
        return {
            "hands": {p: [c.code for c in r.hands[p]] for p in g.players},
            "musiki": [[c.code for c in m] for m in r.musiki],
            "chosen_musik": r.chosen_musik,
            "discards": [c.code for c in r.discards],
            "history": [h for h in history if h.get("round") == r.number],
            "legal_moves": g.legal_moves(g.current_player) if g.current_player else [],
            "bot_observations": {p: ch.last_observation for p, ch in self.channels.items() if ch},
        }


def _summary(log: list[dict[str, Any]]) -> dict[str, Any]:
    """Jakość gry według solvera: średnia strata i odsetek ruchów optymalnych (z widocznych wpisów)."""
    out: dict[str, Any] = {}
    for p in PLAYERS:
        done = [e["analysis"] for e in log if e["player"] == p and (e.get("analysis") or {}).get("status") == "done"
                and e["analysis"].get("loss") is not None]
        losses = [a["loss"] for a in done]
        out[p] = {
            "analysed": len(losses),
            "pending": sum(1 for e in log if e["player"] == p and (e.get("analysis") or {}).get("status") == "pending"),
            "total_loss": round(sum(losses), 1),
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
            "optimal": sum(1 for x in losses if x < 0.5),
            "worst": max(losses) if losses else None,
            "errors": sum(1 for e in log if e["player"] == p
                          and (e.get("analysis") or {}).get("status") in ("error", "unavailable")),
        }
    return out


# ---------------------------------------------------------------------------
# Serwis (wywoływany przez trasy HTTP)
# ---------------------------------------------------------------------------

class ArenaService:
    """Rejestr meczów w pamięci procesu (projekt edukacyjny – bez bazy danych)."""

    def __init__(self) -> None:
        self.matches: dict[str, Match] = {}

    def _gc(self) -> None:
        now = time.time()
        for mid in [m.id for m in self.matches.values() if now - m.touched > MATCH_TTL_S]:
            del self.matches[mid]
        while len(self.matches) >= MAX_MATCHES:
            oldest = min(self.matches.values(), key=lambda m: m.touched)
            del self.matches[oldest.id]

    def get(self, match_id: str) -> Match:
        match = self.matches.get(match_id)
        if not match:
            raise ArenaError(404, "Nie ma takiego meczu (mógł wygasnąć po restarcie serwera).")
        return match

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            req = CreateMatchRequest.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            raise ArenaError(422, f"Niepoprawne dane: {'.'.join(map(str, first['loc']))}: {first['msg']}") from None

        humans = [p for p, s in req.seats.items() if s.kind == "human"]
        if len(humans) > 1:
            raise ArenaError(400, "Człowiekiem może sterować najwyżej jeden gracz.")
        if req.open_cards and (req.mode != "sandbox" or not humans):
            raise ArenaError(400, "Jawne karty są dostępne tylko w sandboxie z graczem-człowiekiem.")

        # --- kanały: każdy gracz-bot dostaje WŁASNY, odizolowany kontekst ------------
        channels: dict[str, BotChannel | HeuristicChannel | None] = {}
        for p, seat in req.seats.items():
            if seat.kind == "human":
                channels[p] = None
            elif seat.kind == "heuristic":
                channels[p] = HeuristicChannel(p)
            else:
                strategy = (load_preset(seat.preset or "") if seat.strategy_source == "preset"
                            else seat.strategy.strip())
                if not strategy:
                    raise ArenaError(400, f"Gracz {p}: wpisz strategię albo wybierz preset.")
                model = ai_service.SETTINGS.resolve_model(seat.model)
                try:
                    ai_service.SETTINGS.ensure_ready(model)
                except LLMConfigError as exc:
                    raise ArenaError(400, f"Gracz {p}: {exc}") from None
                channels[p] = BotChannel(player=p, strategy=strategy, model=model)

        # --- gra: losowe rozdanie albo scenariusz sandboxa ----------------------------
        match_id = uuid.uuid4().hex
        warnings: list[str] = []
        if req.mode == "sandbox":
            spec = dict(req.sandbox or {})
            spec["players"] = list(PLAYERS)
            spec.setdefault("seed", req.seed)
            try:
                game, warnings = build_scenario(spec, game_id=match_id[:12])
            except ScenarioError as exc:
                raise ArenaError(400, f"Sandbox: {exc}") from None
        else:
            game = TysiacGame(PLAYERS, seed=req.seed, game_id=match_id[:12])

        self._gc()
        match = Match(id=match_id, mode=req.mode, game=game, seats=req.seats, channels=channels,
                      max_rounds=req.max_rounds, warnings=warnings, open_cards=req.open_cards,
                      analysis=req.analysis and SOLVER.available)
        if humans:
            match.human = humans[0]
            match.human_token = secrets.token_urlsafe(24)
        match._refresh_status()
        self.matches[match_id] = match
        logger.info("Nowy mecz %s (%s): %s", match_id, req.mode,
                    {p: (s.kind, channels[p].model if channels[p] else None) for p, s in req.seats.items()})
        return {"match_id": match_id, "human": match.human, "human_token": match.human_token,
                "warnings": warnings}

    def view(self, match_id: str, viewer: str, token: str | None) -> dict[str, Any]:
        return self.get(match_id).view(viewer, token)

    async def step(self, match_id: str, viewer: str, token: str | None) -> dict[str, Any]:
        match = self.get(match_id)
        await match.step()
        return match.view(viewer, token)

    def human_move(self, match_id: str, token: str | None, move: dict[str, Any]) -> dict[str, Any]:
        match = self.get(match_id)
        match.human_move(token, move)
        return match.view(match.human or "spectator", token)

    def retry_analysis(self, match_id: str, seq: int, viewer: str, token: str | None) -> dict[str, Any]:
        match = self.get(match_id)
        match.retry_analysis(seq)
        return match.view(viewer, token)

    def set_open_cards(self, match_id: str, token: str | None, value: bool) -> dict[str, Any]:
        match = self.get(match_id)
        match.set_open_cards(token, value)
        return match.view(match.human or "spectator", token)

    @staticmethod
    def solver_status() -> dict[str, Any]:
        return SOLVER.status()

    def delete(self, match_id: str) -> None:
        self.matches.pop(match_id, None)

    @staticmethod
    def presets() -> list[dict[str, str]]:
        return list_presets()

    @staticmethod
    def models() -> dict[str, Any]:
        settings = ai_service.SETTINGS
        return {"default": settings.model, "provider": settings.provider.name,
                "models": settings.available_models()}
