"""
run_battle.py
=============

Narzędzie deweloperskie: pełna partia Tysiąca (2 graczy) między dwoma botami LLM
sterowanymi RÓŻNYMI strategiami, z czytelnym podglądem turowym w konsoli.

Dla każdego ruchu pokazuje: rozdanie i fazę, rękę bota, stół, atut, wykonany
ruch, uzasadnienie modelu (reasoning), liczbę prób i zużycie tokenów.
Na starcie wypisuje pełne prompty systemowe obu botów.

Tryby połączenia:
    --transport http    (domyślnie) przez backend FastAPI: POST /play-turn
    --transport direct  bez serwera – bezpośrednio przez backend/ai_service.py (LiteLLM)

Przykłady:
    python run_battle.py                                   # agresywna vs defensywna
    python run_battle.py -a strategies/agresywna.txt -b "Graj zachowawczo."
    python run_battle.py --transport direct --max-rounds 2 --show-state compact
    python run_battle.py --bot-b heuristic --step          # LLM vs bot regułowy, krok po kroku
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # przed importem ai_service (czyta konfigurację przy imporcie)

from engine.client import (  # noqa: E402
    DEFAULT_BACKEND_URL, BackendClient, BackendConfigError, BackendError, BackendUnavailableError, HeuristicAgent,
    ModelMoveError, concretize, parse_move,
)
from engine.game import IllegalMoveError, TysiacGame  # noqa: E402

STRATEGY_DIR = Path(__file__).parent / "strategies"

# ---------------------------------------------------------------------------
# Prezentacja w konsoli
# ---------------------------------------------------------------------------

SUIT_SYMBOLS = {"hearts": "♥", "diamonds": "♦", "clubs": "♣", "spades": "♠"}
PHASE_PL = {"bidding": "licytacja", "choose_musik": "wybór musiku", "discard": "odkładanie",
            "declare": "deklaracja", "play": "rozgrywka", "game_over": "koniec gry"}
USE_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def c(text: str, code: str) -> str:
    """Kolor ANSI (wyłączany przez --no-color, NO_COLOR lub przekierowanie do pliku)."""
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def card(code: str | None) -> str:
    """'10_hearts' -> '10♥' (kier i karo na czerwono)."""
    if not code:
        return "-"
    rank, suit = code.split("_", 1)
    text = f"{rank}{SUIT_SYMBOLS.get(suit, suit)}"
    return c(text, "31") if suit in ("hearts", "diamonds") else text


def cards(codes: list[str] | None) -> str:
    return " ".join(card(x) for x in codes) if codes else "-"


def describe_move(move: dict[str, Any]) -> str:
    """Opis ruchu po polsku."""
    a = move.get("action")
    return {
        "bid": lambda: f"licytuje {move.get('value')}",
        "pass": lambda: "pasuje",
        "choose_musik": lambda: f"wybiera musik nr {move.get('value')}",
        "discard": lambda: f"odkłada {card(move.get('card'))}",
        "declare": lambda: f"deklaruje kontrakt {move.get('value')}",
        "play_card": lambda: f"zagrywa {card(move.get('card'))}",
        "meld": lambda: f"MELDUJE i zagrywa {card(move.get('card'))}",
    }.get(a, lambda: json.dumps(move, ensure_ascii=False))()


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    """Stan bez długiej historii i ściągi zasad – do szybkiego podglądu."""
    out = {k: v for k, v in state.items() if k not in ("history", "rules_summary")}
    out["history"] = f"<{len(state['history'])} zdarzeń>"
    return out


# ---------------------------------------------------------------------------
# Boty
# ---------------------------------------------------------------------------

class LLMBot:
    """Bot LLM: strategia + transport (HTTP do backendu albo bezpośrednio ai_service)."""

    def __init__(self, label: str, strategy: str, model: str | None, transport: str,
                 client: BackendClient | None) -> None:
        self.label, self.strategy, self.model = label, strategy, model
        self.transport, self.client = transport, client

    def choose(self, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.transport == "http":
            assert self.client is not None
            return self.client.request_turn(state, self.strategy, self.model)
        from backend import ai_service  # import leniwy: tryb http nie wymaga litellm
        try:
            move, meta = asyncio.run(ai_service.get_move(state, self.strategy, model=self.model))
        except ai_service.AIServiceError as exc:   # model nie podał poprawnego ruchu (jak HTTP 502)
            raise ModelMoveError(str(exc)) from exc
        except ai_service.LLMConfigError as exc:   # klucz / model / dostawca (jak HTTP 500)
            raise BackendConfigError(str(exc)) from exc
        except Exception as exc:                   # błąd dostawcy LLM: klucz, limit, sieć (jak HTTP 503)
            raise BackendUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        return parse_move(move.model_dump()), meta


class HeuristicBot:
    """Bot regułowy (punkt odniesienia, zero kosztów API)."""

    def __init__(self, label: str) -> None:
        self.label, self._agent, self.strategy, self.model = label, HeuristicAgent(label), None, None

    def choose(self, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._agent.choose(state), {"model": "heuristic", "attempts": 1, "usage": {}}


def load_strategy(value: str) -> str:
    """Ścieżka do pliku, nazwa presetu ze strategies/ albo dosłowny tekst strategii."""
    for path in (Path(value), STRATEGY_DIR / value, STRATEGY_DIR / f"{value}.txt"):
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return value


# ---------------------------------------------------------------------------
# Pętla partii
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    game = TysiacGame(("A", "B"), seed=args.seed)
    client = None
    if args.transport == "direct":
        import logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.getLogger("backend.llm").setLevel(logging.WARNING if not args.verbose else logging.INFO)
        from backend import ai_service
        llm = ai_service.SETTINGS
        print(c(f"LLM (direct): {llm.provider.name} / {llm.model} | zapasowe: {', '.join(llm.fallbacks) or '-'}"
                + (" – TRYB TESTOWY" if llm.provider.name != "meta" else ""), "32"))
        try:
            llm.ensure_ready()
        except Exception as exc:  # LLMConfigError – np. brak klucza
            raise SystemExit(c(f"\n✖ {exc}", "31"))
    if args.transport == "http":
        client = BackendClient(args.backend, timeout=args.timeout)
        # Fail fast: czy backend działa i ma ten sam kontrakt (/play-turn, api_version).
        # Lepiej przerwać przed pierwszym ruchem niż w środku partii.
        try:
            info = client.verify()
        except BackendError as exc:
            raise SystemExit(c(f"\n✖ {exc}", "31"))
        print(c(f"Backend OK: {client.url} (API v{info['api_version']}) | LLM: {info.get('provider')} / "
                f"{info.get('model')}" + (" – TRYB TESTOWY" if info.get("provider") not in (None, "meta") else ""),
                "32"))

    def make_bot(label: str, kind: str, strategy: str, model: str | None):
        if kind == "heuristic":
            return HeuristicBot(label)
        return LLMBot(label, load_strategy(strategy), model, args.transport, client)

    bots = {"A": make_bot("A", args.bot_a, args.strategy_a, args.model_a),
            "B": make_bot("B", args.bot_b, args.strategy_b, args.model_b)}

    # --- nagłówek: prompty systemowe -------------------------------------------
    from backend.ai_service import build_system_prompt  # czysta funkcja, bez wywołań sieci
    print(c("═" * 78, "1"))
    print(c(f" TYSIĄC – BITWA STRATEGII   (gra {game.game_id}, seed={args.seed}, "
            f"transport={args.transport})", "1"))
    print(c("═" * 78, "1"))
    for label, bot in bots.items():
        print(c(f"\n▶ Bot {label}  [{bot.model or ('heuristic' if bot.strategy is None else 'model z .env')}]", "1;36"))
        if bot.strategy is None:
            print("  (bot regułowy – bez promptu)")
        elif args.show_prompt == "full":
            print(build_system_prompt(bot.strategy))
        else:
            print("  Strategia: " + bot.strategy.replace("\n", "\n             "))

    # buffering=1 -> zapis linia po linii (log przetrwa przerwanie partii Ctrl+C)
    log_file = open(args.log, "w", encoding="utf-8", buffering=1) if args.log else None
    totals = {p: {"moves": 0, "fallbacks": 0, "tokens": 0, "seconds": 0.0} for p in game.players}
    seen_events = 0
    last_header = None

    while not game.is_over:
        if args.max_rounds and game.round.number > args.max_rounds:
            print(c(f"\n■ Limit {args.max_rounds} rozdań – partia przerwana.", "33"))
            break

        player = game.current_player
        state = game.get_state(player)

        # --- nagłówek rozdania/fazy --------------------------------------------
        header = (state["round"], state["phase"])
        if header != last_header:
            print(c(f"\n── Rozdanie {state['round']} · {PHASE_PL[state['phase']]} "
                    f"(rozdaje {state['dealer']}) " + "─" * 30, "1;34"))
            last_header = header

        # --- stan widziany przez bota ------------------------------------------
        info = [f"ręka: {cards(state['hand'])}"]
        if state["phase"] == "play":
            info += [f"stół: {cards([t['card'] for t in state['table']])}",
                     f"atut: {SUIT_SYMBOLS.get(state['trump'], '-') if state['trump'] else '-'}",
                     f"kontrakt: {state['contract']} ({state['bid_winner']})"]
        elif state["phase"] == "bidding":
            info += [f"stawka: {state['current_bid'] or '-'}", f"max: {state['your_max_bid']}"]
        print(f"{c('[Bot ' + player + ']', '1;33')} " + " | ".join(info))
        if args.show_state == "compact":
            print(c(json.dumps(compact_state(state), ensure_ascii=False), "2"))
        elif args.show_state == "full":
            print(c(json.dumps(state, ensure_ascii=False, indent=2), "2"))

        # --- decyzja bota --------------------------------------------------------
        t0 = time.perf_counter()
        fallback = False
        try:
            move, meta = bots[player].choose(state)
            reasoning = move.pop("reasoning", "")
            game.apply_move(player, move)
        except (ModelMoveError, IllegalMoveError) as exc:
            # Model zawiódł merytorycznie. Ruch zastępczy TYLKO na wyraźne życzenie.
            if not args.fallback_on_model_error:
                abort(exc, state, player, game, totals, log_file, args)
            move, meta = concretize(state["legal_moves"][0]), {}
            reasoning, fallback = f"RUCH ZASTĘPCZY – {str(exc)[:200]}", True
            game.apply_move(player, move)
        except BackendError as exc:
            # Połączenie, 404/422, dostawca LLM: nigdy nie gramy za bota.
            abort(exc, state, player, game, totals, log_file, args)
        elapsed = time.perf_counter() - t0

        usage = meta.get("usage") or {}
        tokens = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
        stats = totals[player]
        stats["moves"] += 1
        stats["fallbacks"] += fallback
        stats["tokens"] += tokens
        stats["seconds"] += elapsed

        tag = c("⚠ ", "31") if fallback else "➜ "
        print(f"   {tag}{c(describe_move(move), '1')}   "
              + c(f"({meta.get('attempts', '-')} próba/y, {tokens} tok, {elapsed:.1f}s)", "2"))
        if reasoning:
            print(c(f"   💬 {reasoning}", "3"))

        if log_file:
            log_file.write(json.dumps({"state": state, "move": move, "reasoning": reasoning,
                                       "meta": meta, "fallback": fallback, "seconds": elapsed},
                                      ensure_ascii=False) + "\n")

        # --- zdarzenia jawne po ruchu (lewa, koniec rozdania) ---------------------
        history = game.history_for(player)
        for ev in history[seen_events:]:
            if ev["action"] == "trick_won" and args.verbose:
                print(c(f"   ✓ lewę bierze {ev['player']} ({cards(ev['cards'])}, +{ev['points']})", "32"))
            elif ev["action"] == "round_end":
                made = "UGRAŁ" if ev["contract_made"] else "NIE UGRAŁ"
                print(c(f"\n   ■ Koniec rozdania {ev['round']}: rozgrywający {ev['bidder']} {made} "
                        f"{ev['contract']} | punkty rozdania {ev['round_points']} | "
                        f"zmiana {ev['score_change']} | WYNIK {ev['scores']}", "1;35"))
        seen_events = len(history)

        if args.step:
            input(c("   [Enter] następny ruch…", "2"))

    print_summary(game, totals, log_file, args)
    return {"winner": game.winner, "scores": game.scores, "totals": totals}


def print_summary(game: TysiacGame, totals: dict[str, Any], log_file: Any, args: argparse.Namespace) -> None:
    print(c("\n" + "═" * 78, "1"))
    print(c(f" ZWYCIĘZCA: {game.winner or 'brak (przerwano)'}   WYNIK: {game.scores}", "1;32"))
    for p, s in totals.items():
        print(f"   Bot {p}: ruchów {s['moves']}, zastępczych {s['fallbacks']}, "
              f"tokenów {s['tokens']}, czas {s['seconds']:.1f}s")
    if log_file:
        log_file.close()
        print(f"   Log JSONL: {args.log}")


def abort(exc: Exception, state: dict[str, Any], player: str, game: TysiacGame,
          totals: dict[str, Any], log_file: Any, args: argparse.Namespace) -> None:
    """Czytelne przerwanie partii: co się stało, gdzie i co zrobić. Kod wyjścia 1."""
    print(c("\n" + "✖" * 78, "31"))
    print(c(f" PARTIA PRZERWANA – rozdanie {state['round']}, faza '{PHASE_PL.get(state['phase'], state['phase'])}', "
            f"ruch Bota {player}", "1;31"))
    print(c(f" [{type(exc).__name__}] {exc}", "31"))
    if isinstance(exc, (ModelMoveError, IllegalMoveError)):
        print(c(" Wskazówka: dopracuj strategię w debug_sandbox.py albo uruchom z "
                "--fallback-on-model-error.", "33"))
    if log_file:
        log_file.write(json.dumps({"aborted": True, "error_type": type(exc).__name__, "error": str(exc),
                                   "round": state["round"], "phase": state["phase"], "player": player,
                                   "state": state}, ensure_ascii=False) + "\n")
    print_summary(game, totals, log_file, args)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Partia Tysiąca: Bot A vs Bot B (różne strategie).")
    p.add_argument("-a", "--strategy-a", default="agresywna",
                   help="Strategia bota A: plik, nazwa presetu ze strategies/ lub tekst")
    p.add_argument("-b", "--strategy-b", default="defensywna", help="Strategia bota B (j.w.)")
    p.add_argument("--model-a", help="Model dla A (domyślnie z .env: LLM_PROVIDER/LLM_MODEL); "
                                     "bez prefiksu dostaje prefiks aktywnego dostawcy")
    p.add_argument("--model-b", help="Model LiteLLM dla B")
    p.add_argument("--bot-a", choices=["llm", "heuristic"], default="llm")
    p.add_argument("--bot-b", choices=["llm", "heuristic"], default="llm")
    p.add_argument("--transport", choices=["http", "direct"], default="http")
    p.add_argument("--backend", default=DEFAULT_BACKEND_URL, help="Adres backendu (tryb http)")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=None, help="Ziarno tasowania (powtarzalne rozdania)")
    p.add_argument("--max-rounds", type=int, default=60, help="Limit rozdań (kontrola kosztów, 0 = bez limitu)")
    p.add_argument("--fallback-on-model-error", action="store_true",
                   help="Gdy model nie poda poprawnego ruchu: zagraj pierwszy legalny zamiast przerywać "
                        "(błędy połączenia/API zawsze przerywają)")
    p.add_argument("--show-prompt", choices=["full", "strategy"], default="full",
                   help="Na starcie: pełny prompt systemowy czy tylko strategia")
    p.add_argument("--show-state", choices=["off", "compact", "full"], default="off",
                   help="Co turę: podgląd JSON-a wysyłanego do modelu (user prompt)")
    p.add_argument("-v", "--verbose", action="store_true", help="Pokazuj też wyniki każdej lewy")
    p.add_argument("--step", action="store_true", help="Pauza (Enter) po każdym ruchu")
    p.add_argument("--log", help="Zapisz każdą turę (stan, ruch, reasoning) do pliku JSONL")
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.no_color:
        USE_COLOR = False
    try:
        run(arguments)
    except KeyboardInterrupt:
        print("\nPrzerwano.")
