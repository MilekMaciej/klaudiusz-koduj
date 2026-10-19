"""
debug_sandbox.py
================

Piaskownica do testowania promptów: "oszukany" stan gry -> wybrany model -> surowy JSON.

Zamiast rozgrywać partię od zera, ręcznie ustawiasz sytuację przy stole:
rękę bota, kartę przeciwnika na stole, atut, kontrakt, wyniki, karty już
zagrane itd. Stan budowany jest przez PRAWDZIWY silnik (`engine/game.py`),
więc `legal_moves` i cały JSON są dokładnie takie, jak w realnej grze.
Następnie stan i Twój prompt trafiają do modelu przez LiteLLM.

Tryby wywołania:
    (domyślnie) --mode raw      jedno surowe wywołanie LiteLLM; pokazuje DOKŁADNĄ
                                odpowiedź modelu + ocenę: poprawny JSON? legalny ruch?
                --mode service  ścieżka produkcyjna backend/ai_service.get_move
                                (ponowienia, walidacja) – tak, jak w prawdziwej grze
                --dry-run       tylko pokaż stan/prompty, bez wywołania modelu

Karty można podawać jako kody silnika ('A_hearts') albo skrótem: Ah, 10h, Kc, 9s, Qd
(h=kier, d=karo, c=trefl, s=pik; T = 10).

Przykłady:
    # Bot wychodzi do lewy, ma meldunek kier – czy go zgłosi?
    python debug_sandbox.py --hand "Ah 10h Kh Qh Kc 9s" --tricks-done 3 --contract 120

    # Przeciwnik wyszedł królem trefl, atut pik – czym bot dołoży?
    python debug_sandbox.py --hand "Ah 10c Js 9d" --table Kc --trump spades \
        --role defender --contract 140 --played "As 10s Kd Qd" --strategy defensywna

    # Licytacja: przeciwnik dał 120, ile podbije bot? (5 prób – stabilność decyzji)
    python debug_sandbox.py --phase bidding --hand "Ah 10h Kh Qh Ad 10d Kc Jc 9s 9h" --bid 120 --repeat 5

    # Scenariusz z pliku (CLI nadpisuje pola z pliku) i zapis bieżącego scenariusza
    python debug_sandbox.py --scenario scenarios/meldunek.json --save-scenario moj.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # przed importem ai_service (czyta konfigurację przy imporcie)

from backend import ai_service  # noqa: E402
from engine.game import (  # noqa: E402
    HAND_SIZE, MELD_POINTS, SUITS, Card, Phase, TysiacGame, new_deck, sort_hand,
)

BOT, OPP = "BOT", "OPP"
STRATEGY_DIR = Path(__file__).parent / "strategies"
SHORT_SUITS = {"h": "hearts", "d": "diamonds", "c": "clubs", "s": "spades",
               "♥": "hearts", "♦": "diamonds", "♣": "clubs", "♠": "spades"}

# Pola scenariusza (te same nazwy w CLI i w pliku JSON).
SCENARIO_FIELDS = ("phase", "hand", "table", "trump", "role", "bid", "contract", "played",
                   "tricks_done", "melds_bot", "melds_opp", "scores", "musik", "history",
                   "strategy", "seed")


# ---------------------------------------------------------------------------
# Parsowanie kart
# ---------------------------------------------------------------------------

def parse_card(token: str) -> Card:
    """'A_hearts' | 'Ah' | '10h' | 'Th' | 'Q♠' -> Card."""
    token = token.strip()
    if "_" in token:
        return Card.from_code(token)
    rank, suit = token[:-1].upper(), SHORT_SUITS.get(token[-1].lower())
    rank = "10" if rank == "T" else rank
    if suit is None:
        raise SystemExit(f"Nieznany kolor w karcie {token!r} (użyj h/d/c/s).")
    return Card.from_code(f"{rank}_{suit}")


def parse_cards(value: str | list[str] | None) -> list[Card]:
    if not value:
        return []
    tokens = value if isinstance(value, list) else value.replace(",", " ").split()
    return [parse_card(t) for t in tokens]


def parse_suits(value: str | list[str] | None) -> list[str]:
    if not value:
        return []
    items = value if isinstance(value, list) else value.replace(",", " ").split()
    suits = [SHORT_SUITS.get(s, s) for s in items]
    bad = [s for s in suits if s not in SUITS]
    if bad:
        raise SystemExit(f"Nieznany kolor meldunku: {bad}")
    return suits


# ---------------------------------------------------------------------------
# Budowa "oszukanego" stanu na prawdziwym silniku
# ---------------------------------------------------------------------------

def build_game(sc: dict[str, Any]) -> TysiacGame:
    """
    Tworzy TysiacGame i nadpisuje bieżące rozdanie zgodnie ze scenariuszem.
    Karty, których nie podano (ręka przeciwnika, musiki), są dolosowywane
    z pozostałej części talii – tak, by talia zawsze była spójna (24 karty).
    """
    rng = random.Random(sc.get("seed"))
    game = TysiacGame((BOT, OPP), seed=sc.get("seed"), game_id="sandbox")
    r = game.round
    phase = sc.get("phase") or Phase.PLAY

    hand = parse_cards(sc.get("hand"))
    table = parse_cards(sc.get("table"))
    played = parse_cards(sc.get("played"))
    musik = parse_cards(sc.get("musik"))
    if not hand:
        raise SystemExit("Podaj rękę bota: --hand \"Ah 10h Kc ...\"")
    if len(table) > 1:
        raise SystemExit("W grze dwuosobowej na stole może leżeć najwyżej 1 karta (wyjście przeciwnika).")

    used = hand + table + played
    dupes = [c.code for c, n in Counter(used).items() if n > 1]
    if dupes:
        raise SystemExit(f"Te karty występują więcej niż raz: {dupes}")
    # Wybrany musik to karty, które rozgrywający WZIĄŁ do ręki – mogą więc
    # pokrywać się z ręką/zagranymi kartami; nie mogą trafić do dolosowania.
    rest = [c for c in new_deck() if c not in used and c not in musik]
    rng.shuffle(rest)

    def take(n: int) -> list[Card]:
        if n > len(rest):
            raise SystemExit("Za mało kart w talii dla takiego scenariusza – sprawdź liczby kart.")
        return [rest.pop() for _ in range(n)]

    # --- liczba kart przeciwnika wynika z fazy i sytuacji przy stole -------------
    if phase == Phase.PLAY:
        opp_size = len(hand) - (1 if table else 0)
    elif phase == Phase.DISCARD:
        opp_size = HAND_SIZE
    else:
        opp_size = len(hand)
    opp_hand = take(opp_size)

    # --- nadpisanie stanu rozdania -----------------------------------------------
    r.hands = {BOT: sort_hand(hand), OPP: sort_hand(opp_hand)}
    r.dealer = OPP
    r.taken = {BOT: [], OPP: []}
    r.melds = {BOT: parse_suits(sc.get("melds_bot")), OPP: parse_suits(sc.get("melds_opp"))}
    r.played = played + table
    r.trump = SHORT_SUITS.get(sc.get("trump") or "", sc.get("trump")) or None
    bid = sc.get("bid")
    role = sc.get("role") or "bidder"
    bidder = BOT if role == "bidder" else OPP
    scores = sc.get("scores") or [0, 0]
    game.scores = {BOT: int(scores[0]), OPP: int(scores[1])}

    def revealed_musik(owner: str) -> list[Card]:
        """Odkryty musik: podany w scenariuszu albo 2 karty z ręki rozgrywającego."""
        return musik or rng.sample(r.hands[owner], min(2, len(r.hands[owner])))

    if phase == Phase.BIDDING:
        r.bid, r.high_bidder = (bid, OPP) if bid else (None, None)
        r.bid_turn = BOT
        r.musiki = [take(2), take(2)]
    elif phase == Phase.CHOOSE_MUSIK:
        r.bid, r.high_bidder, r.bidder = bid or 100, BOT, BOT
        r.musiki = [take(2), take(2)]
    elif phase == Phase.DISCARD:          # bot ma 12 kart (10 + musik)
        r.bid, r.high_bidder, r.bidder, r.chosen_musik = bid or 100, BOT, BOT, 0
        r.musiki = [revealed_musik(BOT), take(2)]
    elif phase == Phase.DECLARE:          # bot już odłożył 2 karty
        r.bid, r.high_bidder, r.bidder, r.chosen_musik = bid or 100, BOT, BOT, 0
        r.discards = take(2)
        r.musiki = [revealed_musik(BOT), take(2)]
    elif phase == Phase.PLAY:
        contract = sc.get("contract") or bid or 100
        r.bid = r.contract = contract
        r.high_bidder = r.bidder = bidder
        r.chosen_musik = 0
        r.discards = take(2)              # odłożone karty rozgrywającego (BOT lub OPP)
        r.musiki = [revealed_musik(bidder), take(2)]
        r.tricks_done = int(sc["tricks_done"]) if sc.get("tricks_done") is not None else len(played) // 2
        # Karty zagrane wcześniej trafiają do wziętych lew (rozdzielone umownie: na przemian).
        for i in range(0, len(played) - 1, 2):
            r.taken[BOT if (i // 2) % 2 == 0 else OPP].extend(played[i:i + 2])
        if table:
            r.leader, r.trick = OPP, [(OPP, table[0])]
        else:
            r.leader, r.trick = BOT, []
    else:
        raise SystemExit(f"Nieobsługiwana faza: {phase}")

    game.phase = phase

    # --- historia: notatka piaskownicy + ewentualne zdarzenia ze scenariusza ------
    game._log.clear()
    game._record(None, "sandbox_note",
                 note="Stan przygotowany ręcznie w piaskownicy – historia może być niepełna.")
    for ev in sc.get("history") or []:
        game._log.append({"public": {"seq": len(game._log) + 1, **ev}, "private": {}, "owner": None})
    for suit in r.melds[BOT] + r.melds[OPP]:
        who = BOT if suit in r.melds[BOT] else OPP
        game._record(who, "meld", suit=suit, points=MELD_POINTS[suit], new_trump=suit)

    if not game.legal_moves(BOT):
        raise SystemExit("W tym scenariuszu bot nie ma żadnego legalnego ruchu – sprawdź dane.")
    return game


# ---------------------------------------------------------------------------
# Wywołania modelu
# ---------------------------------------------------------------------------

async def call_raw(state: dict[str, Any], strategy: str, model: str,
                   temperature: float | None) -> dict[str, Any]:
    """Jedno surowe wywołanie LiteLLM – te same prompty i parametry co backend, bez ponowień."""
    import litellm

    kwargs: dict[str, Any] = {}
    if ai_service.LLM_REASONING_EFFORT:
        kwargs["reasoning_effort"] = ai_service.LLM_REASONING_EFFORT
    t0 = time.perf_counter()
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "system", "content": ai_service.build_system_prompt(strategy)},
                  {"role": "user", "content": ai_service.build_user_prompt(state)}],
        temperature=ai_service.LLM_TEMPERATURE if temperature is None else temperature,
        timeout=ai_service.LLM_TIMEOUT,
        response_format={"type": "json_object"},
        **kwargs,
    )
    raw = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)

    verdict: dict[str, Any] = {"valid_json": False, "valid_schema": False, "legal": False}
    move = None
    try:
        parsed = ai_service._extract_json(raw)
        verdict["valid_json"] = True
        move = ai_service.LLMMove.model_validate(parsed)
        verdict["valid_schema"] = True
        verdict["legal"] = ai_service._move_matches_legal(move, state["legal_moves"])
    except Exception as exc:  # noqa: BLE001 – w piaskownicy chcemy zobaczyć każdy błąd
        verdict["error"] = str(exc)[:300]
    return {
        "raw": raw,
        "move": move.model_dump() if move else None,
        "verdict": verdict,
        "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                  "completion_tokens": getattr(usage, "completion_tokens", None)} if usage else None,
        "seconds": round(time.perf_counter() - t0, 2),
    }


async def call_service(state: dict[str, Any], strategy: str, model: str) -> dict[str, Any]:
    """Ścieżka produkcyjna (z ponowieniami i walidacją legalności)."""
    t0 = time.perf_counter()
    try:
        move, meta = await ai_service.get_move(state, strategy, model=model)
        return {"move": move.model_dump(), **meta, "seconds": round(time.perf_counter() - t0, 2)}
    except ai_service.AIServiceError as exc:
        return {"error": str(exc), "seconds": round(time.perf_counter() - t0, 2)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_strategy(value: str | None) -> str:
    if not value:
        return "Graj rozsądnie."
    for path in (Path(value), STRATEGY_DIR / value, STRATEGY_DIR / f"{value}.txt"):
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return value


def section(title: str) -> None:
    print(f"\n\033[1;34m── {title} " + "─" * max(0, 70 - len(title)) + "\033[0m"
          if sys.stdout.isatty() else f"\n── {title} " + "─" * max(0, 70 - len(title)))


def main() -> None:
    p = argparse.ArgumentParser(description="Piaskownica: ręcznie ustawiony stan gry -> model -> surowy JSON.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_argument_group("scenariusz (stan gry)")
    s.add_argument("--scenario", help="Plik JSON ze scenariuszem (pola jak opcje CLI)")
    s.add_argument("--save-scenario", help="Zapisz wynikowy scenariusz do pliku JSON")
    s.add_argument("--phase", choices=["bidding", "choose_musik", "discard", "declare", "play"])
    s.add_argument("--hand", help='Ręka bota, np. "Ah 10h Kc" lub "A_hearts,10_hearts,K_clubs"')
    s.add_argument("--table", help="Karta wyłożona przez przeciwnika w bieżącej lewie (np. Kc)")
    s.add_argument("--trump", help="Atut: hearts/diamonds/clubs/spades (lub h/d/c/s)")
    s.add_argument("--role", choices=["bidder", "defender"], help="Rola bota w rozdaniu (faza play)")
    s.add_argument("--bid", type=int, help="Aktualna stawka (licytacja) / wylicytowana wartość")
    s.add_argument("--contract", type=int, help="Kontrakt w fazie rozgrywki")
    s.add_argument("--played", help="Karty zagrane wcześniej w tym rozdaniu (parami = lewy)")
    s.add_argument("--tricks-done", dest="tricks_done", type=int, help="Ile lew już rozegrano")
    s.add_argument("--melds-bot", dest="melds_bot", help="Meldunki bota w tym rozdaniu, np. 'hearts'")
    s.add_argument("--melds-opp", dest="melds_opp", help="Meldunki przeciwnika w tym rozdaniu")
    s.add_argument("--scores", type=lambda v: [int(x) for x in v.split(",")], help="Wyniki BOT,OPP np. 650,820")
    s.add_argument("--musik", help="Karty wybranego musiku (fazy discard/declare/play)")
    s.add_argument("--seed", type=int, help="Ziarno dolosowania brakujących kart")

    m = p.add_argument_group("model i prompt")
    m.add_argument("--strategy", help="Prompt-strategia: plik, preset ze strategies/ lub tekst")
    m.add_argument("--model", help=f"Model LiteLLM (domyślnie {ai_service.LLM_MODEL})")
    m.add_argument("--temperature", type=float, help="Nadpisz temperaturę (tryb raw)")
    m.add_argument("--mode", choices=["raw", "service"], default="raw")
    m.add_argument("--repeat", type=int, default=1, help="Ile razy zapytać model (test stabilności)")
    m.add_argument("--show-prompt", action="store_true", help="Wypisz pełny system i user prompt")
    m.add_argument("--dry-run", action="store_true", help="Nie wywołuj modelu – tylko pokaż stan")
    args = p.parse_args()

    # --- scenariusz: plik + nadpisania z CLI ----------------------------------------
    scenario: dict[str, Any] = {}
    if args.scenario:
        scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    for key in SCENARIO_FIELDS:
        value = getattr(args, key, None)
        if value is not None:
            scenario[key] = value
    if args.save_scenario:
        Path(args.save_scenario).write_text(json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Zapisano scenariusz: {args.save_scenario}")

    game = build_game(scenario)
    state = game.get_state(BOT)
    strategy = load_strategy(scenario.get("strategy"))
    model = args.model or ai_service.LLM_MODEL

    section("STAN DLA BOTA (skrót)")
    brief = {k: state[k] for k in ("phase", "your_role", "hand", "table", "trump", "current_bid",
                                   "contract", "your_max_bid", "tricks_played", "scores")}
    print(json.dumps(brief, ensure_ascii=False))
    section("LEGALNE RUCHY")
    for lm in state["legal_moves"]:
        print("  ", json.dumps(lm, ensure_ascii=False))

    if args.show_prompt:
        section("SYSTEM PROMPT")
        print(ai_service.build_system_prompt(strategy))
        section("USER PROMPT")
        print(ai_service.build_user_prompt(state))

    if args.dry_run:
        return

    section(f"ODPOWIEDŹ MODELU  [{model}, tryb {args.mode}]")
    tally: Counter[str] = Counter()
    for i in range(1, args.repeat + 1):
        if args.mode == "raw":
            result = asyncio.run(call_raw(state, strategy, model, args.temperature))
            if args.repeat > 1:
                print(f"\n# próba {i}")
            print("SUROWA ODPOWIEDŹ:")
            print(result["raw"])
            print("OCENA:", json.dumps({**result["verdict"], "usage": result["usage"],
                                         "seconds": result["seconds"]}, ensure_ascii=False))
            key = result["move"]
        else:
            result = asyncio.run(call_service(state, strategy, model))
            if args.repeat > 1:
                print(f"\n# próba {i}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            key = result.get("move")
        if key:
            tally[json.dumps({k: v for k, v in key.items() if k != "reasoning" and v is not None},
                             ensure_ascii=False)] += 1
        else:
            tally["<błąd>"] += 1

    if args.repeat > 1:
        section(f"ROZKŁAD DECYZJI ({args.repeat} prób)")
        for move, n in tally.most_common():
            print(f"  {n:>3}×  {move}")


if __name__ == "__main__":
    main()
