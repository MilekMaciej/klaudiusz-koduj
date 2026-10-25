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
(h=kier, d=karo, c=trefl, s=pik; T = 10). Stan buduje ten sam silnik scenariuszy co
sandbox w interfejsie webowym (engine/scenario.py): brakujące karty są dolosowywane,
a historia rozdania odtwarzana zgodnie z zasadami.

Przykłady:
    # Bot wychodzi do lewy, ma meldunek kier – czy go zgłosi?
    python debug_sandbox.py --hand "Ah 10h Kh Qh Kc 9s" --tricks-done 4 --contract 120

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
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # przed importem ai_service (czyta konfigurację przy imporcie)

from backend import ai_service  # noqa: E402
from backend.llm_config import LLMConfigError  # noqa: E402
from engine.scenario import ScenarioError, build_scenario  # noqa: E402

BOT, OPP = "BOT", "OPP"
STRATEGY_DIR = Path(__file__).parent / "strategies"

# Pola scenariusza (te same nazwy w CLI i w pliku JSON).
SCENARIO_FIELDS = ("phase", "hand", "opp_hand", "table", "trump", "role", "bid", "contract", "played",
                   "tricks", "tricks_done", "melds_bot", "melds_opp", "scores", "musik", "history",
                   "strategy", "seed")


# ---------------------------------------------------------------------------
# Budowa "oszukanego" stanu – wspólny silnik scenariuszy (engine/scenario.py)
# ---------------------------------------------------------------------------

def to_spec(sc: dict[str, Any]) -> dict[str, Any]:
    """
    Tłumaczy scenariusz piaskownicy (perspektywa jednego bota: BOT vs OPP) na
    specyfikację `engine.scenario.build_scenario` – tę samą, której używa
    sandbox w interfejsie webowym. Bot ma zawsze ruch (to_move = BOT).
    """
    played = sc.get("played")
    tricks = sc.get("tricks")
    if not tricks and played:  # stary format: karty parami, wychodzący i zwycięzca wg zasad
        tokens = played.replace(",", " ").split() if isinstance(played, str) else list(played)
        if len(tokens) % 2:
            raise ScenarioError("--played: liczba kart musi być parzysta (lewa = 2 karty).")
        tricks = "\n".join(f"{tokens[i]} {tokens[i + 1]}" for i in range(0, len(tokens), 2))
    scores = sc.get("scores") or [0, 0]
    return {
        "players": [BOT, OPP],
        "to_move": BOT,
        "phase": sc.get("phase") or "play",
        "hands": {BOT: sc.get("hand"), OPP: sc.get("opp_hand")},
        "table": sc.get("table"),
        "trump": sc.get("trump"),
        "bidder": OPP if sc.get("role") == "defender" else BOT,
        "bid": sc.get("bid"),
        "contract": sc.get("contract"),
        "tricks": tricks,
        "tricks_done": sc.get("tricks_done"),
        "melds": {BOT: sc.get("melds_bot") or [], OPP: sc.get("melds_opp") or []},
        "scores": {BOT: scores[0], OPP: scores[1]} if isinstance(scores, list) else scores,
        "musik": sc.get("musik"),
        "history": sc.get("history"),
        "seed": sc.get("seed"),
    }


def build_game(sc: dict[str, Any]):
    """Scenariusz piaskownicy -> TysiacGame (błędy jako czytelny komunikat i wyjście)."""
    if not sc.get("hand") and (sc.get("phase") or "play") == "play" and not sc.get("tricks_done"):
        raise SystemExit('Podaj rękę bota: --hand "Ah 10h Kc ..." (albo --tricks-done, by dolosować).')
    try:
        game, warnings = build_scenario(to_spec(sc), game_id="sandbox")
    except ScenarioError as exc:
        raise SystemExit(f"✖ Niepoprawny scenariusz: {exc}") from None
    for w in warnings:
        print(f"⚠ {w}")
    return game


# ---------------------------------------------------------------------------
# Wywołania modelu
# ---------------------------------------------------------------------------

async def call_raw(state: dict[str, Any], strategy: str, model: str,
                   temperature: float | None) -> dict[str, Any]:
    """
    Jedno surowe wywołanie modelu – te same prompty, parametry i obsługa błędów
    dostawcy co w backendzie (ai_service.complete), ale BEZ ponowień z feedbackiem
    i bez modeli zapasowych: testujemy dokładnie wskazany model.
    """
    if temperature is not None:
        ai_service.SETTINGS.temperature = temperature
    t0 = time.perf_counter()
    messages = [{"role": "system", "content": ai_service.build_system_prompt(strategy)},
                {"role": "user", "content": ai_service.build_user_prompt(state)}]
    try:
        result = await ai_service.complete(messages, model=model, use_fallbacks=False)
    except ai_service.ModelOutputError as exc:
        return {"raw": exc.raw or "", "move": None, "usage": None,
                "verdict": {"valid_json": False, "valid_schema": False, "legal": False, "error": str(exc)},
                "seconds": round(time.perf_counter() - t0, 2)}
    except (LLMConfigError, ai_service.LLMUnavailableError) as exc:
        raise SystemExit(f"\n✖ [{type(exc).__name__}] {exc}\n  Diagnostyka: python -m backend.llm_config --ping")
    raw = result.raw

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
        "usage": result.usage,
        "model": result.model,
        "seconds": round(time.perf_counter() - t0, 2),
    }


async def call_service(state: dict[str, Any], strategy: str, model: str) -> dict[str, Any]:
    """Ścieżka produkcyjna (ponowienia, modele zapasowe, walidacja legalności)."""
    t0 = time.perf_counter()
    try:
        move, meta = await ai_service.get_move(state, strategy, model=model)
        return {"move": move.model_dump(), **meta, "seconds": round(time.perf_counter() - t0, 2)}
    except ai_service.AIServiceError as exc:
        return {"error": str(exc), "seconds": round(time.perf_counter() - t0, 2)}
    except (LLMConfigError, ai_service.LLMUnavailableError) as exc:
        raise SystemExit(f"\n✖ [{type(exc).__name__}] {exc}\n  Diagnostyka: python -m backend.llm_config --ping")


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
    s.add_argument("--played", help="Karty zagrane wcześniej w tym rozdaniu (parami = lewy, wychodzący wg zasad)")
    s.add_argument("--tricks", help='Lewy, linia = lewa: "OPP: Qh* 9h\\nAh 9s > BOT" (* = meldunek)')
    s.add_argument("--opp-hand", dest="opp_hand", help="Ręka przeciwnika (domyślnie losowa)")
    s.add_argument("--tricks-done", dest="tricks_done", type=int, help="Ile lew już rozegrano")
    s.add_argument("--melds-bot", dest="melds_bot", help="Meldunki bota w tym rozdaniu, np. 'hearts'")
    s.add_argument("--melds-opp", dest="melds_opp", help="Meldunki przeciwnika w tym rozdaniu")
    s.add_argument("--scores", type=lambda v: [int(x) for x in v.split(",")], help="Wyniki BOT,OPP np. 650,820")
    s.add_argument("--musik", help="Karty wybranego musiku (fazy discard/declare/play)")
    s.add_argument("--seed", type=int, help="Ziarno dolosowania brakujących kart")

    m = p.add_argument_group("model i prompt")
    m.add_argument("--strategy", help="Prompt-strategia: plik, preset ze strategies/ lub tekst")
    m.add_argument("--model", help=f"Model LiteLLM (domyślnie {ai_service.SETTINGS.model} – dostawca "
                                   f"{ai_service.SETTINGS.provider.name}); nazwa bez prefiksu dostaje prefiks dostawcy")
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
    model = ai_service.SETTINGS.resolve_model(args.model)

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

    section(f"ODPOWIEDŹ MODELU  [{ai_service.SETTINGS.provider.name}: {model}, tryb {args.mode}]")
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
