"""
main.py
=======

Serwer FastAPI – punkt wejścia backendu platformy "Tysiąc dla LLM-ów".

Przepływ jednego kroku gry:

    [engine/]  --POST /play-turn-->  [backend/main.py]  -->  [ai_service.py]  -->  LiteLLM  -->  Muse Spark
       ^                                                                                         |
       +-------------------------  { "action": ..., "card": ..., "reasoning": ... }  <-----------+

Backend jest BEZSTANOWY: nie przechowuje partii ani historii. Każde żądanie
niesie kompletny stan gry (przygotowany przez silnik) oraz strategię gracza.

Uruchomienie (z katalogu głównego projektu):
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
from typing import Any

from dotenv import load_dotenv

# .env ładujemy PRZED importem ai_service, bo ten czyta konfigurację przy imporcie.
load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from backend import ai_service  # noqa: E402
from backend.ai_service import AIServiceError, LLMMove  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backend")

app = FastAPI(
    title="Tysiąc – LLM Arena API",
    description="Bezstanowe API, w którym modele językowe grają w Tysiąca "
                "sterowane strategiami (promptami) zdefiniowanymi przez użytkowników.",
    version="0.1.0",
)

# CORS – pozwala w przyszłości podpiąć prosty frontend (np. podgląd partii).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schematy żądań / odpowiedzi
# ---------------------------------------------------------------------------

class GameState(BaseModel):
    """
    Kompletny stan gry z perspektywy JEDNEGO gracza (tylko informacje jawne
    + jego własna ręka). Silnik buduje ten obiekt przed każdym ruchem.

    `extra="allow"` – silnik może dorzucać pola (np. meldunki, statystyki),
    a backend przekaże je modelowi bez zmian.
    """

    model_config = {"extra": "allow"}

    game_id: str
    player_id: str = Field(..., description="Gracz, który ma teraz ruch.")
    phase: str = Field(..., description="bidding | choose_musik | discard | declare | play")
    hand: list[str] = Field(..., description="Karty na ręce, np. ['A_spades', '10_hearts'].")
    table: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Karty zagrane w bieżącej lewie: [{'player': 'p2', 'card': 'K_hearts'}].",
    )
    trump: str | None = Field(default=None, description="Aktualny kolor atutowy (po meldunku).")
    current_bid: int | None = None
    bid_winner: str | None = None
    scores: dict[str, int] = Field(default_factory=dict, description="Łączne punkty graczy.")
    round_points: dict[str, int] = Field(default_factory=dict, description="Jawne punkty w bieżącym rozdaniu.")
    history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Wszystkie jawne ruchy w tej i poprzednich rundach (z perspektywy gracza).",
    )
    legal_moves: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Legalne ruchy wyliczone przez silnik, np. "
                    "[{'action': 'play_card', 'card': 'A_spades'}, {'action': 'bid', 'min': 110, 'max': 300, 'step': 10}].",
    )


class MoveRequest(BaseModel):
    """Żądanie ruchu: stan gry + strategia gracza (prompt użytkownika)."""

    game_state: GameState
    strategy: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="Strategia gracza – trafia do modelu jako system prompt.",
        examples=["Licytuj agresywnie, gdy masz meldunek. W rozgrywce najpierw wyciągaj atuty."],
    )
    model: str | None = Field(
        default=None,
        description="Opcjonalnie inny model LiteLLM (domyślnie z .env).",
    )


class MoveResponse(BaseModel):
    """Odpowiedź dla silnika: sparsowany ruch + metadane wywołania."""

    move: LLMMove
    model: str
    attempts: int
    usage: dict[str, int]


# ---------------------------------------------------------------------------
# Endpointy
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Prosty test, czy serwer działa i jaki model jest skonfigurowany."""
    return {"status": "ok", "model": ai_service.LLM_MODEL}


@app.post("/play-turn", response_model=MoveResponse)
@app.post("/api/move", response_model=MoveResponse, include_in_schema=False)  # alias wsteczny
async def request_move(req: MoveRequest) -> MoveResponse:
    """
    Główny endpoint: przyjmuje stan gry i strategię, zwraca ruch modelu.

    Kody błędów:
        422 – niepoprawne dane wejściowe (walidacja Pydantic, automatycznie),
        502 – model nie zwrócił poprawnego ruchu mimo ponowień,
        503 – błąd komunikacji z dostawcą LLM (sieć, klucz API, limity).
    """
    state = req.game_state.model_dump(exclude_none=True)
    logger.info("Ruch: game=%s player=%s phase=%s", req.game_state.game_id,
                req.game_state.player_id, req.game_state.phase)

    try:
        move, meta = await ai_service.get_move(state, req.strategy, model=req.model)
    except AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # błędy LiteLLM: autoryzacja, timeout, rate limit...
        logger.exception("Błąd wywołania LLM")
        raise HTTPException(status_code=503, detail=f"Błąd dostawcy LLM: {exc}") from exc

    logger.info("Model zagrał: %s", move.model_dump(exclude={"reasoning"}))
    return MoveResponse(move=move, **meta)


@app.post("/api/prompt-preview")
async def prompt_preview(req: MoveRequest) -> dict[str, str]:
    """
    Narzędzie diagnostyczne: zwraca DOKŁADNIE te prompty, które trafiłyby do
    modelu – bez wywoływania LLM (przydatne przy dopracowywaniu strategii).
    """
    state = req.game_state.model_dump(exclude_none=True)
    return {
        "system_prompt": ai_service.build_system_prompt(req.strategy),
        "user_prompt": ai_service.build_user_prompt(state),
    }
