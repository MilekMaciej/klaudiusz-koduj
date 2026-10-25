"""
main.py
=======

Serwer FastAPI – punkt wejścia backendu platformy "Tysiąc – Prompt Battles".

Dwa sposoby gry:

1. Interfejs webowy (http://localhost:8000/) – mecze trzymane po stronie serwera
   (`backend/arena.py`): tryb zwykły (losowe rozdania) i sandbox (ręcznie
   ustawiona sytuacja, opcjonalnie z człowiekiem jako jednym z graczy).

2. Bezstanowe API dla skryptów (`engine/client.py`, `run_battle.py`):

    [engine/]  --POST /play-turn-->  [backend/main.py]  -->  [ai_service.py]  -->  LiteLLM  -->  model
       ^                                                                                          |
       +--------------------------  { "action": ..., "card": ..., "reasoning": ... }  <-----------+

   Każde żądanie niesie obserwację JEDNEGO gracza i jego strategię. Backend
   przepuszcza do modelu tylko pola z whitelisty (engine/visibility.py), a stan
   z polami typu "opponent_hand" odrzuca.

Uruchomienie (z katalogu głównego projektu):
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# .env ładujemy PRZED importem ai_service, bo ten czyta konfigurację przy imporcie.
load_dotenv()

from fastapi import Body, FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from backend import ai_service  # noqa: E402
from backend.ai_service import AIServiceError, LLMMove, LLMUnavailableError  # noqa: E402
from backend.arena import ArenaError, ArenaService  # noqa: E402
from backend.llm_config import LLMConfigError  # noqa: E402
from engine.protocol import API_VERSION, HEALTH_PATH, PLAY_TURN_PATH  # noqa: E402  (wspólny kontrakt)
from engine.visibility import InformationLeakError, sanitize_observation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backend")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
arena = ArenaService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Przy starcie serwera: baner z aktywnym dostawcą i modelem LLM."""
    ai_service.log_active_config()
    logger.info("Interfejs webowy: http://127.0.0.1:8000/")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Tysiąc – Prompt Battles API",
    description="Modele językowe grają w Tysiąca (2 graczy) sterowane strategiami (promptami) "
                "użytkowników. Interfejs webowy pod '/', bezstanowe API dla skryptów pod "
                f"'{PLAY_TURN_PATH}'.",
    version="0.2.0",
)

# CORS – ułatwia podpięcie frontendu z innego portu podczas developmentu.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Schematy żądań / odpowiedzi (API dla skryptów)
# ---------------------------------------------------------------------------

class GameState(BaseModel):
    """
    Obserwacja JEDNEGO gracza (tylko informacje jawne + jego własna ręka),
    zbudowana przez silnik (`TysiacGame.get_state`). Nieznane pola są usuwane
    przed wysłaniem do modelu (whitelist w engine/visibility.py).
    """

    model_config = {"extra": "allow"}

    game_id: str
    player_id: str = Field(..., description="Gracz, który ma teraz ruch.")
    phase: str = Field(..., description="bidding | choose_musik | discard | declare | play")
    hand: list[str] = Field(..., description="Karty na ręce, np. ['A_spades', '10_hearts'].")
    table: list[dict[str, Any]] = Field(default_factory=list,
                                        description="Karty w bieżącej lewie: [{'player': 'B', 'card': 'K_hearts'}].")
    trump: str | None = Field(default=None, description="Aktualny kolor atutowy (po meldunku).")
    current_bid: int | None = None
    bid_winner: str | None = None
    scores: dict[str, int] = Field(default_factory=dict, description="Łączne punkty graczy.")
    round_points: dict[str, int] = Field(default_factory=dict, description="Jawne punkty w bieżącym rozdaniu.")
    history: list[dict[str, Any]] = Field(default_factory=list,
                                          description="Jawne ruchy (z perspektywy gracza).")
    legal_moves: list[dict[str, Any]] = Field(default_factory=list,
                                              description="Legalne ruchy wyliczone przez silnik.")


class MoveRequest(BaseModel):
    """Żądanie ruchu: obserwacja gracza + jego strategia (prompt użytkownika)."""

    game_state: GameState
    strategy: str = Field(..., min_length=1, max_length=8000,
                          description="Strategia gracza – trafia do modelu jako system prompt.")
    model: str | None = Field(default=None, description="Opcjonalnie inny model LiteLLM (domyślnie z .env).")


class MoveResponse(BaseModel):
    """Odpowiedź dla silnika: sparsowany ruch + metadane wywołania."""

    move: LLMMove
    provider: str
    model: str = Field(..., description="Model faktycznie użyty (może być zapasowy).")
    attempts: int
    usage: dict[str, int]
    fallback_used: bool = False


def _clean_state(req: MoveRequest) -> dict[str, Any]:
    """Obserwacja przepuszczona przez filtr widoczności (whitelist pól)."""
    try:
        state, dropped = sanitize_observation(req.game_state.model_dump(exclude_none=True))
    except InformationLeakError as exc:
        logger.warning("Odrzucono stan z ukrytymi informacjami: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if dropped:
        logger.warning("Usunięto pola spoza whitelisty obserwacji: %s", dropped)
    return state


# ---------------------------------------------------------------------------
# Obsługa błędów
# ---------------------------------------------------------------------------

@app.exception_handler(StarletteHTTPException)
async def http_error_with_hint(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Brakująca trasa (goły 404 'Not Found') -> podpowiedź; inne błędy bez zmian."""
    detail: Any = exc.detail
    if exc.status_code == 404 and detail == "Not Found":
        detail = (f"Brak endpointu {request.method} {request.url.path}. Interfejs: GET /, "
                  f"ruchy dla skryptów: POST {PLAY_TURN_PATH} (kontrakt API v{API_VERSION}).")
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(ArenaError)
async def arena_error(_request: Request, exc: ArenaError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


# ---------------------------------------------------------------------------
# API dla skryptów (bezstanowe)
# ---------------------------------------------------------------------------

@app.get(HEALTH_PATH)
async def health() -> dict[str, Any]:
    """Czy serwer działa, jaki dostawca/model jest skonfigurowany i jaki kontrakt API obsługuje."""
    llm = ai_service.SETTINGS.describe()
    return {"status": "ok", "api_version": API_VERSION, "play_turn_path": PLAY_TURN_PATH,
            "provider": llm["provider"], "model": llm["model"], "fallbacks": llm["fallbacks"],
            "llm_ready": llm["ready"]}


@app.post(PLAY_TURN_PATH, response_model=MoveResponse)
async def request_move(req: MoveRequest) -> MoveResponse:
    """
    Przyjmuje obserwację gracza i strategię, zwraca ruch modelu.

    Kody błędów:
        422 – niepoprawne dane albo stan z ukrytymi informacjami (np. 'opponent_hand'),
        500 – błąd KONFIGURACJI LLM (brak/zły klucz, nieistniejący model) – nie ponawiać,
        502 – model nie zwrócił poprawnego ruchu mimo ponowień,
        503 – dostawca LLM chwilowo niedostępny (limit zapytań, timeout).
    """
    state = _clean_state(req)
    logger.info("Ruch: game=%s player=%s phase=%s", req.game_state.game_id,
                req.game_state.player_id, req.game_state.phase)
    try:
        move, meta = await ai_service.get_move(state, req.strategy, model=req.model)
    except AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except LLMConfigError as exc:
        logger.error("Błąd konfiguracji LLM: %s", exc)
        raise HTTPException(status_code=500, detail=f"Błąd konfiguracji LLM: {exc}") from exc
    except LLMUnavailableError as exc:
        logger.error("Dostawca LLM niedostępny: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # nieprzewidziany błąd – logujemy pełny ślad, nie zabijamy serwera
        logger.exception("Nieoczekiwany błąd wywołania LLM")
        raise HTTPException(status_code=500, detail=f"Nieoczekiwany błąd backendu: "
                                                    f"{type(exc).__name__}: {exc}") from exc

    logger.info("Model %s (%s) zagrał: %s%s", meta["model"], meta["provider"],
                move.model_dump(exclude={"reasoning"}), "  [zapasowy]" if meta["fallback_used"] else "")
    return MoveResponse(move=move, **meta)


@app.post("/api/prompt-preview")
async def prompt_preview(req: MoveRequest) -> dict[str, str]:
    """Dokładnie te prompty, które trafiłyby do modelu – bez wywoływania LLM."""
    state = _clean_state(req)
    return {"system_prompt": ai_service.build_system_prompt(req.strategy),
            "user_prompt": ai_service.build_user_prompt(state)}


# ---------------------------------------------------------------------------
# Arena – mecze dla interfejsu webowego
# ---------------------------------------------------------------------------

@app.get("/api/presets")
async def presets() -> list[dict[str, str]]:
    """Gotowe strategie z katalogu strategies/*.txt."""
    return arena.presets()


@app.get("/api/models")
async def models() -> dict[str, Any]:
    """Modele do wyboru dla botów (z informacją, czy w .env jest do nich klucz)."""
    return arena.models()


@app.post("/api/matches")
async def create_match(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Nowy mecz: {mode: normal|sandbox, seats: {A: {...}, B: {...}}, seed, max_rounds, sandbox}."""
    return arena.create(payload)


@app.get("/api/matches/{match_id}")
async def match_view(match_id: str, viewer: str = "spectator",
                     x_player_token: str | None = Header(default=None)) -> dict[str, Any]:
    """Widok meczu: sędziego (bez człowieka w grze) albo gracza-człowieka (z tokenem)."""
    return arena.view(match_id, viewer, x_player_token)


@app.post("/api/matches/{match_id}/step")
async def match_step(match_id: str, viewer: str = "spectator",
                     x_player_token: str | None = Header(default=None)) -> dict[str, Any]:
    """Jeden ruch bota, który ma teraz ruch (gdy ruch ma człowiek – bez zmian)."""
    return await arena.step(match_id, viewer, x_player_token)


@app.post("/api/matches/{match_id}/move")
async def match_human_move(match_id: str, payload: dict[str, Any] = Body(...),
                           x_player_token: str | None = Header(default=None)) -> dict[str, Any]:
    """Ruch gracza-człowieka: {"move": {"action": "play_card", "card": "A_hearts"}}."""
    return arena.human_move(match_id, x_player_token, payload.get("move") or {})


@app.delete("/api/matches/{match_id}")
async def match_delete(match_id: str) -> dict[str, str]:
    arena.delete(match_id)
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Frontend (statyczne pliki z katalogu frontend/)
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
