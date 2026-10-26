"""
analysis.py
===========

Analiza jakości ruchów solverem C++ (`solver/tysiac_solver.cpp`) – w tle.

Solver działa jako osobny proces (tryb `q`, wyjście JSON): dostaje obserwację
gracza z chwili decyzji (engine/solver_bridge.py) i zwraca Q(a) dla każdego
legalnego ruchu, max Q oraz stratę (LOSS = max Q − Q(zagrany ruch)).

Zadania trafiają do jednowątkowej kolejki (sam solver liczy wielowątkowo), więc
analiza nigdy nie blokuje gry ani serwera. Wyniki są cache'owane.

Konfiguracja (.env):
    SOLVER_ENABLED=1           włącz/wyłącz analizę
    SOLVER_AUTO_BUILD=1        skompiluj solver przy starcie serwera, jeśli brak pliku
    SOLVER_SAMPLES=100         liczba losowanych światów na ruch (więcej = dokładniej, wolniej)
    SOLVER_THREADS=            wątki solvera (puste = min(liczba rdzeni, 4))
    SOLVER_TT_BITS=20          rozmiar tablicy transpozycji: 2^bits wpisów (~24 MB na wątek przy 20)
    SOLVER_TIMEOUT=60          limit czasu jednej analizy (s)
    TYSIAC_SOLVER_PATH=        własna ścieżka do skompilowanego solvera
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from engine.solver_bridge import parse_q_output
from solver import build as solver_build

logger = logging.getLogger("backend.solver")


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


class SolverService:
    """Uruchamianie solvera C++ (synchronicznie albo w tle) z cache'em wyników."""

    def __init__(self) -> None:
        self.enabled = _env("SOLVER_ENABLED", "1").lower() not in ("0", "false", "no", "off", "nie")
        self.auto_build = _env("SOLVER_AUTO_BUILD", "1").lower() not in ("0", "false", "no", "off", "nie")
        self.samples = int(_env("SOLVER_SAMPLES", "100"))
        self.threads = int(_env("SOLVER_THREADS") or min(os.cpu_count() or 1, 4))
        self.tt_bits = int(_env("SOLVER_TT_BITS", "20"))
        self.timeout = float(_env("SOLVER_TIMEOUT", "60"))
        self.path = Path(_env("TYSIAC_SOLVER_PATH")) if _env("TYSIAC_SOLVER_PATH") else solver_build.BINARY
        self.reason: str | None = None           # dlaczego solver jest niedostępny
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="solver")
        self._pending = 0
        self._prepared = False

    # --- dostępność ------------------------------------------------------------------

    def prepare(self) -> None:
        """Przy starcie serwera: sprawdza (i w razie potrzeby kompiluje) solver."""
        self._prepared = True
        if not self.enabled:
            self.reason = "analiza wyłączona (SOLVER_ENABLED=0)"
            return
        if self.path == solver_build.BINARY and self.auto_build and not solver_build.is_up_to_date():
            try:
                logger.info("Kompiluję solver C++ (pierwsze uruchomienie lub zmiana źródła)…")
                solver_build.build(quiet=True)
            except solver_build.BuildError as exc:
                self.reason = f"nie udało się skompilować solvera: {exc}"
        if not self.path.is_file():
            self.reason = self.reason or (f"brak pliku solvera {self.path} – uruchom: python -m solver.build")
            return
        self.reason = None

    @property
    def available(self) -> bool:
        if not self._prepared:
            self.prepare()
        return self.enabled and self.reason is None and self.path.is_file()

    def status(self) -> dict[str, Any]:
        return {"available": self.available, "reason": self.reason, "path": str(self.path),
                "samples": self.samples, "threads": self.threads, "pending": self._pending}

    def log_status(self) -> None:
        if self.available:
            logger.info("Solver C++: %s (światów na ruch: %d, wątki: %d)", self.path, self.samples, self.threads)
        else:
            logger.warning("Solver C++ niedostępny – analiza ruchów wyłączona: %s", self.reason)

    # --- uruchamianie ------------------------------------------------------------------

    def _run(self, q_input: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.path), "q", "-", str(self.threads), str(self.tt_bits)],
                              input=q_input, capture_output=True, text=True, timeout=self.timeout)

    def analyse(self, q_input: str, fallback_input: str | None = None) -> dict[str, Any]:
        """
        Synchroniczna analiza jednego ruchu. `fallback_input` (np. bez wywnioskowanych
        pustych kolorów) jest używany, gdy solver uzna obserwację za niespójną – zdarza się
        przy ręcznie ustawionych scenariuszach sandboxa.
        """
        if not self.available:
            return {"status": "unavailable", "message": self.reason}
        with self._lock:
            if q_input in self._cache:
                self._cache.move_to_end(q_input)
                return self._cache[q_input]
        try:
            result = self._run(q_input)
            if result.returncode != 0 and fallback_input and "niespójna" in result.stderr:
                result = self._run(fallback_input)
            if result.returncode != 0:
                out = {"status": "error", "message": (result.stderr or result.stdout).strip()[:300]}
            else:
                out = {"status": "done", **parse_q_output(json.loads(result.stdout))}
        except subprocess.TimeoutExpired:
            out = {"status": "error", "message": f"przekroczono limit {self.timeout:.0f}s"}
        except (OSError, ValueError, KeyError) as exc:
            out = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
        if out["status"] == "done":
            with self._lock:
                self._cache[q_input] = out
                while len(self._cache) > 512:
                    self._cache.popitem(last=False)
        return out

    def submit(self, q_input: str, fallback_input: str | None,
               on_done: Callable[[dict[str, Any]], None]) -> None:
        """Analiza w tle; `on_done(wynik)` wywoływane w wątku solvera."""
        with self._lock:
            self._pending += 1

        def job() -> None:
            try:
                on_done(self.analyse(q_input, fallback_input))
            except Exception as exc:  # noqa: BLE001 – błąd analizy nie może zatrzymać kolejki
                logger.exception("Błąd analizy solvera")
                on_done({"status": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                with self._lock:
                    self._pending -= 1

        self._executor.submit(job)


#: Jedna instancja na proces serwera.
SOLVER = SolverService()
