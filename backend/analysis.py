"""
analysis.py
===========

Analiza jakości ruchów solverem C++ (`solver/tysiac_solver.cpp`) – w tle.

Przepływ jednego zadania:
    1. engine/solver_bridge.build_q_input   – stan gry -> tekst "klucz wartości" (format readKV w C++)
    2. engine/solver_bridge.validate_q_input – walidacja PRZED wywołaniem procesu
    3. podproces:  tysiac_solver q - <wątki> <ttbits>   (wejście: stdin, wyjście: JSON na stdout)
    4. JSON -> Q(a) każdego ruchu, max Q, strata (LOSS) zagranego ruchu

Odporność na błędy:
  * każde wywołanie w try-except – błąd analizy nigdy nie przerywa gry ani zapytania HTTP,
  * pełny stderr i opis kodu wyjścia (sygnał / kod Windows) trafiają do logów serwera,
  * wejście, które wywołało błąd, jest zapisywane do logs/solver/*.txt – plik można
    od razu odtworzyć:  solver/bin/tysiac_solver q logs/solver/<plik>.txt
  * awaria procesu (crash, SIGILL, brak DLL) -> jednorazowa przebudowa solvera
    w wersji przenośnej i ponowienie,
  * do UI wraca czytelny komunikat: {"status": "error", "kind": ..., "message": ...}.

Konfiguracja (.env):
    SOLVER_ENABLED=1           włącz/wyłącz analizę
    SOLVER_AUTO_BUILD=1        kompiluj solver przy starcie / po awarii
    SOLVER_SAMPLES=100         liczba losowanych światów na ruch
    SOLVER_THREADS=            wątki solvera (puste = min(liczba rdzeni, 4))
    SOLVER_TT_BITS=20          tablica transpozycji: 2^bity wpisów (~24 MB na wątek przy 20)
    SOLVER_TIMEOUT=60          limit czasu jednej analizy (s)
    SOLVER_DEBUG_DIR=logs/solver  gdzie zapisywać wejścia, które wywołały błąd
    TYSIAC_SOLVER_PATH=        własna ścieżka do skompilowanego solvera
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from engine.solver_bridge import SolverInputError, parse_q_output, validate_q_input
from solver import build as solver_build

logger = logging.getLogger("backend.solver")
PROJECT_DIR = Path(__file__).resolve().parent.parent

#: Kategorie błędów zwracane do UI (kind) i czy ponowienie ma sens.
ERROR_KINDS = {
    "input": ("Niepoprawne dane dla solvera", False),
    "inconsistent": ("Niespójna sytuacja przy stole", False),
    "rules_mismatch": ("Niezgodność zasad silnika i solvera", False),
    "crash": ("Awaria procesu solvera", True),
    "timeout": ("Przekroczony czas analizy", True),
    "memory": ("Za mało pamięci dla solvera", True),
    "bad_output": ("Nieczytelna odpowiedź solvera", True),
    "internal": ("Błąd wewnętrzny solvera", True),
}


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _int_env(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(_env(name) or default)
    except ValueError:
        logger.warning("%s=%r nie jest liczbą – używam %d", name, _env(name), default)
        value = default
    return max(lo, min(hi, value))


class SolverService:
    """Uruchamianie solvera C++ (synchronicznie albo w tle) z cache'em wyników."""

    def __init__(self) -> None:
        self.enabled = _env("SOLVER_ENABLED", "1").lower() not in ("0", "false", "no", "off", "nie")
        self.auto_build = _env("SOLVER_AUTO_BUILD", "1").lower() not in ("0", "false", "no", "off", "nie")
        self.samples = _int_env("SOLVER_SAMPLES", 100, 1, 5000)
        self.threads = _int_env("SOLVER_THREADS", min(os.cpu_count() or 1, 4), 1, 64)
        self.tt_bits = _int_env("SOLVER_TT_BITS", 20, 12, 26)
        self.timeout = float(_env("SOLVER_TIMEOUT", "60") or 60)
        self.custom_path = bool(_env("TYSIAC_SOLVER_PATH"))
        self.path = Path(_env("TYSIAC_SOLVER_PATH")) if self.custom_path else solver_build.BINARY
        debug_dir = Path(_env("SOLVER_DEBUG_DIR", "logs/solver"))
        self.debug_dir = debug_dir if debug_dir.is_absolute() else PROJECT_DIR / debug_dir
        self.reason: str | None = None            # dlaczego solver jest niedostępny
        self.last_error: dict[str, Any] | None = None
        self.stats = {"done": 0, "errors": 0}
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._rebuild_lock = threading.Lock()
        self._rebuild_attempted = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="solver")
        self._pending = 0
        self._prepared = False

    # --- dostępność ------------------------------------------------------------------

    def prepare(self) -> None:
        """Przy starcie serwera: kompiluje solver (jeśli trzeba) i sprawdza, czy DA SIĘ go uruchomić."""
        self._prepared = True
        self.reason = None
        if not self.enabled:
            self.reason = "analiza wyłączona (SOLVER_ENABLED=0)"
            return
        if not self.custom_path and self.auto_build and not solver_build.is_up_to_date():
            try:
                logger.info("Kompiluję solver C++ (brak pliku, zmiana źródła albo inna platforma)…")
                solver_build.build(quiet=True)
            except solver_build.BuildError as exc:
                self.reason = f"nie udało się skompilować solvera: {exc}"
                logger.error("Solver C++: %s", self.reason)
                return
        if not self.path.is_file():
            self.reason = f"brak pliku solvera {self.path} – uruchom: python -m solver.build"
            return
        self._health_check_or_rebuild()

    def _health_check_or_rebuild(self) -> bool:
        """Selftest pliku wykonywalnego; przy awarii – przebudowa (wersja przenośna) i drugi test."""
        ok, text = self._health_check()
        if ok:
            self.reason = None
            return True
        logger.error("Solver C++ nie działa na tym komputerze: %s", text)
        if not self.custom_path and self.auto_build:
            try:
                logger.info("Przebudowuję solver od zera (kod przenośny, bez -march=native)…")
                solver_build.build(force=True, quiet=True, native=False)
                ok, text = self._health_check()
            except solver_build.BuildError as exc:
                text = f"{text}; przebudowa nie powiodła się: {exc}"
        self.reason = None if ok else f"solver nie działa: {text}"
        if ok:
            logger.info("Solver C++ działa po przebudowie.")
        return ok

    def _health_check(self) -> tuple[bool, str]:
        if self.path == solver_build.BINARY:
            return solver_build.health_check(40, timeout=self.timeout)
        try:
            result = subprocess.run([str(self.path), "selftest", "40"], capture_output=True,
                                    stdin=subprocess.DEVNULL, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return False, f"selftest przekroczył {self.timeout:.0f}s"
        except OSError as exc:
            return False, f"nie można uruchomić {self.path}: {exc}"
        out = (result.stdout + result.stderr).decode("utf-8", "replace").strip()
        return result.returncode == 0, out if result.returncode == 0 else f"{solver_build.describe_exit(result.returncode)} {out}"

    @property
    def available(self) -> bool:
        if not self._prepared:
            self.prepare()
        return self.enabled and self.reason is None and self.path.is_file()

    def status(self) -> dict[str, Any]:
        info = solver_build.build_info() if self.path == solver_build.BINARY else {}
        return {"available": self.available, "reason": self.reason, "path": str(self.path),
                "samples": self.samples, "threads": self.threads, "pending": self._pending,
                "build": {k: info.get(k) for k in ("compiler", "platform", "native")} if info else None,
                "stats": dict(self.stats), "last_error": self.last_error}

    def log_status(self) -> None:
        if self.available:
            info = solver_build.build_info()
            logger.info("Solver C++: %s (światów na ruch: %d, wątki: %d, %s)", self.path, self.samples, self.threads,
                        "-march=native" if info.get("native") else "kod przenośny")
        else:
            logger.warning("Solver C++ niedostępny – analiza ruchów wyłączona: %s", self.reason)

    # --- uruchamianie ------------------------------------------------------------------

    def _run(self, q_input: str) -> tuple[int, str, str]:
        """Jedno wywołanie procesu. Zwraca (kod wyjścia, stdout, stderr) – tekst zawsze w UTF-8."""
        result = subprocess.run([str(self.path), "q", "-", str(self.threads), str(self.tt_bits)],
                                input=q_input.encode("utf-8"), capture_output=True, timeout=self.timeout)
        return (result.returncode, result.stdout.decode("utf-8", "replace"),
                result.stderr.decode("utf-8", "replace"))

    @staticmethod
    def _parse(code: int, stdout: str, stderr: str) -> dict[str, Any]:
        """Wynik procesu -> {"status": "done", ...} albo opis błędu z kategorią."""
        payload = None
        for line in reversed(stdout.strip().splitlines()):   # JSON to ostatnia niepusta linia
            if line.lstrip().startswith("{"):
                try:
                    payload = json.loads(line)
                except ValueError:
                    payload = None
                break
        if isinstance(payload, dict) and "error" in payload:
            kind = payload.get("code") if payload.get("code") in ERROR_KINDS else "internal"
            return _error(kind, payload["error"], exit_code=code, stderr=stderr)
        if code != 0:
            return _error("crash", f"proces zakończył się błędem – {solver_build.describe_exit(code)}",
                          exit_code=code, stderr=stderr)
        if not isinstance(payload, dict):
            return _error("bad_output", "solver nie zwrócił JSON-a (czy plik wykonywalny pochodzi z aktualnego "
                                        "źródła? przebuduj: python -m solver.build --force)",
                          exit_code=code, stderr=stderr or ("stdout: " + stdout[-300:]))
        try:
            parsed = parse_q_output(payload)
        except (KeyError, TypeError, ValueError) as exc:
            return _error("bad_output", f"nieoczekiwany format JSON solvera: {type(exc).__name__}: {exc}",
                          exit_code=code, stderr=stdout[-300:])
        if parsed.get("model") is None:
            return _error("rules_mismatch", "zagrany ruch nie jest legalny według solvera – zasady silnika i "
                                            "solvera się różnią (sprawdź flagi w solver_bridge.solver_rules)",
                          exit_code=code)
        return {"status": "done", **parsed}

    def analyse(self, q_input: str, fallback_input: str | None = None) -> dict[str, Any]:
        """
        Synchroniczna analiza jednego ruchu. Nigdy nie rzuca wyjątku – zwraca wynik albo
        opis błędu. `fallback_input` (bez wywnioskowanych pustych kolorów przeciwnika)
        jest używany, gdy solver uzna obserwację za niespójną.
        """
        if not self.available:
            return {"status": "unavailable", "kind": "unavailable", "message": self.reason, "retryable": False}
        with self._lock:
            if q_input in self._cache:
                self._cache.move_to_end(q_input)
                return self._cache[q_input]

        used_input = q_input
        try:
            try:
                validate_q_input(q_input)
            except SolverInputError as exc:
                if not fallback_input:
                    raise
                validate_q_input(fallback_input)
                used_input = fallback_input
            out = self._parse(*self._run(used_input))
            if out["status"] != "done" and out["kind"] == "inconsistent" and fallback_input and used_input != fallback_input:
                logger.info("Solver: obserwacja niespójna (%s) – ponawiam bez wywnioskowanych pustych kolorów",
                            out["message"])
                used_input = fallback_input
                out = self._parse(*self._run(used_input))
            if out["status"] != "done" and out["kind"] == "crash" and self._try_rebuild():
                out = self._parse(*self._run(used_input))
        except SolverInputError as exc:
            out = _error("input", str(exc))
        except subprocess.TimeoutExpired:
            out = _error("timeout", f"analiza trwała dłużej niż {self.timeout:.0f}s (zmniejsz SOLVER_SAMPLES)")
        except OSError as exc:
            out = _error("crash", f"nie można uruchomić solvera {self.path}: {exc}")
        except Exception as exc:  # noqa: BLE001 – ostatnia linia obrony: analiza nie może wywrócić serwera
            logger.exception("Nieoczekiwany błąd wrappera solvera")
            out = _error("internal", f"{type(exc).__name__}: {exc}")

        if out["status"] == "done":
            self.stats["done"] += 1
            with self._lock:
                self._cache[q_input] = out
                while len(self._cache) > 512:
                    self._cache.popitem(last=False)
        else:
            self.stats["errors"] += 1
            out["input_file"] = self._dump_failure(used_input, out)
            self.last_error = {k: out.get(k) for k in ("kind", "message", "exit_code", "input_file")}
            logger.error("Solver: %s – %s | kod wyjścia: %s | stderr: %s | wejście zapisane: %s",
                         out["kind"], out["message"], out.get("exit_code"),
                         (out.get("stderr") or "(pusty)").strip(), out["input_file"] or "-")
        return out

    def _try_rebuild(self) -> bool:
        """Po awarii procesu: jednorazowa przebudowa (kod przenośny) – usuwa SIGILL / brak DLL / stary plik."""
        if self.custom_path or not self.auto_build:
            return False
        with self._rebuild_lock:
            if self._rebuild_attempted:
                return False
            self._rebuild_attempted = True
            logger.warning("Solver uległ awarii – próbuję go przebudować i ponowić analizę.")
            return self._health_check_or_rebuild()

    def _dump_failure(self, q_input: str, out: dict[str, Any]) -> str | None:
        """Zapisuje wejście, które wywołało błąd (z opisem w komentarzach '#', które readKV pomija)."""
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            name = f"fail-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}-{out['kind']}.txt"
            path = self.debug_dir / name
            header = [f"# {ERROR_KINDS.get(out['kind'], (out['kind'],))[0]}: {out['message']}",
                      f"# kod wyjścia: {out.get('exit_code')}"]
            header += [f"# stderr: {line}" for line in (out.get("stderr") or "").strip().splitlines()[:20]]
            header.append(f"# odtworzenie: {self.path} q {path}")
            path.write_text("\n".join(header) + "\n" + q_input, encoding="utf-8")
            old = sorted(self.debug_dir.glob("fail-*.txt"))
            for f in old[:-100]:
                f.unlink(missing_ok=True)
            try:
                return str(path.relative_to(PROJECT_DIR))
            except ValueError:
                return str(path)
        except OSError as exc:
            logger.warning("Nie udało się zapisać wejścia solvera do %s: %s", self.debug_dir, exc)
            return None

    def submit(self, q_input: str, fallback_input: str | None,
               on_done: Callable[[dict[str, Any]], None]) -> None:
        """Analiza w tle; `on_done(wynik)` wywoływane w wątku solvera (zawsze – także przy błędzie)."""
        with self._lock:
            self._pending += 1

        def job() -> None:
            try:
                result = self.analyse(q_input, fallback_input)
            except Exception as exc:  # noqa: BLE001 – analyse nie powinno rzucać, ale kolejka musi przetrwać
                logger.exception("Błąd analizy solvera")
                result = _error("internal", f"{type(exc).__name__}: {exc}")
            finally:
                with self._lock:
                    self._pending -= 1
            try:
                on_done(result)
            except Exception:  # noqa: BLE001
                logger.exception("Błąd zapisu wyniku analizy")

        self._executor.submit(job)


def _error(kind: str, message: str, exit_code: int | None = None, stderr: str | None = None) -> dict[str, Any]:
    title, retryable = ERROR_KINDS.get(kind, ("Błąd analizy", True))
    return {"status": "error", "kind": kind, "title": title, "message": message,
            "retryable": retryable, "exit_code": exit_code,
            "stderr": (stderr or "").strip()[-1500:] or None}


#: Jedna instancja na proces serwera.
SOLVER = SolverService()
