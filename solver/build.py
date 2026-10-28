"""
build.py
========

Kompilacja solvera C++ (`solver/tysiac_solver.cpp`) do pliku wykonywalnego
`solver/bin/tysiac_solver` (Windows: `tysiac_solver.exe`).

    python -m solver.build            # kompilacja (jeśli potrzebna) + test działania
    python -m solver.build --force    # kompilacja od zera
    python -m solver.build --native   # szybszy kod pod TEN procesor (-march=native)

Kolejność kompilatorów: zmienna CXX, g++, clang++, c++, a na Windows także cl (MSVC).

Przenośność (najczęstsze przyczyny "cichych" awarii solvera):
  * Domyślnie BEZ -march=native. Plik zbudowany z -march=native na jednym komputerze
    pada bez żadnego komunikatu (SIGILL / 0xC000001D) na procesorze bez tych samych
    instrukcji – np. we wspólnym folderze zespołu, w Dockerze albo w WSL. Szybszą wersję
    włączysz flagą --native lub zmienną SOLVER_NATIVE=1.
  * Windows + MinGW (g++ z MSYS2): linkujemy statycznie (-static). Bez tego program
    uruchomiony spoza konsoli MSYS2 nie znajduje libstdc++-6.dll i kończy się kodem
    0xC0000135, też bez komunikatu.
  * O przebudowie decyduje plik bin/build_info.json (suma SHA-256 źródła + system +
    architektura), a nie data modyfikacji – rozpakowanie archiwum lub git checkout
    nie zostawi starego, niepasującego pliku.

Dlaczego osobny program, a nie moduł Pythona? Solver uruchamiany jako podproces:
  * nie wymaga pybind11 ani nagłówków Pythona – wystarczy dowolny kompilator C++17,
  * ewentualny błąd w C++ nie wywraca serwera FastAPI,
  * korzysta z własnych wątków C++ (bez blokady GIL),
  * da się go używać ręcznie z konsoli (tryby solve/q/deal/bench/selftest).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

SOLVER_DIR = Path(__file__).resolve().parent
SOURCE = SOLVER_DIR / "tysiac_solver.cpp"
BIN_DIR = SOLVER_DIR / "bin"
BINARY = BIN_DIR / ("tysiac_solver.exe" if os.name == "nt" else "tysiac_solver")
BUILD_INFO = BIN_DIR / "build_info.json"


class BuildError(RuntimeError):
    """Nie udało się skompilować (albo uruchomić) solvera."""


# ---------------------------------------------------------------------------
# Opis kodów wyjścia (wspólny dla budowania i backendu)
# ---------------------------------------------------------------------------

#: Kody NTSTATUS Windows, którymi kończy się proces po awarii (bez komunikatu na stderr).
WINDOWS_CODES = {
    0xC0000135: "brak wymaganej biblioteki DLL (np. libstdc++-6.dll z MinGW) – przebuduj: "
                "python -m solver.build --force (wersja statyczna)",
    0xC0000005: "naruszenie dostępu do pamięci (access violation)",
    0xC000001D: "niedozwolona instrukcja procesora – plik zbudowany na innym komputerze z -march=native; "
                "przebuduj: python -m solver.build --force",
    0xC0000094: "dzielenie przez zero",
    0xC00000FD: "przepełnienie stosu",
    0xC0000409: "przerwanie programu (abort / błąd bufora stosu)",
    0xC0000017: "brak pamięci",
}

#: Sygnały POSIX (kod wyjścia subprocess = -numer sygnału).
SIGNAL_HINTS = {
    "SIGILL": "niedozwolona instrukcja procesora – plik zbudowany na innym komputerze z -march=native; "
              "przebuduj: python -m solver.build --force",
    "SIGSEGV": "naruszenie pamięci (crash solvera)",
    "SIGFPE": "błąd arytmetyczny (np. dzielenie przez zero)",
    "SIGABRT": "przerwanie programu (abort / nieobsłużony wyjątek C++)",
    "SIGKILL": "proces zabity z zewnątrz – najczęściej brak pamięci (OOM); zmniejsz SOLVER_TT_BITS lub SOLVER_THREADS",
    "SIGBUS": "błąd szyny pamięci",
}


def describe_exit(code: int | None) -> str:
    """Czytelny opis kodu wyjścia procesu solvera."""
    if code is None:
        return "proces nie zakończył się"
    if code == 0:
        return "OK"
    if code == 1:
        return "kod 1 – złe argumenty wywołania solvera"
    if code == 2:
        return "kod 2 – solver odrzucił dane lub obliczenia się nie powiodły (szczegóły na stderr)"
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"sygnał {-code}"
        return f"{name}: {SIGNAL_HINTS.get(name, 'proces zakończony sygnałem')}"
    ntstatus = code & 0xFFFFFFFF
    if ntstatus in WINDOWS_CODES:
        return f"0x{ntstatus:08X}: {WINDOWS_CODES[ntstatus]}"
    if code > 128 and code - 128 < 65:      # powłoka zgłasza sygnał jako 128+n
        return describe_exit(-(code - 128))
    return f"kod {code}"


# ---------------------------------------------------------------------------
# Budowanie
# ---------------------------------------------------------------------------

def _source_hash() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def _platform_id() -> str:
    return f"{platform.system()}-{platform.machine()}"


def _candidates() -> list[str]:
    names = [os.environ.get("CXX", ""), "g++", "clang++", "c++"]
    if os.name == "nt":
        names.append("cl")
    out = []
    for name in names:
        if name and shutil.which(name) and name not in out:
            out.append(name)
    return out


def _commands(compiler: str, native: bool) -> list[list[str]]:
    if Path(compiler).stem.lower() == "cl":
        return [[compiler, "/nologo", "/O2", "/std:c++17", "/EHsc", str(SOURCE), f"/Fe:{BINARY}"]]
    base = [compiler, "-O3", "-std=c++17", "-pthread", str(SOURCE), "-o", str(BINARY)]
    if os.name == "nt":                         # MinGW: bez zależności od DLL-i runtime'u
        base[1:1] = ["-static", "-static-libgcc", "-static-libstdc++"]
    cmds = [base]
    if native:
        cmds.insert(0, base[:2] + ["-march=native"] + base[2:])
    return cmds


def build_info() -> dict:
    try:
        return json.loads(BUILD_INFO.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def is_up_to_date() -> bool:
    """Plik istnieje i został zbudowany z TEGO źródła na TEJ platformie."""
    info = build_info()
    return (BINARY.is_file() and SOURCE.is_file() and info.get("source_sha256") == _source_hash()
            and info.get("platform") == _platform_id())


def build(force: bool = False, quiet: bool = False, native: bool | None = None) -> Path:
    """Kompiluje solver (jeśli trzeba). Zwraca ścieżkę do programu albo rzuca BuildError."""
    if not force and is_up_to_date():
        return BINARY
    if not SOURCE.is_file():
        raise BuildError(f"Brak pliku źródłowego {SOURCE}")
    if native is None:
        native = os.environ.get("SOLVER_NATIVE", "0").lower() in ("1", "true", "yes", "tak")
    compilers = _candidates()
    if not compilers:
        raise BuildError("Nie znaleziono kompilatora C++ (g++, clang++ lub MSVC cl). "
                         "Linux: sudo apt install g++ · macOS: xcode-select --install · "
                         "Windows: MSYS2 (pacman -S mingw-w64-ucrt-x86_64-gcc) albo Visual Studio Build Tools.")
    BIN_DIR.mkdir(exist_ok=True)
    errors = []
    for compiler in compilers:
        for cmd in _commands(compiler, native):
            if not quiet:
                print("→", " ".join(cmd))
            try:
                result = subprocess.run(cmd, capture_output=True, cwd=BIN_DIR, timeout=300)
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"{compiler}: {exc}")
                continue
            if result.returncode == 0 and BINARY.is_file():
                BUILD_INFO.write_text(json.dumps({
                    "source_sha256": _source_hash(), "platform": _platform_id(), "compiler": compiler,
                    "flags": [a for a in cmd[1:] if a.startswith(("-", "/")) and "tysiac" not in a],
                    "native": "-march=native" in cmd}, indent=2), encoding="utf-8")
                if not quiet:
                    print(f"✓ Zbudowano {BINARY}")
                return BINARY
            text = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
            errors.append(f"{' '.join(cmd[:4])}…: {text[:400]}")
    raise BuildError("Kompilacja solvera nie powiodła się:\n" + "\n".join(errors))


def health_check(positions: int = 40, timeout: float = 60) -> tuple[bool, str]:
    """
    Czy zbudowany plik da się uruchomić na tym komputerze i liczy poprawnie
    (selftest: alpha-beta + tablica transpozycji == czysty minimaks).
    Zwraca (ok, opis) – opis zawiera przyczynę awarii (np. SIGILL, brak DLL).
    """
    if not BINARY.is_file():
        return False, f"brak pliku {BINARY}"
    try:
        result = subprocess.run([str(BINARY), "selftest", str(positions)], capture_output=True,
                                stdin=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"selftest przekroczył {timeout:.0f}s"
    except OSError as exc:
        return False, f"nie można uruchomić {BINARY}: {exc}"
    out = (result.stdout + result.stderr).decode("utf-8", "replace").strip()
    if result.returncode != 0:
        return False, f"{describe_exit(result.returncode)}{(' – ' + out[:200]) if out else ''}"
    return True, out


if __name__ == "__main__":
    try:
        path = build(force="--force" in sys.argv, native=True if "--native" in sys.argv else None)
        print(f"✓ Solver gotowy: {path} ({'-march=native' if build_info().get('native') else 'kod przenośny'})")
        ok, text = health_check(300)
        print(("✓ " if ok else "✗ ") + text)
        sys.exit(0 if ok else 1)
    except BuildError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        sys.exit(1)
