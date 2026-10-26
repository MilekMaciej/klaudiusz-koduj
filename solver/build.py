"""
build.py
========

Kompilacja solvera C++ (`solver/tysiac_solver.cpp`) do pliku wykonywalnego
`solver/bin/tysiac_solver` (Windows: `tysiac_solver.exe`).

    python -m solver.build            # kompilacja + szybki selftest
    python -m solver.build --force    # kompilacja nawet gdy plik jest aktualny

Kolejność kompilatorów: zmienna CXX, g++, clang++, c++, a na Windows także cl (MSVC).
Najpierw próbujemy z -march=native (najszybszy kod dla tego procesora); jeśli
kompilator tego nie obsługuje, kompilujemy bez tej flagi.

Dlaczego osobny program, a nie moduł Pythona? Solver uruchamiany jako podproces:
  * nie wymaga pybind11 ani nagłówków Pythona – wystarczy dowolny kompilator C++17,
  * ewentualny błąd w C++ nie wywraca serwera FastAPI,
  * korzysta z własnych wątków C++ (bez blokady GIL),
  * da się go używać ręcznie z konsoli (tryby solve/q/deal/bench/selftest).
Narzut uruchomienia procesu (~ms) jest pomijalny wobec czasu obliczeń.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

SOLVER_DIR = Path(__file__).resolve().parent
SOURCE = SOLVER_DIR / "tysiac_solver.cpp"
BIN_DIR = SOLVER_DIR / "bin"
BINARY = BIN_DIR / ("tysiac_solver.exe" if os.name == "nt" else "tysiac_solver")


class BuildError(RuntimeError):
    """Nie udało się skompilować solvera."""


def _candidates() -> list[str]:
    names = [os.environ.get("CXX", ""), "g++", "clang++", "c++"]
    if os.name == "nt":
        names.append("cl")
    out = []
    for name in names:
        if name and shutil.which(name) and name not in out:
            out.append(name)
    return out


def _commands(compiler: str) -> list[list[str]]:
    if Path(compiler).stem.lower() == "cl":
        return [[compiler, "/nologo", "/O2", "/std:c++17", "/EHsc", str(SOURCE), f"/Fe:{BINARY}"]]
    base = [compiler, "-O3", "-std=c++17", "-pthread", str(SOURCE), "-o", str(BINARY)]
    return [base[:2] + ["-march=native"] + base[2:], base]


def is_up_to_date() -> bool:
    return BINARY.is_file() and BINARY.stat().st_mtime >= SOURCE.stat().st_mtime


def build(force: bool = False, quiet: bool = False) -> Path:
    """Kompiluje solver (jeśli trzeba). Zwraca ścieżkę do programu albo rzuca BuildError."""
    if not force and is_up_to_date():
        return BINARY
    if not SOURCE.is_file():
        raise BuildError(f"Brak pliku źródłowego {SOURCE}")
    compilers = _candidates()
    if not compilers:
        raise BuildError("Nie znaleziono kompilatora C++ (g++, clang++ lub MSVC cl). "
                         "Linux: sudo apt install g++ · macOS: xcode-select --install · "
                         "Windows: MSYS2 (g++) albo Visual Studio Build Tools (cl).")
    BIN_DIR.mkdir(exist_ok=True)
    errors = []
    for compiler in compilers:
        for cmd in _commands(compiler):
            if not quiet:
                print("→", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=BIN_DIR)
            if result.returncode == 0 and BINARY.is_file():
                if not quiet:
                    print(f"✓ Zbudowano {BINARY}")
                return BINARY
            errors.append(f"{' '.join(cmd[:4])}…: {(result.stderr or result.stdout).strip()[:400]}")
    raise BuildError("Kompilacja solvera nie powiodła się:\n" + "\n".join(errors))


def selftest(positions: int = 300) -> str:
    """Porównanie alpha-beta+TT z czystym minimaksem na losowych pozycjach."""
    result = subprocess.run([str(BINARY), "selftest", str(positions)], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise BuildError(f"Selftest solvera nie przeszedł: {result.stdout} {result.stderr}")
    return result.stdout.strip()


if __name__ == "__main__":
    try:
        path = build(force="--force" in sys.argv)
        print(f"✓ Solver gotowy: {path}")
        print("✓", selftest())
    except BuildError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        sys.exit(1)
