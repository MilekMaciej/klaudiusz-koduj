# Tysiąc · Prompt Battles

Platforma, na której modele językowe (LLM) grają w **Tysiąca w wariancie dla 2 graczy**.
Człowiek nie klika ruchów, tylko pisze **strategię**, czyli prompt systemowy, i patrzy, jak radzi sobie
jego bot. Pojedynek toczy się między promptami.

## Szybki start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # ustaw LLM_PROVIDER i klucz (Groq i Gemini są darmowe, ChatGPT płatny)
python -m backend.llm_config --ping                    # sprawdza klucze i modele
python -m solver.build                                 # kompiluje solver C++ (opcjonalnie – serwer zrobi to sam)
uvicorn backend.main:app --reload                      # uruchamiać z katalogu głównego projektu
```

Interfejs: **http://127.0.0.1:8000/**, dokumentacja API: http://127.0.0.1:8000/docs

## Interfejs webowy

Przed startem wybierasz **tryb**:

| Tryb | Co robi |
|---|---|
| **Zwykła rozgrywka** | Losowe rozdania, partia do 1000 pkt (albo do limitu rozdań). |
| **Sandbox** | Sam ustawiasz sytuację: ręce obu graczy, musik, licytację, rozegrane lewy (z meldunkami), kartę na stole, wyniki i dodatkową historię. Brakujące karty są dolosowywane, a partia toczy się dalej według zasad. |

Dla każdego gracza (A, B) wybierasz:
- **kto gra**: bot LLM, bot regułowy (darmowy, bez API) albo **człowiek**, który wybiera ruchy w formularzu (najwyżej jeden gracz),
- **model**: lista modeli, do których w `.env` jest klucz; każdy bot może mieć inny model,
- **strategię**: gotowy preset z `strategies/*.txt` albo własny prompt (preset można skopiować do edycji).

W sandboxie z człowiekiem wybierasz też **widoczność kart**: **Klasyczna** (karty przeciwnika
ukryte) albo **Jawna** (otwarte karty: ręka bota, musiki i odłożone karty widoczne na bieżąco).
Tryb można przełączać w trakcie gry przyciskiem „👁”. Jawne karty widzi tylko człowiek, a bot
nadal dostaje wyłącznie własną obserwację.

Karty w sandboxie zapisujesz skrótem `Ah 10h Kc Qd Js 9s` (h ♥, d ♦, c ♣, s ♠). Lewa to jedna linia:
`Ad 9d`, `A: Qh* 9h` (A wychodzi damą z meldunkiem), `Kc Ac > B` (wymuszony zwycięzca).

## Analiza ruchów: solver C++ (Q i strata)

Każdy ruch w rozgrywce, zarówno bota, jak i człowieka, jest w tle oceniany solverem double-dummy
`solver/tysiac_solver.cpp`:

- **Q(a)**: średnia wartość rozdania (wynik rozgrywającego − obrońcy, z perspektywy gracza) przy
  optymalnej grze obu stron, liczona w losowych układach kart zgodnych z tym, co gracz wiedział
  w chwili decyzji (domyślnie 100 „światów”),
- **max Q**: wartość najlepszego ruchu,
- **strata (Equity Loss) = max Q − Q(zagrany ruch)**: ile punktów kosztował wybór.

W przebiegu partii przy każdym ruchu widać odznakę („✓ optymalny” albo „strata 5.7”), Q i najlepszy
ruch, a po rozwinięciu tabelę Q wszystkich legalnych ruchów. Nad przebiegiem jest podsumowanie
graczy: średnia strata i liczba ruchów optymalnych. W meczu z człowiekiem analiza ruchów bota
z trwającego rozdania jest ukryta (lista alternatyw zdradzałaby karty bota), chyba że wybrano
jawne karty.

**Integracja:** solver działa jako osobny proces (`subprocess`, wejście przez stdin, wyjście JSON).
Nie potrzebuje pybind11, błąd w C++ nie wywraca serwera, a obliczenia idą w wątkach C++ bez GIL.
Narzut uruchomienia procesu (kilka ms) jest pomijalny. Analizy trafiają do kolejki w tle i są
cache'owane, więc gra nigdy na nie nie czeka.

**Zgodność zasad.** Oryginalny solver miał cztery różnice względem silnika. Dodałem flagi, których
wartości domyślne zachowują zachowanie oryginału:

| Zasada | Flaga | Silnik |
|---|---|---|
| Obowiązek przebijania w kolorze | `mustbeat` | wyłączony (`Rules.must_beat_in_suit`) |
| Obowiązek atutu przy braku koloru | `musttrump` | włączony |
| Meldunek w 1. lewie | `meldfirst` | niedozwolony |
| Karty martwe | `dead_to_last 0` | odłożone → rozgrywający, niewybrany musik → nikt |

Wykonane testy:
- lista legalnych ruchów solvera pokrywa się z silnikiem (480 pozycji, 4 warianty zasad),
- wartości double-dummy zgadzają się co do punktu z niezależnym minimaksem w Pythonie,
- selftest solvera (alpha-beta + TT kontra czysty minimaks) losuje także nowe flagi.

Zasada beczki (800+) nie jest modelowana, bo solver ocenia pojedyncze rozdanie.

Ręczne użycie: `python run_battle.py --transport direct --analyse` albo bezpośrednio
`solver/bin/tysiac_solver q plik.txt` (format wejścia opisany w nagłówku pliku C++).

## Izolacja botów (anti-cheat)

1. **Osobne konteksty.** Każdy bot ma własny `BotChannel` (`backend/arena.py`) i zna tylko swoją
   strategię i swój model. Każde wywołanie modelu to nowa, jednorazowa rozmowa (system = strategia,
   user = obserwacja). Boty nie mają wspólnej historii czatu, cache ani pamięci. Uzasadnienia botów
   nie trafiają do historii gry.
2. **Częściowa obserwowalność.** Silnik buduje obserwację dla jednego gracza
   (`TysiacGame.get_state`): jego ręka, stół, jawna historia, legalne ruchy. Ręka przeciwnika,
   niewybrany musik i cudze odłożone karty nigdy się w niej nie znajdują.
3. **Niezależny audyt** (`engine/visibility.py`). Przed każdym zapytaniem do modelu audyt:
   - sprawdza listę dozwolonych pól (whitelist),
   - wylicza z pełnego stanu gry zbiór kart ukrytych przed graczem i upewnia się, że żadna z nich
     nie występuje w obserwacji.

   Gdy wykryje wyciek, zapytanie nie jest wysyłane.
4. **API dla skryptów** (`/play-turn`) usuwa pola spoza whitelisty, a stan z polami w rodzaju
   `opponent_hand` odrzuca (HTTP 422).
5. **Człowiek też nie podejrzy kart.** Dostaje tajny token i tylko swój widok. Widok sędziego
   (karty obu graczy) jest zablokowany, dopóki trwa partia z człowiekiem.

Widok sędziego w meczu dwóch botów pokazuje dokładnie to, co dostał każdy bot („Co dokładnie widzi bot?”).

## Modele i dostawcy (`.env`)

| `LLM_PROVIDER` | Model domyślny | Klucz | Uwagi |
|---|---|---|---|
| `meta` | `meta/muse-spark-1.3` | `META_API_KEY` | docelowy |
| `groq` | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` | darmowy, model zapasowy `llama-3.1-8b-instant` |
| `gemini` | `gemini/gemini-flash-latest` | `GEMINI_API_KEY` | darmowy; alias zawsze wskazuje bieżący Flash |
| `openai` (`chatgpt`) | `openai/gpt-5.6-terra` | `OPENAI_API_KEY` | ChatGPT, płatne API; do wyboru też `gpt-5.6-luna` (najtańszy) i `gpt-5.6-sol` (najmocniejszy) |
| `custom` | `LLM_MODEL` | `LLM_API_KEY_ENV` | np. lokalna Ollama |

- Parametry (temperatura, tryb JSON, `reasoning_effort`, klucz) są dobierane osobno dla każdego modelu.
  Na przykład dla Gemini i GPT-5.x nie wysyłamy temperatury (Gemini 3 zaleca 1.0, a rozumujące modele
  OpenAI przyjmują tylko wartość domyślną). Brak środków na koncie OpenAI (`insufficient_quota`) daje
  od razu czytelny błąd zamiast ponawiania.
- Limit zapytań (429): backend czeka (`Retry-After` / `retryDelay` / backoff) i ponawia **ten sam**
  model. Wyczerpany limit dobowy kończy się od razu czytelnym błędem.
- Gdy model nie poda poprawnego ruchu albo dostawca zawiedzie, mecz wstrzymuje się z komunikatem
  i przyciskiem „Ponów”. Żaden ruch nie jest wykonywany za bota.

## Struktura projektu

```
engine/
  game.py         silnik Tysiąca (2 graczy): zasady, punktacja, beczka, get_state()
  visibility.py   filtr widoczności i audyt anti-cheat
  scenario.py     sandbox: budowanie sytuacji przy stole na prawdziwym silniku
  client.py       klient HTTP (/play-turn) i pętla meczu dla skryptów
  protocol.py     wspólny kontrakt HTTP (ścieżki, wersja API)
  solver_bridge.py  stan gry -> wejście solvera (tylko wiedza gracza), parsowanie wyniku
backend/
  main.py         FastAPI: interfejs, arena, bezstanowe /play-turn
  arena.py        mecze w przeglądarce: izolowane kanały botów, gracz-człowiek
  ai_service.py   LiteLLM: prompty, wymuszanie JSON, obsługa błędów dostawców
  llm_config.py   dostawcy i modele z .env (Meta / Groq / Gemini / OpenAI / custom)
  analysis.py     analiza ruchów solverem C++ w tle (kolejka, cache, auto-kompilacja)
solver/
  tysiac_solver.cpp  solver double-dummy + Q(a) / strata (C++17, wielowątkowy)
  build.py           kompilacja: python -m solver.build (g++ / clang++ / MSVC)
frontend/         index.html, app.js, style.css (bez frameworków)
strategies/       presety strategii (*.txt) – dodaj plik, a pojawi się w interfejsie
scenarios/        przykładowe scenariusze dla debug_sandbox.py
run_battle.py     partia dwóch botów w konsoli (tryb http lub direct)
debug_sandbox.py  test promptu na ręcznie ustawionej sytuacji -> surowy JSON modelu
```

## Narzędzia w konsoli

```bash
python run_battle.py --max-rounds 3 -a agresywna -b defensywna           # przez backend
python run_battle.py --transport direct --model-a gpt-5.6-luna --model-b gemini/gemini-flash-latest
python debug_sandbox.py --scenario scenarios/meldunek.json --show-prompt --repeat 5
python run_battle.py --transport direct --bot-a heuristic --analyse      # strata każdego ruchu w konsoli
python -m engine.client --offline --seed 7                               # boty regułowe, bez API
```

Darmowe plany mają limity zapytań na minutę i na dobę, a pełna partia to kilkaset wywołań. Do testów
warto ustawić mały limit rozdań albo grać przeciwko botowi regułowemu.
