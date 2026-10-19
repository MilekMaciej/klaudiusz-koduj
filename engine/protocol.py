"""
protocol.py
===========

JEDYNE źródło prawdy o kontrakcie HTTP między silnikiem (klientem) a backendem.

Importują go obie strony:
    backend/main.py   -> rejestruje endpointy pod tymi ścieżkami
    engine/client.py  -> wysyła żądania pod te same ścieżki

Dzięki temu ścieżka nie może się "rozjechać" między plikami. Moduł celowo
nie ma żadnych zależności (backend importuje go bez ciągnięcia reszty silnika).
"""

#: Wersja kontraktu. Zmieniamy ją przy każdej niekompatybilnej zmianie formatu
#: żądania/odpowiedzi – klient porównuje ją z wartością z /health.
API_VERSION = "2"

#: Endpoint zwracający ruch modelu (POST, body: {game_state, strategy, model?}).
PLAY_TURN_PATH = "/play-turn"

#: Endpoint diagnostyczny (GET) – zwraca m.in. api_version i play_turn_path.
HEALTH_PATH = "/health"
