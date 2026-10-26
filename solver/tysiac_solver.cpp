// tysiac_solver.cpp  --  Tysiąc na 2 graczy, solver double-dummy + equity loss
//
// Kompilacja:  python -m solver.build           (wybiera g++ / clang++ / MSVC)
//         lub: g++ -O3 -march=native -std=c++17 -pthread tysiac_solver.cpp -o tysiac_solver
//
// [integracja z Prompt Battles] Zmiany względem wersji źródłowej (oznaczone "[integracja]"):
//   * przenośne nagłówki zamiast <bits/stdc++.h> + zamienniki __builtin_* dla MSVC,
//   * flagi zasad w Cfg (wartości domyślne = dotychczasowe zachowanie solvera):
//       musttrump    0/1  obowiązek dania atutu bez koloru (domyślnie = mustbeat)
//       meldfirst    0/1  meldunek dozwolony w 1. lewie   (domyślnie 1)
//       dead_to_last 0/1  karty martwe dla zdobywcy ostatniej lewy (domyślnie 1);
//                         0 = zasady silnika Python: odłożone karty liczą się
//                         rozgrywającemu, niewybrany musik nie liczy się nikomu
//   * obserwacja (tryb q): hidden_disc N (ile ukrytych kart martwych to odłożone karty
//     rozgrywającego) i not_unused <karty> (karty, które nie mogą być w niewybranym musiku),
//   * tryb q: "output json" – wynik jako JSON dla backendu (zamiast tekstu printQ),
//   * selftest losuje także nowe flagi.
//
// Tryby:
//   tysiac_solver solve <plik|->            wartość dokładna stanu o pełnej informacji
//   tysiac_solver q     <plik|-> [wątki] [ttbits]   Q(a), max_a, loss dla obserwacji aktora
//   tysiac_solver bench [rozdań] [ttbits] [kontrakt] [wątki]   czasy na losowych rozdaniach 24 kart
//   tysiac_solver deal [ziarno] [kontrakt]  wypisuje rozdanie, wartość i optymalną grę lewa po lewie
//   tysiac_solver selftest [n]              porównanie alpha-beta+TT z czystym minimaksem
//
// Wartość = (wynik rozgrywającego) - (wynik obrońcy) w rozdaniu:
//   rozgrywający: +kontrakt jeśli punkty >= kontrakt, inaczej -kontrakt
//   obrońca:      jego punkty zaokrąglone do 10
// Kontrakt musi być wielokrotnością 10.
//
// ZAŁOŻENIA ZASAD (zmieniaj w legalMask / genMoves / applyMv / finalDelta):
//   * 24 karty: 9,W,D,K,10,A = 0,2,3,4,10,11 pkt; kolory S,C,D,H; meldunki 40,60,80,100
//   * dokładać do koloru, przebijać wyżej (mustbeat), bez koloru - dać atut jeśli jest
//   * meldunek (K lub D z parą w ręce) przy wyjściu, ustawia atu,
//     punkty liczone od razu
//   * 4 karty martwe (odrzut + niewybrany musik): ich punkty dostaje zdobywca OSTATNIEJ lewy
//   * stan: wychodzący do lewy ma led=-1; odpowiadający ma led=karta wyłożona przez rywala
// [integracja] przenośne nagłówki (clang/libc++ i MSVC nie mają <bits/stdc++.h>)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using namespace std;
typedef uint32_t u32;
typedef uint64_t u64;

// [integracja] zamienniki wbudowanych funkcji GCC/Clang (MSVC ich nie ma)
#if defined(_MSC_VER)
#include <intrin.h>
static inline int ctz32(u32 x) { unsigned long i; _BitScanForward(&i, x); return (int)i; }
static inline int popc32(u32 x) { return (int)__popcnt(x); }
#else
static inline int ctz32(u32 x) { return __builtin_ctz(x); }
static inline int popc32(u32 x) { return __builtin_popcount(x); }
#endif

static const int PTS[6]  = {0, 2, 3, 4, 10, 11};
static const int MELD[4] = {40, 60, 80, 100};
static const char* SUITS = "SCDH";
static const char* RANKS = "9WDKTA";            // T = dziesiątka
static const u32 SM[4]   = {0x3Fu, 0x3Fu << 6, 0x3Fu << 12, 0x3Fu << 18};
static const int INF = 30000;

static inline int sOf(int c) { return c / 6; }
static inline int rOf(int c) { return c % 6; }
static string cname(int c) { string s; s += SUITS[sOf(c)]; s += RANKS[rOf(c)]; return s; }
static int cid(const string& s) {
    if (s.size() != 2) throw runtime_error("zła karta: " + s);
    const char* p = strchr(SUITS, s[0]); const char* q = strchr(RANKS, s[1]);
    if (!p || !q) throw runtime_error("zła karta: " + s);
    return int(p - SUITS) * 6 + int(q - RANKS);
}

struct Cfg {
    int declarer = 0; int contract = 100; bool mustBeat = true;
    bool mustTrump = true;    // [integracja] bez koloru trzeba dać atut (oryginał: tylko gdy mustBeat)
    bool meldFirst = true;    // [integracja] meldunek dozwolony w 1. lewie
    bool deadToLast = true;   // [integracja] karty martwe dla zdobywcy ostatniej lewy (false = zasady silnika)
};
struct St  { u32 h[2]; int8_t turn, led, trump, tn; int16_t p[2]; int16_t dead; };   // dead = punkty 4 kart martwych (dla zdobywcy ostatniej lewy)
struct Mv  { int8_t c; int8_t meld; };

// ------------------------------------------------------------------ reguły --
static inline u32 legalMask(u32 hand, int led, int trump, bool mustBeat, bool mustTrump) {
    if (led < 0) return hand;
    int s = sOf(led);
    u32 same = hand & SM[s];
    if (same) {
        if (mustBeat) {
            u32 higher = same & ~((1u << (led + 1)) - 1u);
            if (higher) return higher;
        }
        return same;
    }
    if (trump >= 0 && mustTrump) { u32 tr = hand & SM[trump]; if (tr) return tr; }   // [integracja] osobna flaga
    return hand;
}

static inline bool isTerminal(const St& s) { return s.h[0] == 0 && s.h[1] == 0 && s.led < 0; }

static inline int finalDelta(const St& s, const Cfg& cf) {
    int D = s.p[cf.declarer], O = s.p[1 - cf.declarer];
    int ds = D >= cf.contract ? cf.contract : -cf.contract;
    return ds - (O + 5) / 10 * 10;
}

// generuje ruchy posortowane heurystycznie (najlepsze pierwsze); ttc = kod ruchu z TT lub -1
static int genMoves(const St& s, const Cfg& cf, Mv* out, int ttc) {
    u32 h = s.h[s.turn];
    u32 lm = legalMask(h, s.led, s.trump, cf.mustBeat, cf.mustTrump);
    int sc[40]; int n = 0;
    while (lm) {
        int c = ctz32(lm); lm &= lm - 1;
        int r = rOf(c), su = sOf(c), base;
        if (s.led < 0) base = PTS[r] + (su == s.trump ? 3 : 0);
        else {
            int led = s.led;
            bool fw = (su == sOf(led) && r > rOf(led)) || (su == s.trump && sOf(led) != s.trump);
            base = fw ? 200 - PTS[r] : -PTS[r] - (su == s.trump ? 20 : 0);
        }
        out[n] = {(int8_t)c, 0}; sc[n] = base + (c * 2 == ttc ? 10000 : 0); n++;
        if (s.led < 0 && (r == 2 || r == 3) && (cf.meldFirst || s.tn > 0)) {   // D lub K; [integracja] 1. lewa wg flagi
            int partner = (r == 2) ? c + 1 : c - 1;
            if (h >> partner & 1) {
                out[n] = {(int8_t)c, 1}; sc[n] = 1000 + MELD[su] + (c * 2 + 1 == ttc ? 10000 : 0); n++;
            }
        }
    }
    for (int i = 1; i < n; i++) {
        int j = i; Mv m = out[i]; int v = sc[i];
        while (j > 0 && sc[j - 1] < v) { out[j] = out[j - 1]; sc[j] = sc[j - 1]; j--; }
        out[j] = m; sc[j] = v;
    }
    return n;
}

static inline St applyMv(const St& s, Mv m) {
    St t = s; int pl = s.turn, c = m.c;
    t.h[pl] &= ~(1u << c);
    if (s.led < 0) {
        if (m.meld) { t.p[pl] += MELD[sOf(c)]; t.trump = (int8_t)sOf(c); }
        t.turn = 1 - pl; t.led = (int8_t)c; return t;
    }
    int led = s.led;
    bool fw = (sOf(c) == sOf(led) && rOf(c) > rOf(led)) || (sOf(c) == s.trump && sOf(led) != s.trump);
    int w = fw ? pl : 1 - pl;
    t.p[w] += PTS[rOf(c)] + PTS[rOf(led)];
    if (t.h[0] == 0 && t.h[1] == 0) t.p[w] += t.dead;       // ostatnia lewa: zdobywca bierze karty martwe
    t.turn = (int8_t)w; t.led = -1; t.tn = s.tn + 1; return t;
}

// ----------------------------------------------------------------- solver ---
struct Ent { u64 a, b; int16_t lo, hi; int8_t mv; int8_t pad; };

static inline u64 mix64(u64 x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x;
}

struct Solver {
    Cfg cf; vector<Ent> tt; u64 msk; u64 nodes = 0;
    Solver(Cfg c, int bits) : cf(c) {
        tt.assign(1ull << bits, Ent{0, 0, (int16_t)-INF, (int16_t)INF, -1, 0});
        msk = (1ull << bits) - 1;
    }
    inline void keyOf(const St& s, u64& a, u64& b) const {
        a = (u64)s.h[0] | ((u64)s.h[1] << 24) | ((u64)(s.trump + 1) << 48) | ((u64)s.turn << 51) | ((u64)(s.tn > 0) << 52);
        b = (u64)(uint16_t)s.p[0] | ((u64)(uint16_t)s.p[1] << 16) | ((u64)(uint16_t)s.dead << 32);
    }

    int ab(const St& s, int alpha, int beta) {
        if (isTerminal(s)) return finalDelta(s, cf);
        nodes++;
        Ent* e = nullptr; int ttc = -1; u64 ka = 0, kb = 0;
        if (s.led < 0) {
            keyOf(s, ka, kb);
            e = &tt[mix64(ka * 0x9E3779B97F4A7C15ULL ^ (kb + 0x1234567)) & msk];
            if (e->a == ka && e->b == kb) {
                if (e->lo >= beta) return e->lo;
                if (e->hi <= alpha) return e->hi;
                if (e->lo == e->hi) return e->lo;
                alpha = max(alpha, (int)e->lo); beta = min(beta, (int)e->hi);
                ttc = e->mv;
            }
        }
        Mv mv[32]; int n = genMoves(s, cf, mv, ttc);
        bool maxi = (s.turn == cf.declarer);
        int g, a0 = alpha, b0 = beta, best = -1;
        if (maxi) {
            g = -INF; int a = alpha;
            for (int i = 0; i < n; i++) {
                int v = ab(applyMv(s, mv[i]), a, beta);
                if (v > g) { g = v; best = mv[i].c * 2 + mv[i].meld; }
                if (g > a) a = g;
                if (g >= beta) break;
            }
        } else {
            g = INF; int b = beta;
            for (int i = 0; i < n; i++) {
                int v = ab(applyMv(s, mv[i]), alpha, b);
                if (v < g) { g = v; best = mv[i].c * 2 + mv[i].meld; }
                if (g < b) b = g;
                if (g <= alpha) break;
            }
        }
        if (e) {
            if (!(e->a == ka && e->b == kb)) { e->a = ka; e->b = kb; e->lo = -INF; e->hi = INF; e->mv = -1; }
            if (g <= a0) e->hi = min((int)e->hi, g);
            else if (g >= b0) e->lo = max((int)e->lo, g);
            else { e->lo = e->hi = (int16_t)g; }
            if (best >= 0) e->mv = (int8_t)best;
        }
        return g;
    }

    // MTD(f) z krokiem 10 (wszystkie wartości są wielokrotnościami 10)
    int mtdf(const St& s, int guess) {
        int g = guess / 10 * 10, lo = -INF, hi = INF;
        while (lo < hi) {
            int beta = (g == lo) ? g + 10 : g;
            g = ab(s, beta - 10, beta);
            if (g < beta) hi = g; else lo = g;
        }
        return g;
    }
};

// czysty minimax do testów poprawności
static int plain(const St& s, const Cfg& cf) {
    if (isTerminal(s)) return finalDelta(s, cf);
    Mv mv[32]; int n = genMoves(s, cf, mv, -1);
    bool maxi = (s.turn == cf.declarer);
    int g = maxi ? -INF : INF;
    for (int i = 0; i < n; i++) {
        int v = plain(applyMv(s, mv[i]), cf);
        g = maxi ? max(g, v) : min(g, v);
    }
    return g;
}

// ------------------------------------------- obserwacja, światy, Q(a) -------
struct Obs {
    int actor = 0; u32 hand = 0, unseen = 0, knownOpp = 0; int oppSize = 0;
    u32 voids = 0;                 // maska kolorów (bity 0..3), w których rywal nie ma kart
    int p[2] = {0, 0}; int hiddenDead = 0, deadKnown = 0;   // deadKnown = punkty martwych kart znanych aktorowi
    int led = -1, trump = -1, tn = 0;
    int hiddenDisc = 0;            // [integracja] ile ukrytych kart martwych to odłożone karty rozgrywającego
    u32 notUnused = 0;             // [integracja] karty, które nie mogą leżeć w niewybranym musiku
};

static bool sampleWorld(const Obs& o, const Cfg& cf, mt19937_64& rng, u32& oppMask, int* p, int& dead) {
    int pool[24], np = 0, cand[24], nc = 0;
    for (int c = 0; c < 24; c++)
        if ((o.unseen >> c & 1) && !(o.knownOpp >> c & 1)) {
            pool[np++] = c;
            if (!(o.voids >> sOf(c) & 1)) cand[nc++] = c;
        }
    int k = o.oppSize - popc32(o.knownOpp);
    if (k < 0 || np - k != o.hiddenDead || k > nc) return false;
    if (!cf.deadToLast && o.hiddenDisc > o.hiddenDead) return false;
    for (int attempt = 0; attempt < 400; attempt++) {
        int cc[24]; memcpy(cc, cand, sizeof(int) * nc);
        u32 chosen = 0;
        for (int i = 0; i < k; i++) {                              // częściowy Fisher-Yates
            int j = i + (int)(rng() % (u64)(nc - i)); swap(cc[i], cc[j]); chosen |= 1u << cc[i];
        }
        oppMask = o.knownOpp | chosen;
        p[0] = o.p[0]; p[1] = o.p[1];
        if (cf.deadToLast) {                                       // oryginał: martwe -> ostatnia lewa
            dead = o.deadKnown;                                    // + punkty ukrytych kart martwych
            for (int i = 0; i < np; i++) if (!(chosen >> pool[i] & 1)) dead += PTS[rOf(pool[i])];
            return true;
        }
        // [integracja] zasady silnika: spośród ukrytych kart martwych hiddenDisc to odłożone karty
        // rozgrywającego (jego punkty), reszta to niewybrany musik (0 pkt). Karty z not_unused
        // (odkryty musik) nie mogą trafić do niewybranego musiku.
        int forced[24], nfc = 0, freec[24], nfr = 0;
        for (int i = 0; i < np; i++) {
            int c = pool[i]; if (chosen >> c & 1) continue;
            if (o.notUnused >> c & 1) forced[nfc++] = c; else freec[nfr++] = c;
        }
        if (nfc > o.hiddenDisc) continue;                          // odrzucenie próbki
        int need = o.hiddenDisc - nfc;
        for (int i = 0; i < nfc; i++) p[cf.declarer] += PTS[rOf(forced[i])];
        for (int i = 0; i < need; i++) {
            int j = i + (int)(rng() % (u64)(nfr - i)); swap(freec[i], freec[j]); p[cf.declarer] += PTS[rOf(freec[i])];
        }
        dead = 0;
        return true;
    }
    return false;
}

struct QRes { vector<Mv> moves; vector<double> mean, se; double ms = 0; u64 nodes = 0; int n = 0; };

static QRes runQ(const Obs& o, const Cfg& cf, int N, int seed, int threads, int bits) {
    St probe{}; probe.h[o.actor] = o.hand; probe.turn = o.actor; probe.led = o.led;
    probe.trump = o.trump; probe.tn = o.tn;
    Mv mv[32]; int nm = genMoves(probe, cf, mv, -1);
    vector<Mv> moves(mv, mv + nm);
    sort(moves.begin(), moves.end(), [](Mv a, Mv b) { return a.c * 2 + a.meld < b.c * 2 + b.meld; });
    { mt19937_64 r(1); u32 om; int pp[2], dd; if (!sampleWorld(o, cf, r, om, pp, dd)) throw runtime_error("niespójna obserwacja (rozmiary ręki/voidy/martwe karty)"); }
    int sign = (o.actor == cf.declarer) ? 1 : -1;
    vector<vector<double>> S(threads, vector<double>(nm, 0)), Q(threads, vector<double>(nm, 0));
    vector<u64> nodes(threads, 0); atomic<int> next{0};
    auto t0 = chrono::steady_clock::now();
    auto work = [&](int tid) {
        Solver sv(cf, bits);
        for (;;) {
            int w = next++; if (w >= N) break;
            mt19937_64 rng((u64)seed * 1000003ULL + w);
            u32 om; int pp[2], dd;
            if (!sampleWorld(o, cf, rng, om, pp, dd)) {              // [integracja] rzadkie odrzucenie: nowe ziarno
                bool ok = false;
                for (int retry = 1; retry <= 50 && !ok; retry++) { mt19937_64 r2(rng() + retry); ok = sampleWorld(o, cf, r2, om, pp, dd); }
                if (!ok) { fprintf(stderr, "błąd: nie udało się wylosować świata zgodnego z obserwacją\n"); exit(2); }
            }
            St root{}; root.h[o.actor] = o.hand; root.h[1 - o.actor] = om;
            root.turn = o.actor; root.led = o.led; root.trump = o.trump; root.tn = o.tn;
            root.p[0] = pp[0]; root.p[1] = pp[1]; root.dead = (int16_t)dd;
            int guess = 0;
            for (int i = 0; i < nm; i++) {
                int v = sv.mtdf(applyMv(root, moves[i]), guess); guess = v;
                double x = sign * v; S[tid][i] += x; Q[tid][i] += x * x;
            }
        }
        nodes[tid] = sv.nodes;
    };
    vector<thread> th; for (int t = 0; t < threads; t++) th.emplace_back(work, t);
    for (auto& t : th) t.join();
    QRes r; r.moves = moves; r.n = N;
    for (int i = 0; i < nm; i++) {
        double s = 0, q = 0; for (int t = 0; t < threads; t++) { s += S[t][i]; q += Q[t][i]; }
        double m = s / N, var = N > 1 ? (q - N * m * m) / (N - 1) : 0;
        r.mean.push_back(m); r.se.push_back(sqrt(max(var, 0.0) / N));
    }
    for (u64 x : nodes) r.nodes += x;
    r.ms = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
    return r;
}

static string mvName(Mv m) { return cname(m.c) + (m.meld ? "+meldunek" : ""); }

static void printQ(const QRes& r, const string& model) {
    int bi = 0;
    for (size_t i = 0; i < r.moves.size(); i++) if (r.mean[i] > r.mean[bi]) bi = (int)i;
    for (size_t i = 0; i < r.moves.size(); i++)
        printf("Q %-14s %9.2f  se %6.2f\n", mvName(r.moves[i]).c_str(), r.mean[i], r.se[i]);
    printf("MAX_A_Q %.2f  best %s\n", r.mean[bi], mvName(r.moves[bi]).c_str());
    if (!model.empty()) {
        for (size_t i = 0; i < r.moves.size(); i++)
            if (mvName(r.moves[i]) == model || cname(r.moves[i].c) == model) {
                printf("MODEL %s  Q %.2f  LOSS %.2f\n", mvName(r.moves[i]).c_str(), r.mean[i], r.mean[bi] - r.mean[i]);
                goto done;
            }
        printf("MODEL %s: ruch nielegalny w tej sytuacji\n", model.c_str());
    }
done:
    printf("worlds %d  nodes %llu  time %.1f ms\n", r.n, (unsigned long long)r.nodes, r.ms);
}

// [integracja] wynik trybu q jako JSON (czytany przez backend Pythona)
static void printQJson(const QRes& r, const string& modelCard, bool modelMeld, bool hasModel) {
    int bi = 0;
    for (size_t i = 0; i < r.moves.size(); i++) if (r.mean[i] > r.mean[bi]) bi = (int)i;
    printf("{\"moves\": [");
    for (size_t i = 0; i < r.moves.size(); i++)
        printf("%s{\"card\": \"%s\", \"meld\": %s, \"q\": %.3f, \"se\": %.3f}", i ? ", " : "",
               cname(r.moves[i].c).c_str(), r.moves[i].meld ? "true" : "false", r.mean[i], r.se[i]);
    printf("], \"best\": {\"card\": \"%s\", \"meld\": %s, \"q\": %.3f}", cname(r.moves[bi].c).c_str(),
           r.moves[bi].meld ? "true" : "false", r.mean[bi]);
    int mi = -1;
    if (hasModel)
        for (size_t i = 0; i < r.moves.size(); i++)
            if (cname(r.moves[i].c) == modelCard && (r.moves[i].meld != 0) == modelMeld) { mi = (int)i; break; }
    if (mi >= 0)
        printf(", \"model\": {\"card\": \"%s\", \"meld\": %s, \"q\": %.3f, \"loss\": %.3f}",
               modelCard.c_str(), modelMeld ? "true" : "false", r.mean[mi], r.mean[bi] - r.mean[mi]);
    else printf(", \"model\": null");
    printf(", \"model_legal\": %s, \"worlds\": %d, \"nodes\": %llu, \"ms\": %.1f}\n",
           (!hasModel || mi >= 0) ? "true" : "false", r.n, (unsigned long long)r.nodes, r.ms);
}

// ------------------------------------------------------------- wejście ------
static map<string, vector<string>> readKV(istream& in) {
    map<string, vector<string>> kv; string line;
    while (getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream ss(line); string k, t; ss >> k; vector<string> v;
        while (ss >> t) v.push_back(t);
        kv[k] = v;
    }
    return kv;
}
static u32 pmask(const vector<string>& v) { u32 m = 0; for (auto& t : v) if (t != "-") m |= 1u << cid(t); return m; }
static int pcardOrNone(const map<string, vector<string>>& kv, const string& k) {
    auto it = kv.find(k); if (it == kv.end() || it->second.empty() || it->second[0] == "-") return -1;
    return cid(it->second[0]);
}
static int ptrump(const map<string, vector<string>>& kv) {
    auto it = kv.find("trump"); if (it == kv.end() || it->second.empty() || it->second[0] == "-") return -1;
    const char* p = strchr(SUITS, it->second[0][0]); if (!p) throw runtime_error("zły atut"); return int(p - SUITS);
}
static int pint(const map<string, vector<string>>& kv, const string& k, int d) {
    auto it = kv.find(k); return (it == kv.end() || it->second.empty()) ? d : stoi(it->second[0]);
}
static Cfg readCfg(const map<string, vector<string>>& kv) {
    Cfg cf; cf.declarer = pint(kv, "declarer", 0); cf.contract = pint(kv, "contract", 100);
    cf.mustBeat = pint(kv, "mustbeat", 1) != 0;
    cf.mustTrump = pint(kv, "musttrump", cf.mustBeat ? 1 : 0) != 0;   // [integracja]
    cf.meldFirst = pint(kv, "meldfirst", 1) != 0;                       // [integracja]
    cf.deadToLast = pint(kv, "dead_to_last", 1) != 0;                   // [integracja]
    if (cf.contract % 10) throw runtime_error("kontrakt musi być wielokrotnością 10");
    return cf;
}

// ------------------------------------------------------------------ tryby ---
// ---------------------------------------------------- podgląd rozdania -------
static string handStr(u32 m) {
    string s;
    for (int su = 0; su < 4; su++) {
        s += SUITS[su]; s += ':'; bool any = false;
        for (int r = 5; r >= 0; r--) if (m >> (su * 6 + r) & 1) { s += ' '; s += RANKS[r]; any = true; }
        if (!any) s += " -";
        s += "   ";
    }
    return s;
}

static void printDeal(const St& s, const Cfg& cf) {
    printf("--- ROZDANIE ---\n");
    printf("rozgrywający: gracz %d   kontrakt: %d   atu: %s   wychodzi: gracz %d\n", cf.declarer, cf.contract,
           s.trump < 0 ? "-" : string(1, SUITS[s.trump]).c_str(), s.turn);
    printf("gracz 0: %s\n", handStr(s.h[0]).c_str());
    printf("gracz 1: %s\n", handStr(s.h[1]).c_str());
    u32 dead = ~(s.h[0] | s.h[1]) & 0xFFFFFFu;
    printf("poza rękami: %s\n", handStr(dead).c_str());
    printf("punkty zdobyte dotąd: gracz 0 = %d, gracz 1 = %d\n", s.p[0], s.p[1]);
    printf("karty martwe: %d pkt -> dla zdobywcy OSTATNIEJ lewy\n", s.dead);
}

// optymalny przebieg: w każdym węźle wybiera ruch, którego wartość dokładna = wartość węzła
static void printPV(Solver& sv, St s) {
    const Cfg& cf = sv.cf;
    int v = sv.mtdf(s, 0);
    printf("--- OPTYMALNA GRA (double-dummy), wartość = %d ---\n", v);
    int trick = 1; Mv lead{-1, 0}; int leader = -1, lastWinner = -1; int deadPts = s.dead;
    while (!isTerminal(s)) {
        Mv mv[32]; int n = genMoves(s, cf, mv, -1);
        bool maxi = (s.turn == cf.declarer); Mv pick = mv[0];
        for (int i = 0; i < n; i++) {
            int cv = sv.mtdf(applyMv(s, mv[i]), v);
            if (cv == v) { pick = mv[i]; break; }
        }
        St t = applyMv(s, pick);
        if (s.led < 0) { lead = pick; leader = s.turn; }
        else {
            int pts = (t.p[0] + t.p[1]) - (s.p[0] + s.p[1]) - (isTerminal(t) ? deadPts : 0);
            printf("lewa %2d: gracz %d %-12s  gracz %d %-4s -> bierze gracz %d (+%2d)%s   stan %d:%d\n", trick, leader,
                   mvName(lead).c_str(), s.turn, cname(pick.c).c_str(), t.turn, pts,
                   lead.meld ? ("  [meldunek " + to_string(MELD[sOf(lead.c)]) + ", atu " + SUITS[sOf(lead.c)] + "]").c_str() : "",
                   t.p[0], t.p[1]);
            trick++; lastWinner = t.turn;
        }
        (void)maxi; s = t;
    }
    printf("ostatnią lewę wziął gracz %d -> zgarnia karty martwe (+%d pkt)\n", lastWinner, deadPts);
    printf("wynik końcowy: gracz 0 = %d pkt, gracz 1 = %d pkt, rozgrywający %s\n", s.p[0], s.p[1],
           s.p[cf.declarer] >= cf.contract ? "ZROBIŁ kontrakt" : "PADŁ");
}

static void makeDeal(int seed, St& s) {
    mt19937_64 rng(seed); int deck[24]; iota(deck, deck + 24, 0); shuffle(deck, deck + 24, rng);
    s = St{}; for (int i = 0; i < 10; i++) { s.h[0] |= 1u << deck[i]; s.h[1] |= 1u << deck[10 + i]; }
    s.turn = 0; s.led = -1; s.trump = -1; s.tn = 0;
    s.p[0] = s.p[1] = 0;
    s.dead = PTS[rOf(deck[20])] + PTS[rOf(deck[21])] + PTS[rOf(deck[22])] + PTS[rOf(deck[23])];  // odrzut + niewybrany musik
}

static int modeDeal(int seed, int contract) {
    Cfg cf; cf.contract = contract; St s; makeDeal(seed, s);
    printDeal(s, cf);
    Solver sv(cf, 22); printPV(sv, s);
    return 0;
}

static int modeSolve(istream& in) {
    auto kv = readKV(in); Cfg cf = readCfg(kv);
    St s{}; s.h[0] = pmask(kv["h0"]); s.h[1] = pmask(kv["h1"]);
    s.turn = pint(kv, "turn", 0); s.led = pcardOrNone(kv, "led"); s.trump = ptrump(kv); s.tn = pint(kv, "tn", 0);
    auto pv = kv["pts"]; s.p[0] = pv.size() > 0 ? stoi(pv[0]) : 0; s.p[1] = pv.size() > 1 ? stoi(pv[1]) : 0;
    if (kv.count("dead")) { int d = 0; for (auto& t : kv["dead"]) if (t != "-") d += PTS[rOf(cid(t))]; s.dead = (int16_t)d; }
    else s.dead = (int16_t)pint(kv, "dead_pts", 0);
    printDeal(s, cf);
    Solver sv(cf, 22); auto t0 = chrono::steady_clock::now();
    int v = sv.mtdf(s, 0);
    double ms = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
    printf("VALUE %d  (rozgrywający - obrońca)\n", v);
    Mv mv[32]; int n = genMoves(s, cf, mv, -1); int sign = s.turn == cf.declarer ? 1 : -1;
    for (int i = 0; i < n; i++)
        printf("MOVE %-14s %d\n", mvName(mv[i]).c_str(), sign * sv.mtdf(applyMv(s, mv[i]), v));
    printf("nodes %llu  time %.1f ms\n", (unsigned long long)sv.nodes, ms);
    if (!isTerminal(s) && s.led < 0) printPV(sv, s);
    return 0;
}

static int modeQ(istream& in, int threads, int bits) {
    auto kv = readKV(in); Cfg cf = readCfg(kv); Obs o;
    o.actor = pint(kv, "actor", 0); o.hand = pmask(kv["hand"]); o.unseen = pmask(kv["unseen"]);
    o.knownOpp = pmask(kv["known_opp"]); o.oppSize = pint(kv, "opp_size", popc32(o.hand));
    for (auto& t : kv["voids"]) { const char* p = strchr(SUITS, t[0]); if (p && t != "-") o.voids |= 1u << (p - SUITS); }
    auto pv = kv["pts"]; o.p[0] = pv.size() > 0 ? stoi(pv[0]) : 0; o.p[1] = pv.size() > 1 ? stoi(pv[1]) : 0;
    o.hiddenDead = pint(kv, "hidden_dead", 0); o.deadKnown = pint(kv, "dead_pts", 0);
    o.led = pcardOrNone(kv, "led"); o.trump = ptrump(kv); o.tn = pint(kv, "tn", 0);
    o.hiddenDisc = pint(kv, "hidden_disc", 0); o.notUnused = pmask(kv["not_unused"]);   // [integracja]
    int N = pint(kv, "samples", 100), seed = pint(kv, "seed", 1);
    string model = kv.count("model") && !kv["model"].empty() ? kv["model"][0] : "";
    bool modelMeld = kv.count("model") && kv["model"].size() > 1;
    QRes res = runQ(o, cf, N, seed, threads, bits);
    if (kv.count("output") && !kv["output"].empty() && kv["output"][0] == "json") {   // [integracja]
        printQJson(res, model, modelMeld, !model.empty());
        return 0;
    }
    if (modelMeld) model += "+meldunek";
    printQ(res, model);
    return 0;
}

static int modeBench(int ndeals, int bits, int contract, int threads) {
    Cfg cf; cf.contract = contract;
    double totMs = 0; u64 totNodes = 0; double maxMs = 0;
    printf("== Pełne rozdanie 24 kart, rozgrywający wychodzi, pełna informacja (tn=0) ==\n");
    for (int d = 0; d < ndeals; d++) {
        St s; makeDeal(1234 + d, s);
        if (ndeals <= 5) printDeal(s, cf);
        Solver sv(cf, bits); auto t0 = chrono::steady_clock::now();
        int v = sv.mtdf(s, 0);
        double ms = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
        printf("rozdanie %2d (ziarno %d): wartość %5d  węzły %10llu  czas %8.1f ms\n", d, 1234 + d, v, (unsigned long long)sv.nodes, ms);
        if (ndeals <= 5 && d == 0) printPV(sv, s);
        totMs += ms; totNodes += sv.nodes; maxMs = max(maxMs, ms);
    }
    printf("średnio %.1f ms / rozdanie, max %.1f ms, %.0f tys. węzłów/s\n", totMs / ndeals, maxMs, totNodes / totMs);

    printf("\n== Widok obrońcy po 1. wyjściu (13 nieznanych kart: 9 rozgrywającego + 4 martwe), 20 światów ==\n");
    for (int d = 0; d < min(ndeals, 3); d++) {
        mt19937_64 rng(1234 + d); int deck[24]; iota(deck, deck + 24, 0); shuffle(deck, deck + 24, rng);
        u32 h0 = 0, h1 = 0; for (int i = 0; i < 10; i++) { h0 |= 1u << deck[i]; h1 |= 1u << deck[10 + i]; }
        int led = deck[0];                                         // losowe pierwsze wyjście rozgrywającego
        Obs o; o.actor = 1; o.hand = h1; o.unseen = (h0 & ~(1u << led)) | (1u << deck[20]) | (1u << deck[21]) | (1u << deck[22]) | (1u << deck[23]);
        o.oppSize = 9; o.hiddenDead = 4; o.led = led; o.trump = -1; o.tn = 0;
        QRes r = runQ(o, cf, 20, 1, threads, bits);
        int bi = 0; for (size_t i = 0; i < r.moves.size(); i++) if (r.mean[i] > r.mean[bi]) bi = (int)i;
        printf("rozdanie %d: %zu ruchów, max_a Q=%.1f (%s), czas %.0f ms (%d wątków, %llu węzłów)\n",
               d, r.moves.size(), r.mean[bi], mvName(r.moves[bi]).c_str(), r.ms, threads, (unsigned long long)r.nodes);
    }
    return 0;
}

static int modeSelftest(int n) {
    mt19937_64 rng(777); int bad = 0;
    for (int it = 0; it < n; it++) {
        Cfg cf; cf.declarer = rng() % 2; cf.contract = 100 + 10 * (rng() % 6); cf.mustBeat = rng() % 4 != 0;
        cf.mustTrump = rng() % 4 != 0; cf.meldFirst = rng() % 2; cf.deadToLast = rng() % 2;   // [integracja]
        int deck[24]; iota(deck, deck + 24, 0); shuffle(deck, deck + 24, rng);
        int k = 2 + rng() % 3;                                    // 2..4 kart na rękę
        St s{}; s.turn = rng() % 2; s.led = -1; s.trump = (int)(rng() % 5) - 1; s.tn = rng() % 2;
        for (int i = 0; i < k; i++) { s.h[0] |= 1u << deck[i]; s.h[1] |= 1u << deck[k + i]; }
        if (rng() % 2) { int lc = deck[2 * k]; s.led = lc; s.h[1 - s.turn] &= ~(1u << deck[(1 - s.turn) * k]); }
        // gdy led != -1: wychodził rywal 'turn', więc musi mieć o jedną kartę mniej
        if (s.led >= 0) { s.h[1 - s.turn] = 0; for (int i = 0; i < k - 1; i++) s.h[1 - s.turn] |= 1u << deck[(1 - s.turn) * k + i]; if (s.h[1 - s.turn] == 0) { s.led = -1; s.h[1 - s.turn] |= 1u << deck[(1 - s.turn) * k]; } }
        s.p[0] = 10 * (rng() % 15); s.p[1] = 10 * (rng() % 10) + 5 * (rng() % 2);
        s.dead = cf.deadToLast ? (int16_t)(rng() % 30) : 0;   // [integracja]
        int a = plain(s, cf); Solver sv(cf, 16); int b = sv.mtdf(s, (int)(rng() % 5) * 10 - 20);
        int c = sv.ab(s, -INF, INF);
        if (a != b || a != c) { if (bad++ < 5) printf("BŁĄD: plain=%d mtdf=%d ab=%d\n", a, b, c); }
    }
    printf("selftest: %d pozycji, błędów %d\n", n, bad);
    return bad != 0;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) { fprintf(stderr, "użycie: solve|q|bench|selftest ...\n"); return 1; }
        string m = argv[1];
        int hw = max(1u, thread::hardware_concurrency());
        if (m == "solve" || m == "q") {
            if (argc < 3) { fprintf(stderr, "podaj plik lub -\n"); return 1; }
            ifstream f; istream* in = &cin;
            if (string(argv[2]) != "-") { f.open(argv[2]); if (!f) { fprintf(stderr, "brak pliku\n"); return 1; } in = &f; }
            if (m == "solve") return modeSolve(*in);
            return modeQ(*in, argc > 3 ? atoi(argv[3]) : hw, argc > 4 ? atoi(argv[4]) : 21);
        }
        if (m == "bench") return modeBench(argc > 2 ? atoi(argv[2]) : 5, argc > 3 ? atoi(argv[3]) : 22,
                                           argc > 4 ? atoi(argv[4]) : 100, argc > 5 ? atoi(argv[5]) : hw);
        if (m == "deal") return modeDeal(argc > 2 ? atoi(argv[2]) : 1234, argc > 3 ? atoi(argv[3]) : 100);
        if (m == "selftest") return modeSelftest(argc > 2 ? atoi(argv[2]) : 3000);
    } catch (exception& e) { fprintf(stderr, "błąd: %s\n", e.what()); return 2; }
    return 1;
}
