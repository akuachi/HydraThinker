import os
import re
import json
import time
import random
import logging
import threading
import http.server
import socketserver
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from ollama import chat, embed

# =========================
# PERCORSI ASSOLUTI
# =========================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
EVOL_DIR   = os.path.join(OUTPUT_DIR, "evolution")
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
ACCEPT_DIR = os.path.join(OUTPUT_DIR, "accepted")

MEMORY_FILE         = os.path.join(BASE_DIR, "memory.json")
BRIEF_FILE          = os.path.join(BASE_DIR, "brief.md")
DASHBOARD_DATA_FILE = os.path.join(OUTPUT_DIR, "dashboard_data.json")

for _d in [EVOL_DIR, LOG_DIR, ACCEPT_DIR]:
    os.makedirs(_d, exist_ok=True)

# =========================
# CONFIG
# =========================

MODEL       = "qwen3:14b"
EMBED_MODEL = "all-minilm"

POP_SIZE   = 6
SURVIVORS  = 3
SLEEP      = 10           # pausa tra generazioni
MIN_SCORE  = 35           # la scala ora è 5-95; idee mediocri iniziano a 30-40
HOF_SCORE  = 85

# ── Tipi di prodotto (formato/struttura tecnica) ─────────────────────────────
IDEA_TYPES = [
    "saas",           # webapp con abbonamento
    "bot",            # Telegram / Discord / Slack bot
    "tool",           # strumento CLI o web utility
    "marketplace",    # piattaforma domanda/offerta
    "api",            # microservizio / SDK
    "browser-ext",    # estensione Chrome/Firefox
    "no-code",        # builder visuale / template shop
    "community",      # forum / community di nicchia
    "data-product",   # dataset, report, newsletter a pagamento
    "game",           # mini-gioco web o mobile-web
]

# ── Domini tematici (argomento dell'idea) ────────────────────────────────────
# Il sistema cicla su questi domini per garantire variety massima.
# La combinazione TIPO + DOMINIO è il vero driver di diversità.
IDEA_DOMAINS = [
    "salute e benessere personale",
    "educazione e apprendimento pratico",
    "finanza personale e risparmio",
    "produttività artigiani e freelance",
    "creatività: musica, arte, scrittura",
    "ambiente e sostenibilità quotidiana",
    "sport, fitness e nutrizione",
    "viaggi, turismo locale e esperienze",
    "cibo, cucina e ristorazione locale",
    "gaming e comunità di giocatori",
    "burocrazia e documenti semplificati",
    "HR, recruiting e carriera",
    "real estate e affitti",
    "e-commerce verticale di nicchia",
    "genitori, famiglie e bambini",
    "animali domestici",
    "artigianato, maker e DIY",
    "musica e podcast indipendenti",
    "volontariato e terzo settore",
    "developer tools",
]

# Varietà di idee — il sistema CICLA su queste per garantire massima diversità
# Include sia idee AI-based che classiche, l'importante è alternare
IDEA_VARIETIES = [
    "SaaS B2B classico (CRUD, dashboard, report, gestione — AI opzionale)",
    "tool AI per nicchia specifica (non AI generico, ma applicato a un settore)",
    "bot utility (Telegram/Discord — logica tradizionale o AI leggera)",
    "community / marketplace verticale (domanda/offerta, matching umano, recensioni)",
    "automazione tradizionale (webhook, schedule, notifiche, integrazioni API)",
    "strumento per creator/developer/artigiani (template, generatori, calcolatori)",
    "estensione browser (utility, produttività, integrazione)",
    "piattaforma contenuti / educational (corsi, template, risorse)",
    "mini-gioco web o mobile-web (single player o multiplayer leggero)",
    "directory / catalogo verificato (elenchi, ricerca, filtri, profili)",
    "e-commerce di nicchia (vetrina, carrello, pagamenti, dropshipping leggero)",
    "automazione AI applicata (non AI generalista, ma risolve un problema preciso)",
]

MAX_MEMORY_CONCEPTS  = 300
MAX_RETRY            = 2
MAX_WORKERS          = 4
LLM_TIMEOUT          = 300
EMBED_SIM_THRESHOLD  = 0.85    # soglia cosine similarity per dedup (0.0-1.0)
ADAPTIVE_WINDOW      = 7       # ultime N generazioni per calibrare MIN_SCORE
NOVELTY_VARIETY_BONUS = 3      # punti extra per varietà mai esplorata
NOVELTY_DOMAIN_BONUS  = 2      # punti extra per dominio inesplorato

# Tipi di mutazione — il sistema CICLA su questi per varietà evolutiva
MUTATION_TYPES = [
    "cambia il target di riferimento (stessa soluzione, utenti/settore diverso)",
    "cambia il modello di business/pricing (es. subscription -> usage / marketplace -> commissioni)",
    "applica la stessa idea a un dominio/settore completamente diverso",
]

# Soglia per enrichment (solo idee migliori ricevono analisi approfondita)
ENRICH_SCORE_THRESHOLD = 70

# Keyword comuni ignorate nella deduplication
COMMON_KEYWORDS = {
    "ai", "tool", "app", "api", "saas", "bot", "user", "data",
    "platform", "service", "online", "web", "cloud", "auto",
    "strumento", "utente", "dati", "piattaforma", "servizio",
    "sistema", "gestione", "management", "solution", "soluzione",
    "assistant", "assistente", "intelligenza", "machine", "learning",
    "tempo", "persona", "personale", "locale", "digitale", "sociale",
    "rapido", "semplice", "facile", "veloce", "nuovo", "nuova",
    "casa", "lavoro", "vita", "giorno", "mese", "anno",
    "creare", "trovare", "avere", "fare", "essere", "potere",
    "primo", "prima", "ogni", "altro", "altra", "stessa", "stesso",
    "privacy", "sicurezza", "gratuito", "premium", "account",
    "notifica", "aggiornamento", "download", "upload", "login",
    "accesso", "ricerca", "filtro", "categoria", "profilo",
    "messaggio", "chat", "email", "telefono", "mobile", "desktop",
}

DASHBOARD_PORT = 8080

EMBEDDING_CACHE_FILE = os.path.join(OUTPUT_DIR, "embeddings_cache.json")

# Limite di tempo per ogni singolo candidato (in secondi)
# Se un candidato impiega più di questo, viene saltato
CANDIDATE_TIMEOUT = 600

# Adattamento dinamico della popolazione
MIN_POP_SIZE = 3
MAX_POP_SIZE = 12
ADAPTIVE_LOOKBACK = 5   # quante generazioni guardare per il tasso di passaggio

# =========================
# LOGGING (stdout + file)
# =========================

_log_file = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ],
)

def log(msg: str):
    logging.info(msg)

# =========================
# BRIEF
# =========================

def load_brief() -> str:
    try:
        with open(BRIEF_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Genera idee originali e realizzabili da un singolo developer."

BRIEF = load_brief()

# =========================
# MEMORY (con migrazione legacy)
# =========================

def _migrate_legacy(data: dict) -> dict:
    migrated = False
    if "concepts" not in data:
        old_ideas = data.get("ideas", [])
        concepts: list[str] = []
        for idea in old_ideas:
            if isinstance(idea, str):
                words = re.findall(r"[a-zA-ZÀ-ÿ]{4,}", idea)
                concepts.extend(w.lower() for w in words[:20])
        data["concepts"] = list(dict.fromkeys(concepts))[:MAX_MEMORY_CONCEPTS]
        migrated = True
    if "types_history" not in data:
        data["types_history"] = []
        migrated = True
    for old_key in ("ideas", "tags_used"):
        if old_key in data:
            del data[old_key]
            migrated = True
    if migrated:
        log("memory.json migrato dal formato legacy")
    return data


def load_memory() -> dict:
    defaults = {
        "concepts":           [],
        "types_history":      [],
        "domains_history":    [],
        "hall_of_fame":       [],
        "recent_ideas":       [],      # testi completi delle ultime idee accettate
        "varieties_history":  [],      # varietà usate di recente
    }
    if not os.path.exists(MEMORY_FILE):
        return defaults
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        log("memory.json corrotto — inizializzato da zero")
        return defaults
    data = _migrate_legacy(data)
    for key, value in defaults.items():
        if key not in data:
            data[key] = value
    return data


def save_memory(m: dict):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

# =========================
# LLM CALL
# Implementazione semplice e robusta:
# - Nessun thread-within-thread (evita ghost threads e overhead)
# - Timeout gestito direttamente su una chiamata bloccante tramite
#   un singolo thread daemon con join(timeout)
# - think=False disabilita il "thinking mode" di qwen3, riducendo
#   drasticamente i tempi da 3-5 min a 15-40 sec per chiamata
# =========================

def ask(system: str, user: str) -> str:
    """Chiama Ollama con retry. think=False disabilita il thinking mode."""
    for attempt in range(MAX_RETRY):
        result: list[str] = []
        exc:    list[Exception] = []

        def _call():
            try:
                resp = chat(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    think=False,          # disabilita thinking mode (qwen3/deepseek)
                    options={
                        "num_ctx":    4096,   # contesto ridotto = risposta più veloce
                        "temperature": 0.8,
                    },
                )
                result.append(resp["message"]["content"])
            except Exception as e:
                exc.append(e)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=LLM_TIMEOUT)

        if t.is_alive():
            log(f"LLM timeout ({LLM_TIMEOUT}s) al tentativo {attempt + 1}/{MAX_RETRY}")
            # Il thread è daemon: verrà terminato quando il processo finisce.
            # Non creiamo un nuovo thread per la stessa chiamata, aspettiamo.
            if attempt < MAX_RETRY - 1:
                time.sleep(3)
            continue

        if exc:
            log(f"LLM error (tentativo {attempt + 1}/{MAX_RETRY}): {exc[0]}")
            if attempt < MAX_RETRY - 1:
                time.sleep(3)
            continue

        if result:
            return result[0]

    return ""

# =========================
# IDEA DIVERSITY ENGINE
# =========================

def get_next_idea_type(memory: dict) -> str:
    history  = memory.get("types_history", [])
    recent   = set(history[-3:])
    available = [t for t in IDEA_TYPES if t not in recent]
    return random.choice(available or IDEA_TYPES)


def get_next_domain(memory: dict) -> str:
    history  = memory.get("domains_history", [])
    recent   = set(history[-5:])
    available = [d for d in IDEA_DOMAINS if d not in recent]
    return random.choice(available or IDEA_DOMAINS)


def get_next_variety(memory: dict) -> str:
    """Sceglie una varietà di idea evitando le ultime 3 usate."""
    history = memory.get("varieties_history", [])
    recent  = set(history[-3:])
    available = [v for v in IDEA_VARIETIES if v not in recent]
    return random.choice(available or IDEA_VARIETIES)

# =========================
# GENERAZIONE IDEA
# =========================

def generate_idea(memory: dict, gen: int = 0) -> tuple[str, str, str, str]:
    idea_type  = get_next_idea_type(memory)
    domain     = get_next_domain(memory)
    variety    = get_next_variety(memory)

    retry_hint = memory.get("_retry_hint")

    last_concepts = ", ".join(memory["concepts"][-25:])
    last_types    = ", ".join(memory["types_history"][-6:])
    last_domains  = ", ".join(memory.get("domains_history", [])[-6:])
    recent_ideas  = memory.get("recent_ideas", [])[-3:]

    idea = ask(
        "Sei un generatore di idee startup originali, concrete e molto varie. "
        "EVITA di ripeterti: se un'idea è già stata generata, cambia completamente "
        "approccio, target e meccanica.",
        f"""BRIEF (usalo come ispirazione):
{BRIEF}

VINCOLI:
- TIPO PRODOTTO: {idea_type}
- DOMINIO TEMATICO: {domain}
- VARIETÀ: {variety}
{'' if not retry_hint else 'ATTENZIONE: ' + retry_hint}

COSA EVITARE (per non ripeterti):
- Idee troppo simili a quelle già generate (vedi sotto)
- Concetti già usati in passato
- Target generici ("persone che vogliono essere più produttive")
- Soluzioni enterprise o che richiedono hardware

ESEMPI DI IDEE AD ALTO POTENZIALE (ispirati ma non copiare):
- Piattaforma verticale per una nicchia SPECIFICA con monetizzazione chiara e competitor assenti
- Tool che risolve un problema frustrante per un target ben definito (non "tutti")
- Marketplace iper-locale che collega domanda/offerta in un settore con pochi intermediari
- Bot/servizio che automatizza un processo manuale doloroso per professionisti di nicchia
- Directory verificata con recensioni autentiche in un settore dove non esistono
- SaaS B2B per un settore artigianale specifico (non "ristoranti" ma "pasticcerie artigianali")
- Piattaforma educational che risolve un PROBLEMA SPECIFICO per un target preciso
- Tool AI che applica ML a un DATASET DI NICCHIA (non AI generalista)
- Mini-gioco sociale con meccanica virale e monetizzazione non invasiva
- Community verticale con curation umana in un settore frammentato

TIPI RECENTI (non ripetere): {last_types}
DOMINI RECENTI (non ripetere): {last_domains}
IDEE RECENTI (devi essere DIVERSO): {', '.join(r[:100] for r in recent_ideas)}
CONCETTI GIÀ USATI (evita): {last_concepts}

REGOLE:
- L'idea DEVE stare nel dominio "{domain}"
- L'idea DEVE avere formato "{idea_type}"
- Sii SPECIFICO: target di nicchia preciso
- Realizzabile da 1 developer in max 2 settimane (MVP)
- Niente enterprise, niente hardware fisico
- NON ripetere concept già usati
- Sii CREATIVO — mescola domini, usa analogie, pensa laterale

OUTPUT (conciso, max 200 parole):
Nome: ...
Problema: ...
Soluzione: ...
Target: ...
Monetizzazione: ..."""
    )
    return idea, idea_type, domain, variety

# =========================
# EVALUATE + CONCETTI (singola chiamata LLM)
# =========================

def _parse_score_json(res: str) -> tuple[int, str, list[str]]:
    """Parsing robusto di risposta LLM in formato JSON."""
    try:
        s = res.find("{"); e = res.rfind("}") + 1
        if s == -1 or e == 0:
            raise ValueError("no JSON")
        j        = json.loads(res[s:e])
        score    = max(0, min(100, int(j.get("score", 0))))
        weakness = j.get("weakness", "nessuna")
        concepts = [str(c) for c in j.get("concepts", [])]
        return score, weakness, concepts
    except Exception:
        return 0, "errore parsing", []


def evaluate_market(idea: str) -> tuple[int, str, list[str]]:
    """Valutazione focalizzata su MERCATO: target, monetizzazione, competitor."""
    if not idea:
        return 0, "idea vuota", []
    res = ask(
        "Sei un analista di mercato con 20 anni di esperienza. "
        "USA TUTTA LA SCALA 0-100. Sii selettivo: 1 idea su 10 merita oltre 70. "
        "La maggior parte delle idee (anche buone) sta tra 30 e 55. "
        "Solo idee con nicchia chirurgica, monetizzazione immediata e "
        "mercato reale non saturo meritano 70+.",
        f"""IDEA:
{idea}

Valuta SOLO questi criteri:
1. TARGET: nicchia specifica e reale (alto) vs target generico (basso)
2. MONETIZZAZIONE: modello concreto e rapido (alto) vs vago (basso)
3. MERCATO: spazio reale non saturo (alto) vs mercato affollato (basso)
4. COMPETITOR: vantaggio chiaro (alto) vs commodity (basso)

ANCORE: 10=inesistente, 25=debole, 50=discreto, 75=buono, 95=eccellente

Rispondi SOLO con JSON:
{{"score": <0-100>, "weakness": "<max 15 parole>", "concepts": ["<kw1>","<kw2>","<kw3>"]}}"""
    )
    return _parse_score_json(res)


def evaluate_technical(idea: str) -> tuple[int, str, list[str]]:
    """Valutazione focalizzata su FATTIBILITÀ: originalità, complessità, stack."""
    if not idea:
        return 0, "idea vuota", []
    res = ask(
        "Sei un CTO con esperienza in startup. "
        "USA TUTTA LA SCALA 0-100. Sii selettivo: 1 idea su 10 merita oltre 70. "
        "La maggior parte delle idee sta tra 30 e 55. Solo idee veramente "
        "originali, semplici da costruire e senza rischi esterni meritano 70+.",
        f"""IDEA:
{idea}

Valuta SOLO questi criteri:
1. ORIGINALITÀ: genuinamente nuova (alto) vs ennesima variante (basso)
2. FATTIBILITÀ: realizzabile da 1 dev in 2 settimane (alto) vs complesso (basso)
3. COMPLESSITÀ: stack semplice (alto) vs infrastruttura complessa (basso)
4. RISCHIO: autonomo (alto) vs dipende da API/dati esterni (basso)

ANCORE: 10=banale e complessa, 25=già vista, 50=onesta, 75=originale e realizzabile, 95=brillante e semplicissima

Rispondi SOLO con JSON:
{{"score": <0-100>, "weakness": "<max 15 parole>", "concepts": ["<kw1>","<kw2>","<kw3>"]}}"""
    )
    return _parse_score_json(res)


def _widen_score(raw: float) -> float:
    """Mappa [35, 80] → [10, 95] — idee eccellenti (raw 75+) raggiungono 85+."""
    lo, hi = 35.0, 80.0
    target_lo, target_hi = 10.0, 95.0
    if raw <= lo:
        return max(0.0, target_lo * (raw / lo))
    if raw >= hi:
        return min(100.0, target_hi + (100 - target_hi) * (raw - hi) / (100 - hi))
    return target_lo + (raw - lo) * (target_hi - target_lo) / (hi - lo)


def evaluate_composite(idea: str) -> tuple[float, str, list[str]]:
    """Valutazione composita: media di market + technical.
    
    Due prospettive diverse sull'idea, poi mediate. Riduce il bias
    di un singolo prompt LLM.
    """
    m_score, m_weak, m_concepts = evaluate_market(idea)
    t_score, t_weak, t_concepts = evaluate_technical(idea)

    raw_score = (m_score + t_score) / 2.0
    score = _widen_score(raw_score)
    weakness = f"Mercato: {m_weak} | Tecnico: {t_weak}"

    # Unisce concetti da entrambe le valutazioni
    all_concepts = list(dict.fromkeys(m_concepts + t_concepts))

    return score, weakness, all_concepts

# =========================
# REALITY CHECK + COMPETITION (singola chiamata LLM)
# Fusione di reality_check + competition_score → -1 chiamata LLM per idea
# =========================

def reality_and_competition(idea: str) -> tuple[str, int]:
    """Reality check e competition score — analisi di mercato approfondita."""
    if not idea:
        return "", 50

    res = ask(
        "Sei un analista che cerca SPAZI VUOTI. Sii selettivo: "
        "1 idea su 10 ha un competition_score > 65. La maggior parte "
        "ha concorrenza significativa (score 30-55). "
        "USA TUTTA LA SCALA 0-100. Identifica competitor reali.",
        f"""IDEA:
{idea}

Analisi (concreta, cita competitor reali):
1. Competitori diretti: esistono soluzioni IDENTICHE o solo simili?
2. Domanda: reale o hype? Ci sono community/gruppi che ne parlano?
3. Barriere: quanto è difficile per altri copiare l'idea?
4. Spazio: questa nicchia è servita o ignorata dai competitor?
5. Competition_score: 90=nicchia vergine, 70=poca concorrenza, 50=mediamente saturo, 30=molto saturo

Rispondi SOLO con JSON:
{{
  "analysis": "<analisi concisa>",
  "competition_score": <intero 0-100>
}}"""
    )

    try:
        s = res.find("{"); e = res.rfind("}") + 1
        if s == -1 or e == 0:
            raise ValueError("no JSON")
        j     = json.loads(res[s:e])
        score = max(0, min(100, int(j.get("competition_score", 50))))
        return j.get("analysis", ""), score
    except Exception:
        # Fallback: cerca un numero nella risposta
        m = re.search(r"\b(\d{1,3})\b", res)
        comp = max(0, min(100, int(m.group(1)))) if m else 50
        return res[:400] if res else "", comp

# =========================
# ENRICHMENT + SWOT (solo per idee top)
# =========================

def enrich_and_swot(idea: str, weak: str, reality: str) -> str:
    """Arricchisce un'idea con dettagli implementativi e analisi SWOT.
    
    Chiamato solo per idee con score >= ENRICH_SCORE_THRESHOLD.
    """
    res = ask(
        "Sei un senior product manager. Analizza l'idea e fornisci dettagli "
        "concreti per passare all'implementazione.",
        f"""IDEA:
{idea}

DEBOLEZZA (VC): {weak}
ANALISI MERCATO: {reality}

Fornisci in formato conciso:

**Stack tecnologico suggerito** (linguaggi, framework, servizi specifici)
**3 funzionalità MVP** (essenziali per il lancio)
**Canale acquisizione** (come trovare i primi 100 utenti)
**SWAT veloce** (Strengths, Weaknesses, Opportunities, Threats — 1 punto ciascuno)

Output max 200 parole, concreto e specifico."""
    )
    return res


# =========================
# FITNESS (ora solo 2 chiamate LLM invece di 3)
# =========================

def fitness(idea: str) -> tuple[float, str, str, list[str]]:
    vc, weak, concepts = evaluate_composite(idea)
    reality, comp      = reality_and_competition(idea)
    score              = (vc * 0.6) + (comp * 0.4)
    return score, weak, reality, concepts

# =========================
# MUTATION
# =========================

_mutation_idx = 0
def structured_mutate(idea: str, weak: str) -> str:
    """Mutazione strutturata: cicla su 3 tipi di trasformazione.
    
    Invece di chiedere genericamente 'crea una variante', guida il LLM
    verso un tipo specifico di cambiamento.
    """
    global _mutation_idx
    mutation_type = MUTATION_TYPES[_mutation_idx % len(MUTATION_TYPES)]
    _mutation_idx += 1

    return ask(
        "Sei un product architect creativo. Trasforma l'idea seguendo il tipo di mutazione indicato.",
        f"""IDEA ORIGINALE:
{idea}

PROBLEMA DA RISOLVERE:
{weak}

TIPO DI MUTAZIONE:
{mutation_type}

Crea una variante seguendo STRETTAMENTE questo tipo di mutazione.
L'idea risultante deve essere diversa dall'originale nel modo indicato.
Output conciso (max 150 parole): Nome, Problema, Soluzione, Target."""
    )

# =========================
# SIMILARITY (usa concetti già estratti)
# =========================

def is_similar_by_concepts(concepts: list[str], memory: dict) -> bool:
    """Controlla similarità usando i concetti già estratti da evaluate_idea."""
    if not concepts:
        return False
    known     = {k.lower() for k in memory["concepts"]}
    rare_hits = sum(
        1 for c in concepts
        if c.lower() in known and c.lower() not in COMMON_KEYWORDS
    )
    return rare_hits >= 3  # servono almeno 3 concetti rari in comune


def is_similar_to_recent_ideas(idea: str, memory: dict, threshold: int = 6) -> bool:
    """Confronto testuale con le ultime idee accettate."""
    recent = memory.get("recent_ideas", [])
    if not recent or not idea:
        return False
    new_words = {w.lower() for w in re.findall(r"[a-zA-ZÀ-ÿ]{4,}", idea) if w.lower() not in COMMON_KEYWORDS}
    if len(new_words) < 8:
        return False
    for prev in recent:
        prev_words = {w.lower() for w in re.findall(r"[a-zA-ZÀ-ÿ]{4,}", prev) if w.lower() not in COMMON_KEYWORDS}
        if len(new_words & prev_words) >= threshold:
            return True
    return False


# =========================
# EMBEDDING SIMILARITY
# =========================

def _text_hash(text: str) -> str:
    return hashlib.md5(text[:800].encode()).hexdigest()


def get_embedding(text: str, cache: dict | None = None) -> list[float]:
    """Vettore embedding via Ollama — ~10ms per chiamata, cache automatica."""
    if not text:
        return []
    h = _text_hash(text)
    if cache is not None and h in cache:
        return cache[h]
    try:
        resp = embed(model=EMBED_MODEL, input=text[:800])
        emb = resp["embeddings"][0]
        if cache is not None:
            cache[h] = emb
        return emb
    except Exception:
        return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def compute_novelty_bonus(idea_type: str, domain: str, variety: str, memory_snapshot: dict) -> float:
    """Bonus per idee che esplorano varietà o domini poco usati.
    
    Incoraggia il sistema a non fossilizzarsi sulle stesse combinazioni.
    Bonus maggiorato se ENTRAMBI variety e domain sono nuovi.
    """
    bonus = 0.0
    v_hist = memory_snapshot.get("varieties_history", [])
    d_hist = memory_snapshot.get("domains_history", [])
    v_new = variety and v_hist.count(variety) <= 1
    d_new = domain and d_hist.count(domain) <= 1
    if v_new:
        bonus += NOVELTY_VARIETY_BONUS
    if d_new:
        bonus += NOVELTY_DOMAIN_BONUS
    if v_new and d_new:
        bonus += 2  # compound bonus per combinazione completamente nuova
    return min(bonus, 7.0)


def is_similar_by_embedding(idea: str, memory_snapshot: dict) -> bool:
    """Similarità semantica via cosine similarity tra embedding.
    
    Molto più accurata del keyword matching — cattura idee che usano
    parole diverse ma concettualmente identiche.
    """
    recent = memory_snapshot.get("_recent_embeddings", [])
    if len(recent) < 1 or not idea:
        return False
    emb = get_embedding(idea)
    if not emb:
        return False
    for prev_emb in recent:
        if cosine_similarity(emb, prev_emb) > EMBED_SIM_THRESHOLD:
            return True
    return False


def update_memory(population: list[dict], memory: dict, embed_cache: dict | None = None):
    """Aggiorna la memoria con concetti, varietà, embedding e testi recenti."""
    top3 = sorted(population, key=lambda x: x["score"], reverse=True)[:3]
    for item in top3:
        memory["concepts"].extend(item.get("concepts", []))
        if item.get("type"):
            memory["types_history"].append(item["type"])
        if item.get("domain"):
            memory["domains_history"].append(item["domain"])
        if item.get("variety"):
            memory["varieties_history"].append(item["variety"])
        if item.get("idea"):
            memory["recent_ideas"].append(item["idea"])
        # Popola cache embedding per uso futuro
        if item.get("idea") and embed_cache is not None:
            get_embedding(item["idea"], cache=embed_cache)

    memory["concepts"]          = list(dict.fromkeys(memory["concepts"]))[-MAX_MEMORY_CONCEPTS:]
    memory["types_history"]     = memory["types_history"][-60:]
    memory["domains_history"]   = memory["domains_history"][-60:]
    memory["varieties_history"] = memory["varieties_history"][-30:]
    memory["recent_ideas"]      = memory["recent_ideas"][-30:]

    if embed_cache is not None:
        # Mantieni cache leggera: ultime 500 entry
        if len(embed_cache) > 500:
            keys = list(embed_cache.keys())[-500:]
            embed_cache = {k: embed_cache[k] for k in keys}
        save_embedding_cache(embed_cache)

# =========================
# HALL OF FAME
# =========================

def update_hall_of_fame(gen: int, population: list[dict], memory: dict):
    for item in population:
        if item["score"] >= HOF_SCORE:
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"hof_gen{gen:04d}_{ts}_score{item['score']:.0f}.json"
            path  = os.path.join(ACCEPT_DIR, fname)
            record = {
                "generation": gen,
                "timestamp":  datetime.now().isoformat(),
                "score":      round(item["score"], 1),
                "type":       item.get("type", "unknown"),
                "idea":       item["idea"],
                "weak":       item.get("weak", ""),
                "reality":    item.get("reality", ""),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

            hof = memory.setdefault("hall_of_fame", [])
            hof.append({
                "gen":   gen,
                "score": round(item["score"], 1),
                "type":  item.get("type", "unknown"),
                "idea":  item["idea"][:400],
            })
            hof.sort(key=lambda x: x["score"], reverse=True)
            memory["hall_of_fame"] = hof[:20]
            log(f"HALL OF FAME  score={item['score']:.1f} -> {fname}")

# =========================
# SAVE BEST (markdown)
# =========================

def save_best(gen: int, pop: list[dict]):
    if not pop:
        return
    best = max(pop, key=lambda x: x["score"])
    path = os.path.join(EVOL_DIR, f"gen_{gen:04d}_best.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Generazione {gen} — Score: {best['score']:.1f}\n\n")
        f.write(f"**Tipo:** {best.get('type', 'N/A')}  |  ")
        f.write(f"**Varietà:** {best.get('variety', 'N/A')}  |  ")
        f.write(f"**VC Score:** {best.get('vc_score', 'N/A')}  |  ")
        f.write(f"**Competition Score:** {best.get('comp_score', 'N/A')}\n\n")
        f.write("## Idea\n\n")
        f.write(best["idea"] + "\n\n")
        f.write("## Debolezze\n\n")
        f.write(best.get("weak", "") + "\n\n")
        f.write("## Reality Check\n\n")
        f.write(best.get("reality", "") + "\n")
        if best.get("enrichment"):
            f.write("\n## Enrichment & SWOT\n\n")
            f.write(best["enrichment"] + "\n")
    log(f"BEST SAVED  score={best['score']:.1f} -> {path}")

# =========================
# DASHBOARD DATA
# =========================

def load_dashboard_data() -> dict:
    if os.path.exists(DASHBOARD_DATA_FILE):
        try:
            with open(DASHBOARD_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"generations": [], "hall_of_fame": []}


def save_dashboard_data(gen: int, population: list[dict], memory: dict, total_candidates: int = 0):
    data       = load_dashboard_data()
    ideas_list = [
        {
            "type":       p.get("type", "unknown"),
            "score":      round(p["score"], 1),
            "idea":       p["idea"][:600],
            "weak":       p.get("weak", "")[:300],
            "variety":    p.get("variety", ""),
            "vc_score":   round(p.get("vc_score", 0), 1),
            "comp_score": round(p.get("comp_score", 0), 1),
            "enrichment": p.get("enrichment", "")[:500] if p.get("enrichment") else "",
        }
        for p in sorted(population, key=lambda x: x["score"], reverse=True)
    ]
    best_score = max(p["score"] for p in population) if population else 0.0
    avg_score  = sum(p["score"] for p in population) / len(population) if population else 0.0

    data["generations"].append({
        "gen":             gen,
        "timestamp":       datetime.now().isoformat(),
        "best_score":      round(best_score, 1),
        "avg_score":       round(avg_score, 1),
        "total_candidates": total_candidates,
        "ideas":           ideas_list,
    })
    data["last_updated"] = datetime.now().isoformat()
    data["current_gen"]  = gen
    data["min_score"]    = MIN_SCORE
    data["hof_score"]    = HOF_SCORE
    data["hall_of_fame"] = memory.get("hall_of_fame", [])

    with open(DASHBOARD_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# =========================
# HTTP SERVER (dashboard)
# =========================

def _start_server(output_dir: str):
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=output_dir, **kwargs)
        def log_message(self, fmt, *args):
            pass

    try:
        with socketserver.TCPServer(("", DASHBOARD_PORT), QuietHandler) as httpd:
            httpd.serve_forever()
    except OSError:
        log(f"Porta {DASHBOARD_PORT} gia' in uso — dashboard non disponibile")


def start_dashboard_server():
    t = threading.Thread(target=_start_server, args=(OUTPUT_DIR,), daemon=True)
    t.start()
    log(f"Dashboard -> http://localhost:{DASHBOARD_PORT}/dashboard.html")

# =========================
# PARALLEL CANDIDATE WORKER
# Ora ogni candidato fa solo 3 chiamate LLM:
#   1. generate_idea
#   2. evaluate_idea  (vc_score + weakness + concepts)
#   3. reality_and_competition (reality + comp_score)
# In totale da ~6 chiamate → 3 chiamate per idea
# =========================

def _generate_candidate(idx: int, memory_snapshot: dict) -> dict | None:
    try:
        gen = memory_snapshot.get("_gen", 0)
        current_min = memory_snapshot.get("_min_score", MIN_SCORE)

        for attempt in range(2):  # auto-retry su similarità
            # 1 — Genera idea
            idea, idea_type, domain, variety = generate_idea(memory_snapshot, gen=gen)
            if not idea:
                log(f"[{idx}] Idea vuota, skip")
                return None

            # 2 — Embedding similarity check (più accurato del keyword matching)
            if attempt == 0 and is_similar_by_embedding(idea, memory_snapshot):
                log(f"[{idx}] type={idea_type:<14} domain={domain[:20]:<20} EMBEDDING SIMILE -> retry")
                memory_snapshot["_retry_hint"] = f"L'idea '{idea[:80]}...' è troppo simile a una già generata. Cambia completamente approccio."
                continue
            memory_snapshot.pop("_retry_hint", None)

            # 3 — Valutazione composita (market + technical)
            vc, weak, concepts = evaluate_composite(idea)

            # Similarità: concetti già estratti (soglia più alta per evitare falsi positivi)
            if concepts and is_similar_by_concepts(concepts, memory_snapshot):
                log(f"[{idx}] type={idea_type:<14} domain={domain[:20]:<20} CONCETTI SIMILI -> skip")
                return None

            # NOTA: similarità testuale rimossa perché troppo aggressiva con 5+ idee in memoria.
            # L'embedding check sopra è semanticamente più accurato.

            # 4 — Reality check + competition score
            reality, comp = reality_and_competition(idea)

            # 5 — Score composito + novelty bonus (cap a 100)
            # vc è già widened dentro evaluate_composite, widiamo solo comp
            # Peso: VC 75% (qualità idea), Competition 25% (mercato)
            comp_w = _widen_score(float(comp))
            score = (vc * 0.75) + (comp_w * 0.25)
            novelty_bonus = compute_novelty_bonus(idea_type, domain, variety, memory_snapshot)
            score = min(100.0, score + novelty_bonus)

            if score < current_min:
                log(f"[{idx}] type={idea_type:<14} domain={domain[:20]:<20} score={score:.1f} < {current_min:.0f} -> scartata")
                return None

            log(f"[{idx}] type={idea_type:<14} domain={domain[:24]:<24} variety={variety[:18]:<18} score={score:.1f} (bonus={novelty_bonus:.0f}) OK")

            # Embedding per futuro confronto (usa cache)
            embedding = get_embedding(idea, cache=memory_snapshot.get("_embed_cache"))

            return {
                "idea":       idea,
                "score":      score,
                "weak":       weak,
                "reality":    reality,
                "type":       idea_type,
                "domain":     domain,
                "concepts":   concepts,
                "variety":    variety,
                "embedding":  embedding,
                "vc_score":   vc,
                "comp_score": comp,
            }

    except Exception as e:
        log(f"[{idx}] Errore generazione: {e}")
        return None

# =========================
# EVOLUTION
# =========================

def evolve(population: list[dict]) -> list[dict]:
    population.sort(key=lambda x: x["score"], reverse=True)
    n_survivors = min(SURVIVORS, len(population))
    survivors   = population[:n_survivors]

    new_pop = []
    for s in survivors:
        mutated = structured_mutate(s["idea"], s.get("weak", ""))
        if not mutated:
            new_pop.append(s)
            continue

        # Valutazione composita del mutante (cap a 100)
        vc, weak, concepts = evaluate_composite(mutated)
        reality, comp      = reality_and_competition(mutated)
        comp_w             = _widen_score(float(comp))
        score              = min(100.0, (vc * 0.75) + (comp_w * 0.25))

        new_pop.append({
            "idea":      mutated,
            "score":     score,
            "weak":      weak,
            "reality":   reality,
            "type":      s.get("type", "unknown"),
            "concepts":  concepts,
            "variety":   s.get("variety"),
            "embedding": s.get("embedding", []),
        })

    return new_pop

# =========================
# MAIN LOOP
# =========================

def parse_args():
    """CLI arguments per override rapido della configurazione."""
    p = argparse.ArgumentParser(description="HydraThinker — Evolutionary Idea Engine")
    p.add_argument("--model", default=None, help="Modello Ollama (es. qwen3:32b)")
    p.add_argument("--pop", type=int, default=None, help="Popolazione per generazione")
    p.add_argument("--workers", type=int, default=None, help="Worker paralleli")
    p.add_argument("--sleep", type=int, default=None, help="Pausa tra generazioni (sec)")
    p.add_argument("--min-score", type=int, default=None, help="Score minimo iniziale")
    p.add_argument("--port", type=int, default=None, help="Porta dashboard HTTP")
    p.add_argument("--reset", action="store_true", help="Resetta memoria e ricomincia")
    p.add_argument("--brief", default=None, help="Percorso alternativo per brief.md")
    return p.parse_args()


def apply_cli_args(args):
    """Applica override da CLI sulle costanti globali."""
    global MODEL, POP_SIZE, MAX_WORKERS, SLEEP, MIN_SCORE, DASHBOARD_PORT, BRIEF_FILE
    if args.model:       MODEL = args.model
    if args.pop:         POP_SIZE = args.pop
    if args.workers:     MAX_WORKERS = args.workers
    if args.sleep:       SLEEP = args.sleep
    if args.min_score:   MIN_SCORE = args.min_score
    if args.port:        DASHBOARD_PORT = args.port
    if args.brief:       BRIEF_FILE = args.brief
    if args.reset:
        log("Reset memoria forzato da CLI")
        for f_path in [MEMORY_FILE, DASHBOARD_DATA_FILE, EMBEDDING_CACHE_FILE]:
            try:
                os.remove(f_path)
            except FileNotFoundError:
                pass
        # Svuota anche le cartelle output
        for d in [EVOL_DIR, LOG_DIR, ACCEPT_DIR]:
            for ff in os.listdir(d):
                fp = os.path.join(d, ff)
                if os.path.isfile(fp):
                    os.remove(fp)


def load_embedding_cache() -> dict:
    """Carica cache embedding da disco."""
    try:
        with open(EMBEDDING_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_embedding_cache(cache: dict):
    """Salva cache embedding su disco (solo se ci sono novità)."""
    try:
        with open(EMBEDDING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def main():
    args = parse_args()
    apply_cli_args(args)
    memory = load_memory()
    embed_cache = load_embedding_cache()
    gen = 0

    # Variabili per adattamento dinamico
    current_pop = POP_SIZE
    pass_rates = []

    log("THINKER ULTRA v10 STARTED")
    log(f"MODEL={MODEL}  POP={current_pop}  WORKERS={MAX_WORKERS}  MIN_SCORE={MIN_SCORE}  HOF={HOF_SCORE}  TIMEOUT={LLM_TIMEOUT}s")
    log(f"TYPES={len(IDEA_TYPES)}  DOMAINS={len(IDEA_DOMAINS)}  VARIETIES={len(IDEA_VARIETIES)}")
    log(f"BASE_DIR -> {BASE_DIR}")
    log(f"Log file -> {_log_file}")
    if args.reset:
        log("Memoria resettata — nuova run")

    start_dashboard_server()

    while True:
        gen += 1

        # ── MIN_SCORE adattivo ──
        dashboard_data = load_dashboard_data()
        all_dashboard_gens = dashboard_data.get("generations", [])
        recent_gens = all_dashboard_gens[-ADAPTIVE_WINDOW:]
        all_scores = []
        for g_entry in recent_gens:
            for idea_entry in g_entry.get("ideas", []):
                all_scores.append(idea_entry["score"])
        if all_scores:
            all_scores.sort()
            median = all_scores[len(all_scores) // 2]
            adaptive_min = max(35, min(70, round(median * 0.75)))
        else:
            adaptive_min = MIN_SCORE

        # ── POP_SIZE adattivo ──
        if len(all_dashboard_gens) >= 2:
            recent_pass_rates = []
            for g_entry in all_dashboard_gens[-ADAPTIVE_LOOKBACK:]:
                total = g_entry.get("total_candidates", POP_SIZE)
                valid = len(g_entry.get("ideas", []))
                recent_pass_rates.append(valid / max(total, 1))
            avg_pass = sum(recent_pass_rates) / len(recent_pass_rates)
            if avg_pass < 0.4 and current_pop < MAX_POP_SIZE:
                current_pop = min(current_pop + 1, MAX_POP_SIZE)
                log(f"Pass rate {avg_pass:.0%} < 40% -> POP_SIZE={current_pop}")
            elif avg_pass > 0.85 and current_pop > MIN_POP_SIZE:
                current_pop = max(current_pop - 1, MIN_POP_SIZE)
                log(f"Pass rate {avg_pass:.0%} > 85% -> POP_SIZE={current_pop}")
        else:
            avg_pass = 1.0

        log(f"\n=== GENERATION {gen} ===  [MIN={adaptive_min}  POP={current_pop}  pass={avg_pass:.0%}]")

        # ── Embedding per idee recenti (dalla cache, no ricalcolo necessario) ──
        recent_embeddings = []
        for ridea in memory.get("recent_ideas", [])[-10:]:
            emb = get_embedding(ridea, cache=embed_cache)
            if emb:
                recent_embeddings.append(emb)

        # ── Snapshot per worker ──
        memory_snapshot = {
            "_gen":               gen,
            "_min_score":         adaptive_min,
            "_recent_embeddings": recent_embeddings,
            "_retry_hint":        None,
            "_embed_cache":       embed_cache,
            "concepts":           list(memory["concepts"]),
            "types_history":      list(memory["types_history"]),
            "domains_history":    list(memory.get("domains_history", [])),
            "varieties_history":  list(memory.get("varieties_history", [])),
            "recent_ideas":       list(memory.get("recent_ideas", [])),
        }

        # ── Generazione parallela con timeout per candidato ──
        population: list[dict] = []
        completed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(_generate_candidate, i, memory_snapshot): i
                for i in range(current_pop)
            }
            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result(timeout=CANDIDATE_TIMEOUT)
                    if result:
                        population.append(result)
                except Exception as e:
                    idx = futures[future]
                    log(f"[{idx}] Timeout/Errore: {e}")

        if not population:
            log("Popolazione vuota, skip evoluzione")
            time.sleep(SLEEP)
            continue

        pass_rate = len(population) / current_pop
        log(f"Idee valide: {len(population)}/{current_pop} ({pass_rate:.0%})")

        # ── Enrichment + SWOT per la miglior idea ──
        if population:
            best = max(population, key=lambda x: x["score"])
            if best["score"] >= ENRICH_SCORE_THRESHOLD:
                log(f"Enrichment idea top (score={best['score']:.1f})...")
                enrichment = enrich_and_swot(best["idea"], best.get("weak", ""), best.get("reality", ""))
                best["enrichment"] = enrichment
                log(f"Enrichment completato")

        # ── Hall of Fame ──
        update_hall_of_fame(gen, population, memory)

        # ── Salvataggio (PRIMA dell'evoluzione, così salviamo le idee originali) ──
        save_best(gen, population)
        save_dashboard_data(gen, population, memory, total_candidates=current_pop)

        # ── Memoria + cache embedding ──
        update_memory(population, memory, embed_cache)
        save_memory(memory)

        # ── Evoluzione (dopo salvataggio, per non contaminare i dati di questa gen) ──
        try:
            population = evolve(population)
        except Exception as e:
            log(f"Errore evoluzione: {e}")
            time.sleep(SLEEP)
            continue

        domains_used    = len(set(memory.get("domains_history", [])))
        varieties_used  = len(set(memory.get("varieties_history", [])))
        log(f"Concetti: {len(memory['concepts'])}  Domini: {domains_used}/{len(IDEA_DOMAINS)}  Varietà: {varieties_used}/{len(IDEA_VARIETIES)}")
        log(f"Cache embedding: {len(embed_cache)} testi")
        log(f"sleeping {SLEEP}s...")
        time.sleep(SLEEP)


if __name__ == "__main__":
    main()