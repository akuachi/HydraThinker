# HydraThinker

Motore evolutivo per la generazione di idee startup. Usa LLM locali (via Ollama) per produrre, valutare e far evolvere idee di business concrete e diversificate.

## Requisiti

- Python ≥ 3.10
- [Ollama](https://ollama.com/) in esecuzione locale
- Modelli Ollama:
  - `qwen3:14b` — generazione e valutazione
  - `all-minilm` — embedding per dedup semantico

```
pip install -r requirements.txt
```

## Utilizzo Base

```
python thinker_ultra.py
```

### Opzioni CLI

| Flag            | Default       | Descrizione                                    |
|-----------------|---------------|------------------------------------------------|
| `--config`      | config.json   | Percorso file di configurazione                |
| `--model`       | da config     | Modello LLM                                    |
| `--pop`         | da config     | Popolazione per generazione                    |
| `--workers`     | da config     | Worker paralleli                               |
| `--sleep`       | da config     | Pausa tra generazioni (sec)                    |
| `--min-score`   | da config     | Score minimo iniziale                          |
| `--port`        | da config     | Porta dashboard HTTP                           |
| `--reset`       | —             | Resetta memoria e ricomincia                   |
| `--brief`       | da config     | Percorso alternativo per brief.md              |

## Configurazione (config.json)

Tutti i parametri sono configurabili via `config.json`:

- **Modelli**: LLM, embedding, temperatura, contesto
- **Popolazione**: pop_size, survivors, min/max pop, score minimo
- **Pesi**: VC score, competition score, bonus novità
- **Liste**: tipi idea, domini tematici, varietà, mutazioni
- **Soglie**: enrichment, embedding similarity, concept dedup
- **Prompts**: ogni prompt LLM (sistema, criteri, ancore) è personalizzabile
- **brief_mode**: `"inspiration"` (default) o `"strict"` (brief come regole)
- **Esempi**: lista esempi per generazione idea

### Multi-istanza

Per eseguire più istanze in contemporanea:

```
# Crea una cartella per ogni istanza
mkdir instances\istanza1 instances\istanza2

# Copia config.json in ogni cartella, modifica porta e percorso
cp config.json instances\istanza1\
# modifica port, instance_dir, brief_file in instances\istanza1\config.json

# Lancia ogni istanza con la propria config
python thinker_ultra.py --config instances\istanza1\config.json
python thinker_ultra.py --config instances\istanza2\config.json
```

Ogni istanza ha memoria, output, brief e porta propri.

## Dashboard

Avviata automaticamente su `http://localhost:{port}/dashboard.html`.

## Come Funziona

1. **Generazione** — combina tipo + dominio + varietà, LLM produce idea
2. **Valutazione** — mercato (market) + fattibilità (technical) + competizione
3. **Evoluzione** — sopravvivenza dei migliori + mutazione strutturata
4. **Adattamento** — MIN_SCORE e POP_SIZE si calibrano automaticamente
5. **Hall of Fame** — idee con score ≥ HOF_SCORE

## Output

- `output/evolution/gen_NNNN_best.md` — miglior idea con SWOT
- `output/accepted/hof_genNNNN_*.json` — Hall of Fame
- `output/dashboard_data.json` — dati dashboard in tempo reale

## Licenza

MIT
