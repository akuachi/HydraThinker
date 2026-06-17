# HydraThinker

Motore evolutivo per la generazione di idee startup. Usa LLM locali (via Ollama) per produrre, valutare e far evolvere idee di business concrete e diversificate, senza AI fixation.

## Requisiti

- Python ≥ 3.10
- [Ollama](https://ollama.com/) in esecuzione locale
- Modelli Ollama:
  - `qwen3:14b` — generazione e valutazione idee
  - `all-minilm` — embedding per dedup semantico

```
pip install -r requirements.txt
```

## Utilizzo

```
python thinker_ultra.py
```

### Opzioni CLI

| Flag            | Default    | Descrizione                                   |
|-----------------|------------|-----------------------------------------------|
| `--model`       | qwen3:14b  | Modello LLM per generazione/eval              |
| `--pop`         | 6          | Popolazione per generazione                   |
| `--workers`     | 4          | Worker paralleli                              |
| `--sleep`       | 10         | Pausa tra generazioni (secondi)               |
| `--min-score`   | 35         | Score minimo iniziale (adattivo)              |
| `--port`        | 8080       | Porta dashboard HTTP                          |
| `--reset`       | —          | Resetta memoria e riparte da zero             |
| `--brief`       | brief.md   | Percorso alternativo per file contesto        |

### Dashboard

Avviata automaticamente su `http://localhost:8080/dashboard.html`.

## Come funziona

1. **Generazione** — per ogni idea, il sistema combina tipo (es. `saas`, `bot`, `game`), dominio (es. `cibo`, `viaggi`) e varietà (es. `marketplace`, `tool AI`), e chiede al LLM di produrre un'idea concreta.
2. **Valutazione** — ogni idea viene valutata su mercato, fattibilità tecnica e competizione. Score composito con bonus di novità per esplorare nuove combinazioni tipo+dominio+varietà.
3. **Evoluzione** — le idee migliori sopravvivono e vengono mutate (cambio target, cambio business model, cambio dominio) per generazioni successive.
4. **Adattamento** — `MIN_SCORE` si calibra automaticamente in base allo storico recente. `POP_SIZE` si adatta in base al tasso di superamento.
5. **Hall of Fame** — idee con score ≥ 85 entrano nella Hall of Fame.

## Output

- `output/evolution/gen_NNNN_best.md` — miglior idea per generazione con analisi SWOT
- `output/accepted/hof_genNNNN_*.json` — Hall of Fame entries
- `output/dashboard_data.json` — dati per dashboard in tempo reale
- `memory.json` — memoria persistente (concetti, storico, embedding)

## Licenza

MIT
