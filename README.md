# llm-bench

Run the same tasks across every LLM provider. Get real cost, latency, and quality data.

```bash
pip install -e .
llm-bench prices
llm-bench run --task classification --budget 5.00
llm-bench compare --baseline gpt-4o-mini --challenger groq-llama-3.1-70b --task summarization
```

## Setup

Copy `.env.example` to `.env` and add API keys for the providers you want to benchmark:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

## Commands

| Command | Description |
|---------|-------------|
| `llm-bench prices` | Show current model pricing across providers |
| `llm-bench recommend --task summarization` | Suggest best value models for a task |
| `llm-bench snapshot` | Save pricing snapshot to `data/snapshots/` |
| `llm-bench run --task <name>` | Run benchmark across configured models |
| `llm-bench compare --baseline X --challenger Y` | Head-to-head comparison |
| `llm-bench run --dry-run` | Show what would run without API calls |

## Tasks

- **classification** — accuracy vs ground-truth labels (deterministic)
- **summarization** — LLM-as-judge quality score (uses a different model as judge)

## License

MIT
