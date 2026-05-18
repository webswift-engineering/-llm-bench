# llm-bench

Run the same tasks across every LLM provider. Get real cost, latency, and quality data.

**Dashboard:** after publishing, live at `https://<org>.github.io/-llm-bench/`

## Workflow

### 1. Local (before you push)

```bash
pip install -e ".[dev]"
cp .env.example .env   # add API keys

pytest -q
llm-bench run --task classification --models gpt-4o-mini,groq-llama-8b --budget 2.00 --save
llm-bench run --task summarization --models gpt-4o-mini --budget 1.50 --save
llm-bench dashboard    # preview at docs/index.html

git add data/benchmarks/ docs/
git commit -m "chore: update benchmark results"
git push
```

### 2. GitHub (automatic)

On push to `main` (or weekly schedule), Actions will:

1. Run `pytest`
2. Refresh pricing snapshot
3. Run benchmarks **if** repository secrets are set
4. Generate `docs/index.html` and deploy to **GitHub Pages**

**One-time setup:**

1. Repo → **Settings** → **Secrets** → add `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY` (optional but needed for CI benchmarks)
2. Repo → **Settings** → **Secrets and variables** → **Actions** → **Variables** → add `GA_MEASUREMENT_ID` (example: `G-XXXXXXXXXX`) for dashboard analytics
3. Repo → **Settings** → **Pages** → Source: **GitHub Actions**

### GA4 analytics

The dashboard injects Google Analytics 4 only when `GA_MEASUREMENT_ID` is set.
For GitHub Pages, create a GA4 web stream for your published URL, then add the
measurement ID as an Actions variable:

```text
GA_MEASUREMENT_ID=G-XXXXXXXXXX
```

If you prefer to keep it hidden, you can add it as an Actions secret with the
same name instead. The workflow supports both.

## Commands

| Command | Description |
|---------|-------------|
| `llm-bench prices` | Show model pricing |
| `llm-bench run --task <name> --save` | Run benchmark and save for dashboard |
| `llm-bench dashboard` | Generate `docs/index.html` for GitHub Pages |
| `llm-bench snapshot` | Save pricing snapshot JSON |
| `llm-bench compare` | Head-to-head comparison |

## Tasks

- **classification** — accuracy vs ground-truth labels
- **summarization** — LLM-as-judge quality score

## License

MIT
