# WebSearch LLM Agents

A free-first LangChain project with two agents:

- `agent.py`: a ReAct tool-calling agent with web search, Wikipedia, arXiv, and short-term memory.
- `rag_agent.py`: a RAG agent over a web page using local HuggingFace embeddings and a persistent Chroma vector DB.

Groq is used as the default chat model provider. Ollama is supported for the RAG agent when you want a fully local fallback.

## Features

- Free web search via `ddgs`
- Wikipedia lookup
- arXiv paper search
- LangChain ReAct tool use
- Thread-based short-term memory
- RAG over indexed web pages
- Persistent Chroma vector DB for faster retrieval
- Unit tests for tools and RAG helpers
- RAGAS, DeepEval, and a custom scorecard runner
- Graceful scorecard handling for Groq rate limits

## Setup

Python 3.11 or 3.12 is recommended.

```powershell
py -3.11 -m venv rag_env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\rag_env\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
CURRENT_DATE=2026-05-19
RAG_MODEL_PROVIDER=groq
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL=llama3.1:8b
```

`llama-3.1-8b-instant` is the default because it is lighter on Groq free-tier quota than `llama-3.3-70b-versatile`.

## Tool Agent

Run:

```powershell
python agent.py "Who is Ada Lovelace?" --show-tools
python agent.py "Find research papers about retrieval augmented generation." --show-tools
python agent.py "What is the current temperature in Edmonton, AB in Celsius?" --show-tools
```

Short-term memory:

```powershell
python agent.py "My name is Neeraj. I am testing this agent." --thread-id demo
python agent.py "What is my name?" --thread-id demo
python agent.py "Start fresh" --thread-id demo --clear-memory
```

The tool agent uses LangChain’s classic ReAct pattern:

```text
Thought -> Action -> Action Input -> Observation -> Final Answer
```

Available tools:

- `web_search`: current events, weather, latest facts, sports, news
- `wikipedia_search`: stable encyclopedic facts and background
- `arxiv_search`: academic papers and research literature

## RAG Agent

Index the default article:

```powershell
python rag_agent.py --index
```

Ask questions:

```powershell
python rag_agent.py "What is task decomposition?" --show-tools
python rag_agent.py "How does reflection help autonomous agents?" --show-tools
```

Rebuild the vector DB:

```powershell
python rag_agent.py --reindex
```

Index another page:

```powershell
python rag_agent.py "What is this page about?" --url https://example.com --selector "" --show-tools
```

RAG flow:

```text
web page -> BeautifulSoup text extraction -> chunking -> HuggingFace embeddings
-> Chroma vector DB -> retrieve_context tool -> LLM answer with sources
```

Vector data is stored locally in:

```text
.vector_db/chroma
```

## Evaluation

Run unit tests:

```powershell
pytest
```

Run live tool tests:

```powershell
$env:RUN_LIVE_TOOL_TESTS="1"
pytest tests/test_agent_tools.py
```

Generate reusable RAG eval records:

```powershell
python -m evals.rag_eval_runner
```

Run the custom scorecard:

```powershell
python -m evals.scorecard
```

The scorecard is Groq-aware. By default it uses `llama-3.1-8b-instant`, limits the number of test cases, and waits between calls to stay friendlier to the free tier.

According to Groq's rate-limit docs, free-plan limits include:

| Model | RPM | RPD | TPM | TPD |
|---|---:|---:|---:|---:|
| `llama-3.1-8b-instant` | 30 | 14,400 | 6,000 | 500,000 |
| `llama-3.3-70b-versatile` | 30 | 1,000 | 12,000 | 100,000 |

Use the 8B model for test runs because it has a much larger daily token budget.

Conservative free-tier run:

```powershell
python -m evals.scorecard --model llama-3.1-8b-instant --max-cases 3 --delay-seconds 10
```

Fuller run after quota resets:

```powershell
python -m evals.scorecard --model llama-3.1-8b-instant --max-cases 5 --delay-seconds 8
```

Outputs:

```text
results/scorecard.json
results/scorecard.csv
results/scorecard_summary.json
```

The scorecard tracks:

- question type coverage: memory, factual, research, current, RAG factual, out-of-scope
- expected tool/path selection
- answer content match
- final format compliance
- source citation quality
- tool efficiency
- latency and p95 latency
- retrieved context count
- unique source count
- average context size
- graceful error/rate-limit handling

If Groq quota is exhausted:

```powershell
python -m evals.scorecard --skip-live
```

The scorecard now marks Groq quota failures as `skipped_rate_limit` so they do not distort quality scores.

## RAGAS And DeepEval

RAGAS and DeepEval are optional and are commented in `requirements.txt` because they add heavier dependencies. For Python 3.11 or 3.12, uncomment:

```text
# ragas>=0.2.0
# deepeval>=3.0.0
```

Then reinstall:

```powershell
pip install -r requirements.txt
```

Run RAGAS:

```powershell
python -m evals.ragas_eval
```

Run DeepEval gently on Groq free tier:

```powershell
$env:DEEPEVAL_THROTTLE_SECONDS="8"
python -m evals.deepeval_rag_eval
```

If Groq returns a token-per-day `429`, wait for the reset shown in the error or switch the RAG agent to Ollama for local answering.

## Troubleshooting

PowerShell activation blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\rag_env\Scripts\Activate.ps1
```

Groq `429` rate limit:

- Use `GROQ_MODEL=llama-3.1-8b-instant`
- Wait for the reset time in the error
- Reduce eval dataset size
- Run `python -m evals.scorecard --skip-live`
- Use Ollama for RAG answering with `RAG_MODEL_PROVIDER=ollama`

HuggingFace warning:

```text
You are sending unauthenticated requests to the HF Hub
```

This is safe. Add `HF_TOKEN` only if you want higher download limits.
