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
- 20-link web corpus ingestion
- Unstructured HTML loading
- Semantic chunking
- Hybrid dense + BM25 retrieval
- Metadata filtering
- Reranking
- Self-reflection before final RAG answers
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
ENABLE_LLM_CACHE=1
AGENT_MAX_ITERATIONS=3
RAG_MODEL_PROVIDER=groq
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_EMBEDDING_DEVICE=cpu
RAG_EMBEDDING_CACHE_DIR=.cache/embeddings
RAG_ENABLE_RERANKING=1
RAG_RERANKER_MODEL=BAAI/bge-reranker-base
RAG_RERANKER_DEVICE=cpu
RAG_ENABLE_SELF_REFLECTION=1
RAG_ENABLE_LLM_CACHE=1
RAG_ENABLE_QUERY_ENHANCER=1
RAG_ENABLE_HYDE=0
RAG_CONTEXT_CHAR_LIMIT=5000
RAG_RESPONSE_CACHE_THRESHOLD=0.9
OLLAMA_MODEL=llama3.1:8b
```

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

The tool agent uses LangChain's classic ReAct pattern:

```text
Thought -> Action -> Action Input -> Observation -> Final Answer
```

Available tools:

- `web_search`: current events, weather, latest facts, sports, news
- `wikipedia_search`: stable encyclopedic facts and background
- `arxiv_search`: academic papers and research literature

## RAG Agent

Index the default 20-link corpus:

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
python rag_agent.py --reindex --clear-response-cache
```

Index a single custom page instead of the default corpus:

```powershell
python rag_agent.py --index --single-url --url https://example.com
python rag_agent.py "What is this page about?" --single-url --url https://example.com --show-tools
```

Use metadata filtering:

```powershell
python rag_agent.py "How does hybrid search work?" --topic-filter retrieval --show-tools
python rag_agent.py "What does LangChain say about agents?" --domain-filter docs.langchain.com --show-tools
```

RAG flow:

```text
user query -> local router -> query enhancer -> optional HyDE
-> semantic response cache lookup -> parallel Chroma dense search + BM25 sparse search
-> metadata filtering -> dedupe -> reranking -> limited context window
-> answer draft -> optional self-reflection -> final answer -> response cache write
```

Vector data is stored locally in:

```text
.vector_db/chroma
```

Cached HTML, embeddings, final responses, and BM25 chunks are stored locally in:

```text
.cache/html
.cache/embeddings
.cache/rag_responses.json
.vector_db/chunks.jsonl
```

Final response caching is semantic, not only exact-string based. The RAG agent embeds the new question, compares it against cached question embeddings, and reuses a cached answer when cosine similarity is above `RAG_RESPONSE_CACHE_THRESHOLD` (`0.9` by default).

Latency controls:

| Control | Default | Why it helps |
|---|---:|---|
| Local router | on | Picks obvious metadata topics before retrieval without an LLM call. |
| Query enhancer | on | Expands terms such as `RAG` to `retrieval augmented generation` before search. |
| Semantic response cache | on | Reuses answers for semantically similar questions, not just exact repeats. |
| LangChain LLM cache | on | Uses `langchain_core.caches.InMemoryCache` for exact prompt/model repeats during a process. |
| Parallel hybrid search | on | Runs dense Chroma retrieval and BM25 retrieval concurrently. |
| Direct RAG pipeline | on | Uses one deterministic routed retrieval path for normal RAG questions. |
| Limited context | 5000 chars | Caps tokens sent to the LLM. |
| HyDE | off | Available with `RAG_ENABLE_HYDE=1`, but disabled by default because it adds an LLM call. |

Default corpus topics include `agents`, `rag`, `tools`, `memory`, `retrieval`, `embeddings`, `chunking`, `vectorstores`, `loaders`, and `reranking`.

**Default Corpus**

| # | Topic | URL |
|---:|---|---|
| 1 | agents | https://lilianweng.github.io/posts/2023-06-23-agent/ |
| 2 | langchain | https://docs.langchain.com/oss/python/langchain/overview |
| 3 | agents | https://docs.langchain.com/oss/python/langchain/agents |
| 4 | tools | https://docs.langchain.com/oss/python/langchain/tools |
| 5 | rag | https://docs.langchain.com/oss/python/langchain/rag |
| 6 | memory | https://docs.langchain.com/oss/python/langchain/short-term-memory |
| 7 | loaders | https://docs.langchain.com/oss/python/integrations/document_loaders/url |
| 8 | retrieval | https://docs.langchain.com/oss/python/integrations/providers/rank_bm25 |
| 9 | embeddings | https://docs.langchain.com/oss/python/integrations/embeddings/index |
| 10 | vectorstores | https://docs.langchain.com/oss/python/integrations/vectorstores/chroma |
| 11 | chunking | https://python.langchain.com/docs/how_to/semantic-chunker/ |
| 12 | retrieval | https://python.langchain.com/docs/how_to/MultiQueryRetriever/ |
| 13 | reranking | https://python.langchain.com/docs/how_to/contextual_compression/ |
| 14 | retrieval | https://python.langchain.com/docs/how_to/ensemble_retriever/ |
| 15 | rag | https://www.pinecone.io/learn/retrieval-augmented-generation/ |
| 16 | retrieval | https://www.pinecone.io/learn/hybrid-search-intro/ |
| 17 | retrieval | https://weaviate.io/blog/hybrid-search-explained |
| 18 | retrieval | https://www.elastic.co/what-is/vector-search |
| 19 | embeddings | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| 20 | reranking | https://huggingface.co/BAAI/bge-reranker-base |

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

Conservative run:

```powershell
python -m evals.scorecard --max-cases 3 --delay-seconds 10
```

Fuller run:

```powershell
python -m evals.scorecard --max-cases 5 --delay-seconds 8
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

## Streamlit Dashboard

Run the dashboard:

```powershell
streamlit run streamlit_app.py
```

The app shows:

- the full RAG pipeline
- the 20-link corpus
- index/cache status
- tool-agent querying
- RAG querying with semantic response caching
- semantic response-cache status
- retrieved contexts and sources
- scorecard summaries
- RAGAS results
- local cache/artifact sizes

### Latest Scorecard

This run evaluated 8 cases across the tool-calling agent and the RAG agent.

**Overall Summary**

| Metric | Score |
|---|---:|
| Cases | 8 |
| Scored cases | 8 |
| Skipped due to rate limit | 0 |
| Average latency | 7.208s |
| Tool/path selection | 0.750 |
| Content match | 0.750 |
| Format compliance | 0.375 |
| Source quality | 0.300 |
| Tool efficiency | 0.500 |
| Latency score | 0.938 |
| Error handling | 1.000 |

**By Agent**

| Agent | Cases | Avg latency | P95 latency | Tool selection | Content match | Format | Sources | Tool efficiency | Error handling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tool agent | 4 | 5.864s | 6.149s | 0.500 | 1.000 | 0.750 | 0.600 | 0.125 | 1.000 |
| RAG agent | 4 | 8.553s | 8.972s | 1.000 | 0.500 | 0.000 | 0.000 | 0.875 | 1.000 |

**Question Coverage**

| Case | Agent | Category | Expected path | Observed behavior | Latency |
|---|---|---|---|---|---:|
| `memory_update` | Tool agent | Memory | `no_tool` | Answered conversationally, but ReAct reported `None('None')` instead of clean `no_tool` | 2.525s |
| `memory_followup` | Tool agent | Memory | `no_tool` | Used memory content, but again reported `None('None')` | 2.679s |
| `factual_wiki` | Tool agent | Factual | `wikipedia_search` | Found Ada Lovelace, but used repeated Wikipedia calls plus web search | 12.101s |
| `research_arxiv` | Tool agent | Research | `arxiv_search` | Retrieved arXiv papers; one extra `None('None')` step reduced efficiency | 6.149s |
| `rag_task_decomposition` | RAG agent | RAG factual | `retrieve_context` | Retrieved 3 chunks but answered "I don't know" | 7.776s |
| `rag_reflection` | RAG agent | RAG factual | `retrieve_context` | Retrieved 3 chunks but answered "I don't know" | 8.972s |
| `rag_memory` | RAG agent | RAG factual | `retrieve_context` | Correctly answered from retrieved memory context | 8.280s |
| `rag_out_of_scope` | RAG agent | Out-of-scope | `retrieve_context` | Correctly refused with "I don't know" | 9.185s |

**Takeaways**

- Error handling is strong: no crashes or leaked tracebacks in the completed run.
- Latency is acceptable overall, with most cases under 10 seconds.
- Tool selection needs cleanup for memory/no-tool cases because `None('None')` is being counted as a tool call.
- The tool agent over-calls tools on factual queries, hurting tool efficiency.
- The RAG agent retrieves context reliably, but answer formatting and source citation need improvement.
- RAG factual quality is mixed: memory-related retrieval worked, while task decomposition and reflection were too conservative.

## RAGAS And DeepEval

RAGAS is included in `requirements.txt` and is the main RAG evaluation path. DeepEval is optional because it adds heavier dependencies. For Python 3.11 or 3.12, uncomment:

```text
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

The RAGAS runner evaluates the upgraded RAG stack over the 20-link corpus. It regenerates records with:

- Unstructured HTML loading
- semantic chunking
- sentence-transformers embeddings
- hybrid dense + BM25 retrieval
- metadata filters from the eval dataset
- reranked contexts
- self-reflected answers

It requests every compatible RAGAS metric available in your installed version, including faithfulness, answer relevancy, answer correctness/similarity, context precision/recall, context entity recall, and noise sensitivity when supported. Results are written to:

```text
results/ragas_results.csv
results/ragas_results.json
results/ragas_summary.json
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

- Wait for the reset time in the error
- Reduce eval dataset size
- Run `python -m evals.scorecard --skip-live`
- Use Ollama for RAG answering with `RAG_MODEL_PROVIDER=ollama`

HuggingFace warning:

```text
You are sending unauthenticated requests to the HF Hub
```

This is safe. Add `HF_TOKEN` only if you want higher download limits.
