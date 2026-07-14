# AI Engineering Interview Prep — Weeks 1–2

Covers everything built and discussed across `week1/` (structured extraction & prompt engineering) and `week2/` (RAG). Organized to double as a concept refresher and an interview-answer bank — each section ends with the kind of question an interviewer would actually ask and a strong answer grounded in what you built.

---

## Part 1: Structured Extraction & Prompt Engineering (Week 1)

### 1.1 Structured outputs with Pydantic

**What you built:** `week1/day1/job_extractor.py` — a `JobAnalysis` Pydantic model (title, company, required/preferred skills, salary, match %, missing skills) populated by `client.beta.chat.completions.parse(..., response_format=JobAnalysis)`.

**Core concept:** instead of asking the model to "return JSON" and hoping it's parseable, `response_format` constrains decoding so the output *is* the schema — no manual JSON parsing, no regex extraction, no malformed-output retries. This is OpenAI's structured outputs feature, distinct from older "function calling as a JSON hack" patterns.

**Why it matters:** unstructured LLM output is a reliability liability in production — a downstream system consuming free-text can't handle "the model added a sentence before the JSON this time." Structured outputs push validation to the API boundary.

**Interview angle:**
> *"How do you get reliable structured data out of an LLM?"*
> Use schema-constrained decoding (Pydantic + `response_format`) rather than prompting for JSON and parsing it yourself. The schema does validation work the prompt can't guarantee — the model literally cannot emit a field of the wrong type. Reserve manual parsing/regex for models or APIs that don't support structured outputs.

---

### 1.2 Prompting strategies: zero-shot, few-shot, CoT, structured CoT

**What you built:** `week1/day2/job_extractor_prompting.py` implements the same extraction task four ways and benchmarks them (`match_percentage`, prompt/completion tokens, latency) across 3 real job descriptions.

| Strategy | What it does | When it helps |
|---|---|---|
| **Zero-shot** | Just the task instruction, no examples | Default choice — cheapest, fastest, and the schema already constrains output shape |
| **Few-shot** | 1-2 worked examples embedded in the system prompt | When output *format/style* is ambiguous and examples disambiguate it better than instructions |
| **Chain-of-thought (CoT)** | "Think step by step" | When the task requires actual reasoning (e.g., a judgment call), not extraction |
| **Structured CoT** | Explicit numbered steps ("step 1: identify title, step 2: ...") | Most control over reasoning path, but most tokens |

**The finding that matters (from your actual data, `week1/day2/read.me`):** zero-shot matched or beat CoT on accuracy across all 3 jobs, using fewer tokens and comparable latency. CoT/structured-CoT produced more *completion* tokens because they're forced to show reasoning steps — not because they extracted more correctly.

**The principle (this is your strongest interview line, already validated by your own benchmark):**
> "Don't use a complex prompting strategy when a simple one works. More tokens = more cost = more latency = more failure surface. Start simple, add complexity only when the data shows you need it. For pure extraction, the schema does the heavy lifting — I'd reserve CoT for tasks where the model needs to *reason*, like 'should we hire this person,' not just extract fields."

**Interview angle:**
> *"When would you use chain-of-thought prompting?"*
> Not by default. CoT adds cost and latency by forcing longer completions, and for structured extraction tasks a schema already constrains the output — reasoning steps don't improve field accuracy, they just add tokens. I'd reach for CoT when the task requires actual judgment or multi-step reasoning that the model would otherwise skip, and I'd validate that decision with a benchmark rather than assume it helps.

---

### 1.3 Cost, latency, and token economics

**Core concept:** the strategy that produces the *most output tokens* isn't always giving you more value — CoT/structured-CoT are verbose because they show reasoning, few-shot is verbose because the model mirrors your example's format. More output tokens = more $ + more latency, independent of correctness.

**Pricing intuition you derived:** `gpt-4o-mini` is roughly $0.15/1M input tokens, $0.60/1M output. Few-shot's extra ~350 input tokens/call, at 10,000 calls, costs about $0.52 extra — negligible. The same extra tokens on `gpt-4o` (~$2.50/1M input) costs ~$8.75 extra; worse again on `gpt-4`. **Model choice affects cost more than prompt strategy does.**

**Interview angle:**
> *"How do you control LLM API costs at scale?"*
> Two levers, in order of impact: model selection (mini/small models vs. flagship models is often a 10-20x price difference) and prompt strategy (verbose strategies like CoT increase completion tokens). I'd benchmark both — token count and $ cost — before assuming a "smarter-sounding" strategy or model is worth its cost. Also: cache duplicate requests, batch where the API supports it, and know your provider's per-minute rate limits before you scale traffic.

---

### 1.4 Reliability & determinism

**Concept: temperature.** At `temperature=0`, the model always picks the highest-probability token — deterministic-ish output (not perfectly, since some randomness exists in serving infra, but close). At higher temperatures (e.g. `0.7`, used in the chatbot for natural conversation variety), the model samples from a wider distribution — good for creative/conversational tasks, bad for structured extraction where you want the same JD to produce the same fields every run.

**Concept: prompt sensitivity even at temperature=0.** Different prompt wording (e.g., structured-CoT's step-by-step instructions vs. zero-shot's single instruction) sends the model down a different reasoning path and can produce different results even fully deterministic. This is a real production gotcha — you can't assume "same input, same prompt template version → same output" across prompt *edits*.

**Production mitigation strategies (your own notes, well worth repeating verbatim in an interview):**
- Don't trust the model's arithmetic — calculate derived values (like `match_percentage`) in code, not via the LLM.
- Run ambiguous/high-stakes inputs multiple times and take a consensus.
- Add validation rules as a backstop (e.g., "if `missing_skills` is empty, `match_percentage` must be 100").

**Interview angle:**
> *"How do you make LLM outputs reliable enough for production?"*
> Assume non-determinism even at temperature=0 — different prompt versions can diverge. I push anything calculable (math, aggregation) out of the LLM and into code, add schema validation as a hard constraint, and for high-stakes fields I'll run multiple samples and take consensus rather than trusting a single completion.

---

### 1.5 Multi-turn conversation state

**What you built:** `week1/day3/chatbot.py` — an interview-prep coach using `gr.ChatInterface`, where `chat(user_message, history)` reconstructs the full message list (`system` + alternating `user`/`assistant` from `history`) on every call.

**Core concept:** LLM APIs are stateless — the model has no memory between calls. "Conversation" is an illusion maintained entirely client-side: you resend the *entire* transcript (or a summarized/truncated version of it) as the `messages` array on every single request. This is why token costs grow with conversation length, and why long conversations eventually need summarization or truncation strategies.

**Interview angle:**
> *"How does an LLM 'remember' earlier turns in a conversation?"*
> It doesn't — each API call is independent. The client resends the full message history every turn. That has real cost implications: a 50-turn conversation resends (and re-bills) all 50 turns' worth of tokens on turn 51 unless you truncate or summarize older context.

---

### 1.6 Production scaling considerations

From your own Q&A grading notes — worth internalizing as a checklist for "how would you scale this to 10,000 requests":
- **Batching / async** (`asyncio`) — don't serialize independent API calls.
- **Caching** — avoid re-processing duplicate inputs (e.g., the same JD).
- **Rate limiting** — respect provider request-per-minute caps.
- **Error handling & retries** — expect transient API failures at scale.

---

## Part 2: Retrieval-Augmented Generation (Week 2)

### 2.1 The RAG pipeline, end to end

You built this exact pipeline in `week2/rag.py`, iteratively:

```
load → split (chunk) → embed → store (vector DB) → retrieve → augment prompt → generate
```

**Why RAG exists:** it gives an LLM knowledge beyond its training data or context window *without fine-tuning*. Swap the source document, re-embed, and the system's "knowledge" changes — no retraining. This is the single most important framing for any RAG interview question: **RAG is a retrieval problem wrapped around a generation problem**, and most of the engineering work (and most of what breaks) is in the retrieval half, not the LLM call.

---

### 2.2 Document loading & chunking

- **`PyPDFLoader`** parses a PDF into one `Document` per page — `page_content` (text) + `metadata` (source, page number). Metadata is what enables citations later.
- **Why chunk at all:** you can't embed a whole book as one vector — embedding models have token limits, and a single vector can't precisely represent a whole document's meaning (retrieval gets vague/generic).
- **`RecursiveCharacterTextSplitter`** tries to split on paragraph → sentence → word boundaries, not blindly at character N — it's "recursive" because it falls back through separators until chunks fit `chunk_size`.
- **`chunk_size` / `chunk_overlap` tradeoffs** (you validated this empirically in `compare_chunk_sizes.py`):
  - Too small (**200 chars**, tested): chunks cut mid-thought — retrieval found the right *pages* but the LLM said "I don't know" because no single chunk contained a coherent, complete answer.
  - **500 / 1000 chars** (tested): both produced correct, well-cited answers. 1000 slightly improved MRR (better-ranked top results) and pushed judge quality to a perfect average, likely because each chunk carried more surrounding context for generation.
  - Overlap (kept at 50 chars) exists to avoid severing a sentence/idea exactly at a chunk boundary.

**Interview angle:**
> *"How do you choose chunk size for RAG?"*
> Empirically, not by rule of thumb — I ran the same query set across 200/500/1000-char chunks against the same document and measured both retrieval accuracy (hit rate/recall/MRR) and answer quality. 200 chars was provably too small: retrieval found the correct source pages, but individual chunks lacked enough coherent context for the LLM to answer, so the model said "I don't know" despite retrieval technically "working." That's a concrete example of retrieval quality and generation quality being separate failure modes — a good chunk-size choice optimizes for the second stage as much as the first.

---

### 2.3 Embeddings

- An embedding is a vector (list of floats) encoding semantic meaning, such that similar meaning → close vectors (cosine/Euclidean distance).
- You used OpenAI's embedding model via `OpenAIEmbeddings()` (`text-embedding-3-small` by default) — a paid API call **per chunk** at ingestion time, and **per query** at retrieval time.
- **Critical constraint:** the same embedding model must be used for both documents and queries — otherwise the two vector spaces aren't comparable and retrieval breaks silently (no error, just bad results).

**Interview angle:**
> *"What's an embedding, mechanically?"*
> A fixed-length vector produced by a model trained so that semantically similar text produces vectors that are close together under some distance metric (cosine similarity is common). It's not a compression of the exact text — it's a projection into a space where "closeness" approximates "similar meaning," which is what makes nearest-neighbor search a viable retrieval mechanism.

---

### 2.4 Vector stores & Chroma internals

- **Chroma** stores each chunk's text, metadata, and embedding vector, and builds an index for fast approximate similarity search.
- **`persist_directory`** — the difference between an in-memory store (gone when the process exits) and one durable across runs. You hit this exact gotcha: a stale persisted store silently kept old 500-char-chunk embeddings even after the code changed to `chunk_size=1000`, because the ingestion code only checked "does a store exist," not "does the store match the current config." Fix: separate persist directories per experiment (`chroma_db`, `chroma_db_compare`, `chroma_db_eval`), or a guard that checks chunk count against expected count before reusing a store.
- **Collections** — a named partition inside a Chroma DB; you used this to hold three separate chunk-size experiments (`chunks_200`, `chunks_500`, `chunks_1000`) inside one `chroma_db_compare` directory.
- **HNSW (Hierarchical Navigable Small World)** — the graph-based approximate-nearest-neighbor index Chroma uses under the hood instead of brute-force comparing the query vector against every stored vector. This is what makes similarity search fast at scale (sub-linear rather than O(n) per query).
- **On-disk structure** you inspected directly: `chroma.sqlite3` (source of truth for chunk text/metadata) + a UUID-named folder per collection containing `data_level0.bin` (the actual HNSW graph/vectors), `header.bin`, `length.bin`, `link_lists.bin`.

**Interview angle:**
> *"How does a vector database find similar items quickly?"*
> Not by brute-force comparing against every stored vector — that's O(n) per query and doesn't scale. Chroma (like most vector DBs) builds an approximate-nearest-neighbor index, typically HNSW: a multi-layer graph structure that lets you navigate toward nearest neighbors in roughly logarithmic time, trading a small amount of recall accuracy for large speed gains at scale.

---

### 2.5 Retrieval mechanics

- **`vectorstore.as_retriever(search_kwargs={"k": N})`** wraps the vector store in LangChain's standard `Retriever` interface (`.invoke(query) -> list[Document]`) — the same abstraction chains, agents, and tools expect, so you can swap vector stores without touching downstream code.
- **`k` (top-k)** is the main retrieval quality/noise knob: too low risks missing the answer (especially if it spans multiple non-adjacent chunks — this is exactly why your URL-shortener eval question failed, see 2.7); too high dilutes the LLM's context with irrelevant chunks, which can actively hurt answer quality (models get distracted by noise) and costs more tokens.
- Retrieval mechanically: embed the query with the *same* embedding model → run ANN search over the index → return the `k` nearest chunks by distance (L2 or cosine, configurable).

---

### 2.6 Prompt grounding & citations

- **"Stuffing"** — the context-injection strategy you used: concatenate all retrieved chunks into one string and hand the whole thing to the LLM in the prompt. Simple and effective at small `k`/`chunk_size`; breaks down at scale (context window limits) where you'd need map-reduce or refine strategies instead.
- **Grounding instruction** ("answer using only the context below... if it doesn't contain the answer, say you don't know") is the core anti-hallucination technique in RAG — it constrains the model to retrieved facts and gives it an explicit escape hatch instead of confabulating.
- **Citations, and the trust distinction that matters:** you built two kinds —
  - *Inline citation* (`[page 9]`) is the **model's own claim** about which page it used — it can be wrong or incomplete.
  - The `Sources: pages [...]` line is **ground truth** — it's just the actual metadata from whichever chunks were retrieved, not something the model could get wrong.
  
  This distinction (model-generated vs. system-verified provenance) is a genuinely good interview point about trustworthy AI systems.

**Interview angle:**
> *"How do you prevent a RAG system from hallucinating?"*
> Two layers: prompt-level grounding (explicit "answer only from context, say 'I don't know' otherwise" instructions), and system-level citation — surfacing which chunks were *actually retrieved* (ground truth, from retrieval metadata) separately from what the model *claims* it cited (which is itself a generation and can be wrong). The second is more trustworthy than the first, and a serious RAG system should expose both, not conflate them.

---

### 2.7 Evaluating RAG systems

**What you built:** `week2/eval.py` — a 10-question eval set with verified ground-truth pages (checked against the actual PDF text, not guessed), scoring:

| Metric | What it measures | How it's computed |
|---|---|---|
| **Hit Rate** | Did retrieval find *at least one* correct source page, at all? | 1 if any expected page ∈ retrieved pages, else 0 |
| **Recall@k** | What *fraction* of all expected pages did retrieval find? | `found / len(expected_pages)` |
| **MRR** (Mean Reciprocal Rank) | *How high-ranked* was the first correct hit? | `1 / (rank of first correct hit)`, 0 if none found |
| **LLM-as-judge quality (1-5)** | Is the generated *answer* actually good? | A second LLM call scores `(question, expected_answer, actual_answer)` |

**Why you need both retrieval metrics and judge scores — this is the single most important insight from your eval run:** they measure different failure modes and can *diverge*. Your eval surfaced a case where hit rate was 0 (retrieval genuinely failed to find the URL-shortener content) but judge quality still scored 3-5/5 — because the LLM produced a plausible-sounding answer from general knowledge or adjacent context, not from the retrieved chunks. **Judge-only evaluation would have completely hidden that retrieval failure.** This is exactly the blind spot that makes "the LLM's answer sounded good" an unreliable signal on its own.

**Root-causing a specific retrieval failure (your URL-shortener case):** hit rate stayed 0 at *both* 500-char and 1000-char chunk sizes — ruling out chunk-boundary cutting as the cause. More likely explanations: the query's phrasing ("how would you design a URL shortener") doesn't lexically/semantically align with how the book actually discusses it (hashing + collision resolution, without necessarily using "URL shortener" near that explanation), or the answer spans multiple non-adjacent pages (`[11, 13]`) that no single top-k chunk captures well. This is the kind of diagnosis interviewers want to hear — not "it just doesn't work," but a specific, testable hypothesis.

**Interview angle:**
> *"How do you evaluate a RAG system?"*
> Two axes, and you need both: retrieval quality (hit rate, recall@k, MRR — do we find the right source material) and answer quality (LLM-as-judge, or human eval, against a reference answer). They can diverge — a model can produce a plausible answer even when retrieval completely failed, which judge-only evaluation won't catch, and that's a dangerous blind spot in production because it looks like the system is working. Ground truth for the eval set has to come from the actual source document, not assumption — I verified every expected page by grepping the real PDF text rather than guessing.

---

### 2.8 Experimentation methodology & pitfalls

- **Stale persisted state as a silent experiment killer.** The single biggest "gotcha" across this project: a `persist_directory` existence check (`if os.path.exists(...)`) tells you *a* store exists, not that it matches your *current* config. You hit this twice — once in `rag.py` (chunk_size changed in code but the old store kept loading), and it's the exact reason `eval.py` was later pointed at its own `chroma_db_eval/` directory. **Lesson: any time an experiment changes a parameter that affects what gets embedded, either use a fresh persist directory or explicitly validate the existing store's chunk count/config before reusing it.**
- **Idempotent ingestion guards matter for cost, not just correctness.** Without a "does this already exist" check, every script run re-embeds and re-inserts all chunks — duplicating data in the collection and re-billing the embedding API for no reason.
- **Ground truth must come from the source, not assumption** — you built the eval set's expected pages by programmatically searching the actual PDF text for each topic (`docs[p].page_content`) rather than guessing which page a concept "should" be on.

---

### 2.9 Serving & deploying

- **`gr.ChatInterface` vs. `gr.Interface`** — different UI primitives for different interaction shapes. `ChatInterface` assumes a stateful back-and-forth conversation (chat bubbles); `Interface` is a generic "one function, some inputs, some outputs" wrapper, better suited to single-shot Q&A (what you used for the eval UI). Note: your chat UI's `history` parameter was accepted but unused — each question was answered independently with no memory of prior turns, unlike the week1 chatbot which *did* reconstruct history.
- **`if __name__ == "__main__": demo.launch()`** — without this guard, importing the module anywhere (e.g., to reuse a helper function) would block on launching a web server. Basic Python packaging hygiene that matters a lot for scripts meant to be both runnable and importable.
- **Git hygiene for ML projects:** derived/regeneratable artifacts (vector stores) and large source binaries (the PDF) don't belong in version control — only the code that reproduces them does. You iteratively `.gitignore`'d `chroma_db*/`, the PDF, and later `hf_space/` for this reason.
- **Deployment reality check (Hugging Face Spaces):** free-tier Gradio/Docker Space hosting now requires a paid PRO subscription — only *static* Spaces (HTML/JS, no Python backend) are free. This is a genuinely useful thing to know before promising a live demo link in an interview context — check current hosting costs before committing to a platform.
- **Copyright/licensing awareness:** a public Q&A tool built on a copyrighted book can effectively reproduce that book's content through retrieval + citation, even without redistributing the raw PDF — that's a real consideration for anything you'd demo publicly with someone else's source material.
- **Secrets hygiene:** API keys belong in `.env` (loaded via `load_dotenv()`, gitignored) or platform-level secrets (e.g., HF Space secrets), never hardcoded or pasted into a chat/log — a token pasted into any text channel should be treated as compromised and rotated immediately, regardless of whether it was "used" for anything malicious.

---

## Part 3: How the two weeks connect (a good interview narrative)

If asked "walk me through a project you've built," the throughline across both weeks is:

1. **Week 1** is about *getting reliable, structured signal out of a single LLM call* — schema-constrained outputs over prompt-and-pray, and empirically choosing the cheapest prompting strategy that meets the accuracy bar (validated by your own zero-shot vs. CoT benchmark).
2. **Week 2** is about *giving an LLM access to knowledge it doesn't have*, and rigorously **measuring** whether that actually works — not just "the demo looked good," but hit rate/recall/MRR for retrieval and LLM-judge for generation, run against a verified ground-truth set, with the discipline to catch when those two signals disagree.

The strongest overall interview stance: you don't trust an LLM system because the output *looks* right — you built the instrumentation (token/cost/latency benchmarks in week 1, retrieval + quality metrics in week 2) to verify it, and that instrumentation caught real problems (CoT wasn't worth its cost; 200-char chunks silently broke answers; a stale vector store invalidated a comparison; hit-rate-0 hid behind a plausible-sounding answer).
