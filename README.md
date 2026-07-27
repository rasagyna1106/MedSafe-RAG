# MedSafe

Caregiver-facing medication safety assistant powered by **Retrieval-Augmented Generation (RAG)** over **openFDA drug labels**.

MedSafe answers everyday questions like *“Can mom take warfarin and ibuprofen together?”* using **only retrieved FDA label text** with citations, abstention when evidence is weak, and a shared traffic-light risk level for both **Ask a Question** and **Medication Checker**.

---

## Aim

Help family caregivers understand medication safety information from FDA labels in plain English, without requiring medical training while staying grounded in source text and clearly marking uncertainty.

## Goals

- Ground every answer in **openFDA drug label** excerpts (not general web knowledge).
- Support **brand → generic** rewriting (e.g. Tylenol → acetaminophen).
- Provide **citations** back to drug + label section (+ product when available).
- **Abstain** when retrieval is weak, ambiguous, or the drug is outside the corpus.
- Classify interaction risk **deterministically from retrieved FDA chunks** (HIGH / MODERATE / LOW / UNKNOWN) — shared by both UI tabs.
- Offer two caregiver workflows:
  - **Ask a Question** — free-text Q&A
  - **Medication Checker** — multi-med list → pairwise interaction matrix

## Outcomes

| Outcome | What you get |
|--------|----------------|
| Working web app | FastAPI UI at `http://localhost:7860` |
| Indexed corpus | 18 target generics → filtered labels → ~3,497 chunks → FAISS index |
| Caregiver answers | GPT-4o-mini answers in warm, plain English with `(Source: …)` citations |
| Risk UI | Red / amber / green / gray cards for HIGH / MODERATE / LOW / UNKNOWN |
| Pair checker | Matrix + short pair summaries + FDA source table |
| Eval harness | 30 test questions in `eval/test_questions.json` + `src/evaluate.py` |

**Not medical advice.** Always consult a pharmacist or doctor for real decisions.

---

## How to clone and run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd "MedSafe RAG"
```

If you already have the project folder locally, skip clone and `cd` into it:

```bash
cd "MedSafe RAG"
```

> Replace `<YOUR_GITHUB_REPO_URL>` with your GitHub URL, for example:  
> `https://github.com/<username>/MedSafe-RAG.git`

### 2. Create a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your OpenAI API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-key-here
```

### 5. Confirm data / index are present

The repo is expected to include (or you rebuild — see [Rebuild the pipeline](#rebuild-the-pipeline-optional)):

| Path | Purpose |
|------|---------|
| `data/raw/drug-label-0001-of-0013.json` … `0003` | Raw openFDA dumps |
| `data/processed/drugs_filtered.json` | Filtered label records |
| `data/processed/chunks.json` | Chunked label text |
| `data/processed/brand_mapping.json` | Brand → generic map |
| `data/index/faiss.index` | Vector index |
| `data/index/metadata.json` | Chunk metadata for retrieval |

### 6. Start the app

```bash
python app.py
```

Open: **http://localhost:7860**

---

## Demo questions (Ask a Question)

| Color | Risk | Example question |
|-------|------|------------------|
| Green | LOW | `Can insulin glargine and acetaminophen be taken together?` |
| Yellow | MODERATE | `Can metformin and lisinopril be taken together?` |
| Red | HIGH | `Can warfarin and ibuprofen be taken together?` |
| Gray | UNKNOWN | `Can amlodipine and lisinopril be taken together?` |

---

## Project architecture

```text
Browser (Ask / Medication Checker)
        │
        ▼
   app.py  (FastAPI routes)
        │
        ▼
 web_service.py  (UI metrics, TLDR, pair matrix shaping)
        │
        ▼
 src/generate.py  (retrieve → classify risk → GPT-4o-mini → citations)
        │
        ▼
 data/index (FAISS + metadata)  ← built by ingest → mapping → chunk → embed
```

### Offline pipeline (`src/`)

| Module | Role |
|--------|------|
| `ingest.py` | Stream openFDA JSON; keep 18 target generics; write `drugs_filtered.json` |
| `mapping.py` | Build brand→generic map; `resolve_query()` for caregiver brand names |
| `chunk.py` | Section-aware chunking (~500 tokens, 50 overlap) → `chunks.json` |
| `embed.py` | Embed with `all-MiniLM-L6-v2`; build FAISS index |
| `generate.py` | RAG generation, abstention, `classify_risk_from_chunks()` |
| `evaluate.py` | Run eval set; write results CSV |

### Online app

| File | Role |
|------|------|
| `app.py` | FastAPI entrypoint (`/`, `/api/ask`, medication endpoints) |
| `web_service.py` | API adapter: Ask structuring + Medication Checker |
| `templates/index.html` | UI shell |
| `static/js/app.js` | Frontend logic (both tabs) |
| `static/css/style.css` | Shared risk color tokens and layout |

---

## Dataset

### Source
- **openFDA Drug Label** files: parts **1–3 of 13** (~2 GB total) under `data/raw/`
- Streamed with **`ijson`** (not fully loaded into RAM)

### Target drugs (18 generics)

| Category | Drugs |
|----------|--------|
| Blood thinners | warfarin, apixaban, rivaroxaban |
| Diabetes | metformin, insulin glargine, glipizide |
| Blood pressure | lisinopril, amlodipine, metoprolol |
| Cholesterol | atorvastatin, simvastatin |
| Pain | ibuprofen, naproxen, acetaminophen |
| Antibiotics | amoxicillin, azithromycin |
| Mental health | sertraline, lorazepam |

### Processed stats (current build)

| Artifact | Approx. size |
|----------|----------------|
| Filtered label records | **302** (`drugs_filtered.json`) |
| Chunks | **~3,497** (`chunks.json`) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Default retrieval | **top-k = 5** |
| LLM | **gpt-4o-mini** |

### Known dataset limits
- **apixaban:** 0 records in openFDA parts 1–3 (documented coverage gap).
- Sparse coverage for some drugs (e.g. insulin glargine has few labels in these parts).
- Ask-tab drug extraction is strongest for the **18 corpus drugs** (and mapped brands). Drugs outside the corpus may show as Unknown / abstain on Ask even when Medication Checker can still run an explicit pair.

---

## Risk classification

Risk is computed **from retrieved FDA chunks before the LLM runs** (`classify_risk_from_chunks` in `src/generate.py`):

1. Consider only chunks for the two named drugs.
2. Restrict to sections: `drug_interactions`, `warnings`, `contraindications`.
3. Assign:
   - **UNKNOWN** — no relevant text for the pair  
   - **HIGH** — bleeding / contraindicated / fatal-style patterns  
   - **MODERATE** — monitor / caution / may affect patterns  
   - **LOW** — relevant text present but no HIGH/MODERATE match  

Both **Ask** and **Medication Checker** use this same signal for UI colors.

---

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Web UI |
| `GET` | `/api/examples` | Example Ask questions |
| `POST` | `/api/ask` | Ask a Question |
| `POST` | `/api/medications/resolve` | Resolve one med (brand/generic) |
| `POST` | `/api/medications/prepare` | Prepare multi-med check list |
| `POST` | `/api/medications/check-pair` | Check one drug pair |

---

## Rebuild the pipeline (optional)

Only needed if you change raw data, target drugs, chunking, or embeddings.

From the project root (with `src` on `PYTHONPATH` or run modules from `src` as your project already does):

```bash
# 1) Filter openFDA labels → data/processed/drugs_filtered.json
python src/ingest.py

# 2) Brand mapping → data/processed/brand_mapping.json
python src/mapping.py

# 3) Chunk → data/processed/chunks.json
python src/chunk.py

# 4) Embed + FAISS → data/index/
python src/embed.py
```

Then restart:

```bash
python app.py
```

---

## Evaluation

```bash
python src/evaluate.py
```

- Questions: `eval/test_questions.json` (30 items)  
- Categories: single-drug warnings/dosage, two-drug interaction, brand name, out-of-scope  
- Output: `eval/results.csv` (pass/fail + optional RAGAS / LLM-as-judge scores)

---

## Dependencies

See `requirements.txt`:

- `ijson`, `langchain` / `langchain-text-splitters`
- `faiss-cpu`, `sentence-transformers`
- `openai`, `python-dotenv`
- `fastapi`, `uvicorn`, `jinja2`
- `pandas`, `ragas`

Python **3.10+** recommended.

---

## Repository layout

```text
MedSafe RAG/
├── app.py                 # FastAPI entrypoint
├── web_service.py         # Ask + Checker API layer
├── requirements.txt
├── .env                   # OPENAI_API_KEY 
├── templates/index.html
├── static/js/app.js
├── static/css/style.css
├── src/
│   ├── ingest.py
│   ├── mapping.py
│   ├── chunk.py
│   ├── embed.py
│   ├── generate.py
│   └── evaluate.py
├── data/
│   ├── raw/               # openFDA label dumps
│   ├── processed/         # filtered labels, chunks, brand map
│   └── index/             # FAISS + metadata
└── eval/
    └── test_questions.json
```

---