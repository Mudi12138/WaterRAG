# WaterRAG

A RAG system for wastewater treatment research. The knowledge base includes 7,637 peer-reviewed papers and 11 engineering references.

## What it does

- **Q&A over literature** — ask questions, get answers grounded in actual wastewater treatment papers
- **Literature review generation** — iteratively builds comprehensive reviews with self-auditing (useful for getting a quick survey of a topic)

## Installation
```bash
pip install python-dotenv openai requests langchain langchain-community faiss-cpu sentence-transformers
```

Use `faiss-gpu` instead if you have a GPU.

## Setup

1. Make sure [Git LFS](https://git-lfs.github.com/) is installed, then clone the repo — the index in `0520_256/` will be pulled automatically:
```bash
git lfs install
git clone https://github.com/your-username/waterrag.git
```

2. Copy the example env file and fill it in:
```bash
cp .env.example .env
```
```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_API_URL=https://api.openai.com/v1/chat/completions
INDEX_PATH=./0520_256
DEFAULT_GENERATION_MODEL=gpt-4.1
DEFAULT_RERANK_MODEL=gpt-4.1-mini
DEFAULT_CHUNK_TOP_K=20
DEFAULT_FINAL_TOP_K=10
DEFAULT_MAX_TOKENS=4096
DEFAULT_MIN_SCORE=0.5
RERANKER_BATCH_SIZE=8
```

## Usage

### CLI
```bash
python waterrag_app.py
```

### In code

**Basic Q&A:**
```python
from retrieval_simplified import RetrievalSystem

system = RetrievalSystem()  # reads config from .env
answer, docs = system.retrieve_and_answer(
    "What are effective nitrogen removal techniques?"
)
print(answer)
```

**Generate a literature review:**
```python
from iterative_review_simplified import IterativeReviewSystem

review_system = IterativeReviewSystem()
review, stats = review_system.generate_iterative_review(
    user_question="Nitrogen removal in wastewater treatment",
    max_iterations=3,
    min_score_threshold=7,
    enable_expansion=True
)

print(review)
print(f"Quality score: {stats['final_score']}/10")
```

## Configuration reference

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI key | required |
| `OPENAI_API_URL` | API endpoint | `https://api.openai.com/v1/chat/completions` |
| `INDEX_PATH` | Path to FAISS/BM25 index | required |
| `DEFAULT_GENERATION_MODEL` | Model used for answers | `gpt-4.1` |
| `DEFAULT_RERANK_MODEL` | Model used for reranking | `gpt-4.1-mini` |
| `DEFAULT_CHUNK_TOP_K` | Candidates to retrieve initially | `20` |
| `DEFAULT_FINAL_TOP_K` | How many to keep after reranking | `10` |
| `DEFAULT_MAX_TOKENS` | Max generation tokens | `4096` |
| `DEFAULT_MIN_SCORE` | Minimum reranker score to keep | `0.5` |
| `RERANKER_BATCH_SIZE` | Reranking batch size | `8` |
| `REVIEW_MAX_ITERATIONS` | Max review refinement rounds | `3` |
| `REVIEW_MIN_SCORE_THRESHOLD` | Min score to stop iterating | `7` |
| `REVIEW_INITIAL_CANDIDATES` | Initial candidates for review | `30` |
| `REVIEW_INITIAL_TOP_K` | Initial top-k for review | `20` |
| `REVIEW_SUPPLEMENT_CANDIDATES` | Candidates per aspect (expansion) | `15` |
| `REVIEW_SUPPLEMENT_TOP_K` | Top-k per aspect (expansion) | `10` |
| `REVIEW_ENABLE_EXPANSION` | Enable query expansion | `True` |

## Citation

If you use WaterRAG in your research, please cite:
```bibtex
@software{waterrag2025,
  author       = {Your Name},
  title        = {WaterRAG: A Retrieval-Augmented Generation System for Wastewater Treatment Research},
  year         = {2025},
  url          = {https://github.com/your-username/waterrag}
}
```
