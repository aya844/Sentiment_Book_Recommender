# Semantic Book Recommender

A content-based book recommendation system powered by LLMs and semantic search,
built on a dataset of 5,000+ books using LangChain as the core orchestration framework.

## Overview

This project builds an end-to-end book recommendation pipeline that combines:
- **Text classification** using zero-shot learning (facebook/bart-large-mnli)
- **Semantic search** using LangChain + Sentence Transformers embeddings
- **Vector storage** with ChromaDB via LangChain
- **LLM-powered recommendations** with natural language explanations

## Tech Stack

| Tool | Usage |
|------|-------|
| Python / Pandas | Data processing & cleaning |
| HuggingFace Transformers | Zero-shot text classification (BART) |
| LangChain | Document loading, splitting & RAG pipeline |
| Sentence Transformers (all-MiniLM-L6-v2) | Semantic embeddings |
| ChromaDB | Vector store for similarity search |

## Key LangChain Components

- **TextLoader** — Load book descriptions as LangChain Documents
- **CharacterTextSplitter** — Chunk long descriptions for better embedding quality
- **HuggingFaceEmbeddings** — Generate dense vectors locally (no API cost)
- **Chroma** — Persist and query the vector store

## Dataset

**Source:** [Kaggle - 7K Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)

- ~5,200 books with title, author, description, category, ratings
- 479 raw categories consolidated into 4 simplified labels:
  `Fiction`, `Nonfiction`, `Children's Fiction`, `Children's Nonfiction`
- Missing categories (~1,454 books) predicted via zero-shot classification
  achieving **~87% accuracy**

## How It Works

1. **Data Cleaning** — Normalize and consolidate book categories
2. **Category Prediction** — Use `bart-large-mnli` to classify books
   with missing categories
3. **Document Loading** — Load book descriptions via LangChain `TextLoader`
4. **Text Splitting** — Chunk descriptions with `CharacterTextSplitter`
5. **Embedding Generation** — Encode chunks using `all-MiniLM-L6-v2` locally
6. **Vector Storage** — Persist embeddings in ChromaDB via LangChain
7. **Semantic Search** — Retrieve top-k similar books for a user query
8. **LLM Ranking** — Re-rank and explain recommendations in natural language

## Getting Started

```bash
git clone 
cd llm-semantic-book-recommender
pip install -r requirements.txt
jupyter notebook
```


## Results

- Zero-shot classifier accuracy: **~87%** on Fiction/Nonfiction distinction
- Semantic search returns contextually relevant results beyond keyword matching
- Embeddings generated fully **locally** — no external API cost

## Key Learnings

- Zero-shot NLP classification without labeled training data
- Building RAG-style pipelines with LangChain from scratch
- Local embedding generation with Sentence Transformers
- Vector databases (ChromaDB) for scalable semantic search
