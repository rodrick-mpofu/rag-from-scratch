# RAG From Scratch

An evaluation-driven Retrieval-Augmented Generation project built
to understand and experiment with the components of a RAG system
without hiding the core pipeline behind high-level frameworks.

## Motivation

The goal of this project is not simply to build a "chat with your
PDF" application.

Instead, the project explores how changes to retrieval,
chunking, ranking, and query processing affect RAG quality.

The development approach is:

Baseline → Evaluate → Analyze Failures → Change One Component →
Evaluate Again

## Current Pipeline

Documents
    ↓
Text extraction
    ↓
Sentence-aware chunking
    ↓
Embeddings
    ↓
ChromaDB

User Query
    ↓
Retriever
    ↓
Top-k document chunks
    ↓
Context construction
    ↓
Qwen3 4B via Ollama
    ↓
Grounded response

## Current Stack

- Python 3.12
- uv
- ChromaDB
- Sentence Transformers
- Ollama
- Qwen3 4B
- Hydra
- FastAPI
- pytest

## Retrieval Experiments

### Dense Retrieval Baseline

The initial implementation uses `all-MiniLM-L6-v2` embeddings
with ChromaDB semantic search.

One observed failure involved the query:

> What is a natural parameter?

The correct definition existed in the corpus but was not ranked
inside the top retrieved chunks.

Expanding the query to include "definition" and
"exponential family distribution" caused the relevant chunk to
rank first.

This motivated comparison against lexical retrieval.

### BM25

Status: In progress

BM25 will be evaluated against the same corpus, chunking strategy,
questions, and top-k values as dense retrieval.

Initial metrics:

- Hit@k
- Mean Reciprocal Rank (MRR)

## Evaluation Philosophy

RAG failures are separated into:

1. Retrieval failure
2. Context insufficiency
3. Generation failure

A generated answer is not considered evidence that retrieval is
working correctly.

Retrieval is evaluated independently before generation quality.

## Project Structure

[add tree here]

## Running the Project

Install dependencies:

```bash
uv sync