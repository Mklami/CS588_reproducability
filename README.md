# Automated Legacy Code Comments: LLM vs Traditional Methods

This repository contains the implementation and evaluation code for comparing Large Language Models (Code Llama) against traditional approaches (Roslyn Analyzer) in generating code documentation.

## Overview

This project evaluates different approaches to automated code documentation generation:
- LLM-based generation using Code Llama via Ollama
- Traditional generation using Microsoft's Roslyn Analyzer
- Comparison against original developer comments

## Repository Structure
├── data/                      # Data directory
│   ├── raw/                  # Original method files
│   └── processed/            # Processed and matched comments
├── src/                      # Source code
│   ├── comment_generation/   # Code for generating comments
│   │   ├── llm/             # Code Llama implementation
│   │   └── roslyn/          # Roslyn Analyzer implementation
│   ├── evaluation/          # Evaluation metrics implementation
│   │   ├── semantic/        # Semantic similarity evaluation
│   │   ├── precision/       # Jaccard similarity calculation
│   │   ├── recall/          # ROUGE-N score calculation
│   │   └── readability/     # Readability metrics
│   └── utils/               # Utility functions
└── notebooks/               # Analysis notebooks

## Features

- Comment generation using Code Llama (LLM approach)
- Comment generation using Roslyn Analyzer
- Evaluation metrics implementation:
  - Semantic Similarity using BERT embeddings
  - Precision using Jaccard Similarity
  - Recall using ROUGE-N scores
  - Readability assessment (Flesch-Kincaid and SMOG)
- Data preprocessing and analysis utilities
- Visualization tools for results

## Requirements

- Python 3.8+
- Ollama with Code Llama model
- .NET SDK (for Roslyn Analyzer)
- Required Python packages:
transformers
torch
nltk
numpy
pandas
matplotlib

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Mklami/CS588_reproducability.git
```
2. Install dependencies:
   pip install -r requirements.txt

3. Set up Ollama with Code Llama:
   ollama pull codellama

4. Run the evaluation:
   python src/main.py


