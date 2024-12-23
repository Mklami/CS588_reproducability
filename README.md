# Automated Legacy Code Comments Evaluation

This repository contains the implementation and evaluation pipeline for the paper "Automated Legacy Code Comments: Evaluating Large Language Models vs. Traditional Methods". The pipeline compares comments generated by Code Llama against traditional methods (Roslyn Analyzer) and original developer comments.

## Prerequisites

- Python 3.8+
- Ollama (for Code Llama)
- NVIDIA GPU recommended but not required
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd automated-legacy-code-comments
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama and the Code Llama model:
```bash
# Install Ollama from https://ollama.ai/
ollama pull codellama
```

5. Required Manual Setup:
```bash
# Create logs directory
mkdir -p logs

# Create __init__.py files in each directory
touch src/__init__.py
touch src/generators/__init__.py
touch src/evaluators/__init__.py
touch src/evaluators/precision/__init__.py
touch src/evaluators/readability/__init__.py
touch src/evaluators/recall/__init__.py
touch src/evaluators/semantic/__init__.py
touch src/matchers/__init__.py
```

## Repository Structure

```
reproducible-code-comments/
├── README.md
├── requirements.txt
├── config.json                 # Configuration file
├── run.py                     # Main pipeline script
├── src/
│   ├── generators/
│   │   └── llm_generator.py   # Code Llama integration
│   ├── matchers/
│   │   └── comment_matcher.py # Comment alignment
│   └── evaluators/            # Evaluation metrics
│       ├── precision/
│       │   └── jaccard_evaluator.py
│       ├── readability/
│       │   └── readability_evaluator.py
│       ├── recall/
│       │   └── rouge_evaluator.py
│       └── semantic/
│           └── semantic_evaluator.py
├── data/
│   ├── raw/                   # Input data
│   ├── interim/              # Intermediate results
│   └── processed/           # Final results
└── logs/                    # Pipeline logs
```

## Data Preparation

Place your input files in the `data/raw/` directory:
1. `500_methods.csv` - Methods for comment generation
2. `500_methods_and_summaries.csv` - Original developer comments
3. `500_comment_generated_nonllm.csv` - Pre-generated Roslyn comments

## Configuration

The `config.json` file contains all configurable parameters:
```json
{
    "input_methods_file": "data/raw/500_methods.csv",
    "original_comments_file": "data/raw/500_methods_and_summaries.csv",
    "roslyn_comments_file": "data/raw/500_comment_generated_nonllm.csv",
    "llm_model": "codellama"
}
```

## Running the Pipeline

1. Ensure all input files are in place
2. Run the pipeline:
```bash
python run.py
```

The pipeline will:
1. Generate comments using Code Llama
2. Match comments from all sources
3. Run evaluations (semantic similarity, ROUGE, Jaccard, readability)
4. Save results in `data/processed/`

## Output Files

The pipeline generates several output files in `data/processed/`:
- `semantic_results.csv` - Semantic similarity evaluation
- `rouge_results.csv` - ROUGE scores
- `jaccard_results.csv` - Jaccard similarity results
- `readability_results.csv` - Readability metrics
- `evaluation_stats.json` - Aggregated statistics

## Logging

Pipeline logs are saved in the `logs` directory with timestamp-based filenames.
