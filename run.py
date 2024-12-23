#!/usr/bin/env python3
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

from src.generators.llm.llm_generator import CommentGenerator as LLMGenerator
from src.matchers.comment_matcher import compare_and_merge_all_comments
from src.evaluators.semantic.semantic_evaluator import evaluate_comments as evaluate_semantic
from src.evaluators.recall.rouge_evaluator import evaluate_rouge
from src.evaluators.precision.jaccard_evaluator import evaluate_precision
from src.evaluators.readability.readability_evaluator import evaluate_readability

class Pipeline:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the evaluation pipeline."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        with open(config_path) as f:
            return json.load(f)
    
    def setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/pipeline_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('Pipeline')
    
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            'data/raw',
            'data/interim',
            'data/processed',
            'logs'
        ]
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def generate_llm_comments(self):
        """Generate comments using Code Llama."""
        self.logger.info("Generating comments using Code Llama...")
        input_file = self.config['input_methods_file']
        output_file = os.path.join('data/interim', 'llm_comments.csv')
        
        llm_generator = LLMGenerator(self.config['llm_model'])
        llm_generator.process_methods_file(input_file, output_file)
        return output_file
    
    def match_comments(self, llm_file: str):
        """Match and align comments from all sources."""
        self.logger.info("Matching comments from all sources...")
        output_file = os.path.join('data/interim', 'matched_comments.csv')
        
        result_df, _ = compare_and_merge_all_comments(
            llm_file,
            self.config['roslyn_comments_file'],  # Pre-existing Roslyn comments
            self.config['original_comments_file'],
            output_file
        )
        return output_file
    
    def run_evaluations(self, matched_comments_file: str):
        """Run all evaluation metrics."""
        self.logger.info("Running evaluations...")
        results_dir = 'data/processed'
        
        # Semantic similarity evaluation
        self.logger.info("Running semantic similarity evaluation...")
        semantic_df, semantic_stats = evaluate_semantic(
            matched_comments_file,
            os.path.join(results_dir, 'semantic_results.csv')
        )
        
        # ROUGE evaluation
        self.logger.info("Running ROUGE evaluation...")
        rouge_df, rouge_stats = evaluate_rouge(
            matched_comments_file,
            os.path.join(results_dir, 'rouge_results.csv')
        )
        
        # Jaccard similarity evaluation
        self.logger.info("Running Jaccard similarity evaluation...")
        jaccard_df, jaccard_stats = evaluate_precision(
            matched_comments_file,
            os.path.join(results_dir, 'jaccard_results.csv')
        )
        
        # Readability evaluation
        self.logger.info("Running readability evaluation...")
        readability_df, readability_stats = evaluate_readability(
            matched_comments_file,
            os.path.join(results_dir, 'readability_results.csv')
        )
        
        # Save aggregated statistics
        stats = {
            'semantic': semantic_stats,
            'rouge': rouge_stats,
            'jaccard': jaccard_stats,
            'readability': readability_stats
        }
        
        with open(os.path.join(results_dir, 'evaluation_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    
    def run_pipeline(self):
        """Run the complete evaluation pipeline."""
        try:
            # Step 1: Generate LLM comments
            llm_file = self.generate_llm_comments()
            
            # Step 2: Match comments with existing files
            matched_file = self.match_comments(llm_file)
            
            # Step 3: Run evaluations
            self.run_evaluations(matched_file)
            
            self.logger.info("Pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Run the code comment evaluation pipeline')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    pipeline = Pipeline(args.config)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()