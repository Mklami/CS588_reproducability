#!/usr/bin/env python3
import os
import json
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.generators.llm_generator import CommentGenerator as LLMGenerator
from src.matchers.comment_matcher import compare_and_merge_all_comments
from src.evaluators.semantic_evaluator import evaluate_comments as evaluate_semantic
from src.evaluators.rouge_evaluator import evaluate_rouge
from src.evaluators.jaccard_evaluator import evaluate_precision
from src.evaluators.readability_evaluator import evaluate_readability

class Pipeline:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the evaluation pipeline."""
        # Get the project root directory (where run.py is located)
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load config and create directories before setting up logging
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.setup_logging()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        config_file = os.path.join(self.root_dir, config_path)
        with open(config_file) as f:
            return json.load(f)

    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            'data/raw',
            'data/interim',
            'data/processed',
            'logs'
        ]
        
        # Create each directory with absolute paths
        for dir_path in directories:
            abs_path = os.path.join(self.root_dir, dir_path)
            os.makedirs(abs_path, exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(self.root_dir, 'logs')
        log_file = os.path.join(log_dir, f'pipeline_{timestamp}.log')

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('Pipeline')

    def generate_llm_comments(self):
        """Generate comments using Code Llama."""
        self.logger.info("Generating comments using Code Llama...")
        input_file = os.path.join(self.root_dir, self.config['input_methods_file'])
        output_file = os.path.join(self.root_dir, 'data', 'interim', 'llm_comments.csv')
        
        llm_generator = LLMGenerator(self.config['llm_model'])
        llm_generator.process_methods_file(input_file, output_file)
        return output_file

    def match_comments(self, llm_file: str):
        """Match and align comments from all sources."""
        self.logger.info("Matching comments from all sources...")
        output_file = os.path.join(self.root_dir, 'data', 'interim', 'matched_comments.csv')
        
        roslyn_file = os.path.join(self.root_dir, self.config['roslyn_comments_file'])
        original_file = os.path.join(self.root_dir, self.config['original_comments_file'])
        
        result_df, _ = compare_and_merge_all_comments(
            llm_file,
            roslyn_file,
            original_file,
            output_file
        )
        return output_file

    def run_evaluations(self, matched_comments_file: str):
        """Run all evaluation metrics."""
        self.logger.info("Running evaluations...")
        results_dir = os.path.join(self.root_dir, 'data', 'processed')
        
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
