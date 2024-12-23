import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class CommentPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_xml_tags(self, text):
        """
        Remove XML tags and standardize the text.
        """
        if pd.isna(text):
            return ""
            
        text = str(text)
        text = re.sub(r'<summary>|</summary>|<[^>]+>', '', text)
        text = ' '.join(text.split())
        return text.strip()
    
    def preprocess(self, text, verbose=False):
        """
        Preprocess the comment text and return a list of tokens.
        """
        if pd.isna(text):
            return []
            
        if verbose:
            print("\nOriginal text:", text)
            
        text = self.clean_xml_tags(text)
        if verbose:
            print("After XML removal:", text)
            
        if not text.strip():
            return []
            
        text = text.lower()
        if verbose:
            print("After lowercase:", text)
            
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        if verbose:
            print("After special char removal:", text)
            
        tokens = word_tokenize(text)
        if verbose:
            print("After tokenization:", tokens)
            
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token.strip()]
        if verbose:
            print("After stopword removal and lemmatization:", tokens)
            
        return tokens

def get_ngrams(tokens, n):
    """Generate n-grams from tokens."""
    return list(ngrams(tokens, n))

def calculate_rouge_n(candidate_tokens, reference_tokens, n):
    """
    Calculate ROUGE-N score.
    Returns precision, recall, and F1 score.
    """
    if not candidate_tokens or not reference_tokens:
        return 0.0, 0.0, 0.0
        
    candidate_ngrams = get_ngrams(candidate_tokens, n)
    reference_ngrams = get_ngrams(reference_tokens, n)
    
    if not candidate_ngrams or not reference_ngrams:
        return 0.0, 0.0, 0.0
    
    # Convert to sets for intersection
    candidate_ngrams_set = set(candidate_ngrams)
    reference_ngrams_set = set(reference_ngrams)
    
    # Calculate matches
    matches = len(candidate_ngrams_set.intersection(reference_ngrams_set))
    
    # Calculate precision and recall
    precision = matches / len(candidate_ngrams) if candidate_ngrams else 0
    recall = matches / len(reference_ngrams) if reference_ngrams else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate_rouge(input_file, output_file, n_values=[1, 2], show_examples=3):
    """
    Evaluate using ROUGE-N metrics.
    """
    try:
        print("\nReading matched comments file...")
        df = pd.read_csv(input_file)
        
        preprocessor = CommentPreprocessor()
        
        # Show preprocessing examples
        print("\nPreprocessing Examples:")
        for i in range(min(show_examples, len(df))):
            print(f"\n=== Example {i+1} ===")
            print("\nLLM Comment:")
            preprocessor.preprocess(df.iloc[i]['LLM_Comment'], verbose=True)
            print("\nNon-LLM Comment:")
            preprocessor.preprocess(df.iloc[i]['NonLLM_Comment'], verbose=True)
            print("\nOriginal Comment:")
            preprocessor.preprocess(df.iloc[i]['Original_Comment'], verbose=True)
        
        results = {}
        
        print("\nCalculating ROUGE scores...")
        for n in n_values:
            results[f'ROUGE-{n}'] = {
                'LLM': {'precision': [], 'recall': [], 'f1': []},
                'NonLLM': {'precision': [], 'recall': [], 'f1': []}
            }
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'ROUGE-{n}'):
                # Preprocess comments
                llm_tokens = preprocessor.preprocess(row['LLM_Comment'])
                nonllm_tokens = preprocessor.preprocess(row['NonLLM_Comment'])
                original_tokens = preprocessor.preprocess(row['Original_Comment'])
                
                # Calculate ROUGE scores
                llm_scores = calculate_rouge_n(llm_tokens, original_tokens, n)
                nonllm_scores = calculate_rouge_n(nonllm_tokens, original_tokens, n)
                
                # Store results
                results[f'ROUGE-{n}']['LLM']['precision'].append(llm_scores[0])
                results[f'ROUGE-{n}']['LLM']['recall'].append(llm_scores[1])
                results[f'ROUGE-{n}']['LLM']['f1'].append(llm_scores[2])
                
                results[f'ROUGE-{n}']['NonLLM']['precision'].append(nonllm_scores[0])
                results[f'ROUGE-{n}']['NonLLM']['recall'].append(nonllm_scores[1])
                results[f'ROUGE-{n}']['NonLLM']['f1'].append(nonllm_scores[2])
        
        # Add results to dataframe
        for n in n_values:
            for system in ['LLM', 'NonLLM']:
                for metric in ['precision', 'recall', 'f1']:
                    col_name = f'ROUGE-{n}_{system}_{metric}'
                    df[col_name] = results[f'ROUGE-{n}'][system][metric]
        
        # Calculate statistics
        stats = {}
        for n in n_values:
            stats[f'ROUGE-{n}'] = {}
            for system in ['LLM', 'NonLLM']:
                stats[f'ROUGE-{n}'][system] = {
                    metric: {
                        'Mean': np.mean(results[f'ROUGE-{n}'][system][metric]),
                        'Median': np.median(results[f'ROUGE-{n}'][system][metric]),
                        'Std Dev': np.std(results[f'ROUGE-{n}'][system][metric])
                    }
                    for metric in ['precision', 'recall', 'f1']
                }
        
        # Create visualizations
        for n in n_values:
            plt.figure(figsize=(12, 6))
            metrics = ['precision', 'recall', 'f1']
            x = np.arange(len(metrics))
            width = 0.35
            
            llm_means = [np.mean(results[f'ROUGE-{n}']['LLM'][m]) for m in metrics]
            nonllm_means = [np.mean(results[f'ROUGE-{n}']['NonLLM'][m]) for m in metrics]
            
            plt.bar(x - width/2, llm_means, width, label='LLM')
            plt.bar(x + width/2, nonllm_means, width, label='NonLLM')
            
            plt.ylabel('Score')
            plt.title(f'ROUGE-{n} Scores')
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(f'rouge_{n}_scores.png')
            plt.close()
        
        # Save results
        df.to_csv(output_file, index=False)
        
        # Print summary statistics
        print("\nROUGE Score Statistics:")
        for n in n_values:
            print(f"\nROUGE-{n}:")
            for system in ['LLM', 'NonLLM']:
                print(f"\n{system}:")
                for metric in ['precision', 'recall', 'f1']:
                    stats_dict = stats[f'ROUGE-{n}'][system][metric]
                    print(f"\n{metric.capitalize()}:")
                    for stat_name, value in stats_dict.items():
                        print(f"{stat_name}: {value:.3f}")
        
        print(f"\nResults saved to: {output_file}")
        print("Visualizations saved as rouge_N_scores.png")
        
        return df, stats
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None, None

if __name__ == "__main__":
    input_file = "matched_comments_with_original.csv"
    output_file = "rouge_evaluation_results.csv"
    
    results, statistics = evaluate_rouge(input_file, output_file)